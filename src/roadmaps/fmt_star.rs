//! FMT*: the fast marching tree (Janson, Schmerling, Clark and Pavone 2015).

use alloc::vec::Vec;

use rand_core::Rng;

use super::{
    Error,
    march::{Batch, Direction, Wave},
};
use crate::radius::{dimension_f64, unit_ball_volume};
use crate::{GoalRegion, LinearScan, NearestNeighbors, Path, Sampler, StateSpace, ValidityChecker};

/// Settings for [`FmtStar`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FmtStarConfig {
    /// The connection radius constant: with `n` nodes in `d` dimensions, nodes within
    /// `gamma * (ln n / n)^(1/d)` are neighbors.
    ///
    /// Asymptotic optimality needs `gamma` above [`FmtStarConfig::gamma_lower_bound`].
    /// Positive and finite.
    pub gamma: f64,
}

impl FmtStarConfig {
    /// The smallest `gamma` for which FMT* is asymptotically optimal (Janson et al. 2015,
    /// Theorem 4.1, with `eta = 0`): `2 (1/d)^(1/d) (free_volume / zeta_d)^(1/d)`, where
    /// `zeta_d` is the volume of the unit `d`-ball.
    ///
    /// The theorem needs `(1 + eta)` times this for some `eta > 0`, so pick `gamma` somewhat
    /// above it. `free_volume` is the Lebesgue measure of the obstacle-free space; the volume of
    /// the sampling box is a safe over-estimate.
    #[must_use]
    pub fn gamma_lower_bound(dimension: usize, free_volume: f64) -> f64 {
        let d = dimension_f64(dimension);

        2.0 * libm::pow(1.0 / d, 1.0 / d) * libm::pow(free_volume / unit_ball_volume(d), 1.0 / d)
    }
}

/// FMT*: draws a batch of samples, then marches a tree outward from the start in order of
/// cost-to-come, connecting each unvisited neighbor of the cheapest open node to its locally
/// cheapest open neighbor.
///
/// Single-query and asymptotically optimal, like [`RrtStar`](crate::trees::RrtStar), but it
/// checks each motion at most once and only for the locally best parent, so it typically needs
/// far fewer collision checks. Each [`FmtStar::plan`] call draws a fresh batch.
///
/// Edges are direct motions checked in travel direction, parent to child. The goal center is
/// added as a sample when it is valid.
///
/// # Examples
///
/// ```
/// use rand_core::SeedableRng;
/// use sampling_planning::{
///     Euclidean, GoalRegion, UniformBox, Validity, ValidityChecker,
///     roadmaps::{FmtStar, FmtStarConfig},
/// };
///
/// /// Free space everywhere.
/// struct Free;
///
/// impl ValidityChecker<Euclidean<2>> for Free {
///     fn check_state(&self, _: &[f64; 2]) -> Validity {
///         Validity::Valid
///     }
///
///     fn check_motion(&self, _: &Euclidean<2>, _: &[f64; 2], _: &[f64; 2]) -> Validity {
///         Validity::Valid
///     }
/// }
///
/// let sampler = UniformBox::new([0.0, 0.0], [10.0, 10.0]);
/// let config = FmtStarConfig { gamma: 1.5 * FmtStarConfig::gamma_lower_bound(2, 100.0) };
/// let mut fmt = FmtStar::new(Euclidean::<2>, sampler, Free, config).expect("valid");
///
/// let goal = GoalRegion::new([9.0, 9.0], 0.5);
/// let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(1);
/// let path = fmt.plan(&[1.0, 1.0], &goal, 1000, &mut rng).expect("valid start");
///
/// assert!(path.expect("free space").cost < 11.5);
/// ```
#[derive(Debug, Clone)]
pub struct FmtStar<Sp: StateSpace, Sa, C, N = LinearScan<<Sp as StateSpace>::State>> {
    /// The configuration space.
    space: Sp,
    /// Draws the sample batch.
    sampler: Sa,
    /// Decides which states and motions are allowed.
    checker: C,
    /// The settings; `gamma` is positive and finite.
    config: FmtStarConfig,
    /// An empty index, cloned for every batch.
    nn: N,
}

impl<Sp, Sa, C> FmtStar<Sp, Sa, C>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
{
    /// Creates a planner with a [`LinearScan`] nearest-neighbor index.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRadius`] if `gamma` is not positive and finite.
    pub fn new(space: Sp, sampler: Sa, checker: C, config: FmtStarConfig) -> Result<Self, Error> {
        Self::with_nearest_neighbors(space, sampler, checker, config, LinearScan::default())
    }
}

impl<Sp, Sa, C, N> FmtStar<Sp, Sa, C, N>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    N: NearestNeighbors<Sp> + Clone,
{
    /// Creates a planner; every batch gets a clone of the given empty nearest-neighbor index.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRadius`] if `gamma` is not positive and finite.
    pub fn with_nearest_neighbors(
        space: Sp,
        sampler: Sa,
        checker: C,
        config: FmtStarConfig,
        nn: N,
    ) -> Result<Self, Error> {
        if !(config.gamma.is_finite() && config.gamma > 0.0) {
            return Err(Error::InvalidRadius);
        }

        Ok(Self {
            space,
            sampler,
            checker,
            config,
            nn,
        })
    }

    /// Draws `samples` states, keeps the valid ones, and marches from `start` until a node in
    /// `goal` is the cheapest open node.
    ///
    /// Returns `None` if the wavefront dies out first: the batch does not connect start and
    /// goal under the current radius.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidStart`] if `start` does not check as valid.
    pub fn plan<R: Rng + ?Sized>(
        &mut self,
        start: &Sp::State,
        goal: &GoalRegion<Sp::State>,
        samples: usize,
        rng: &mut R,
    ) -> Result<Option<Path<Sp::State>>, Error> {
        let validity = self.checker.check_state(start);
        if !validity.is_valid() {
            return Err(Error::InvalidStart(validity));
        }

        let states = self.draw_batch(start, goal, samples, rng);

        Ok(self.march(states, goal))
    }

    /// Marches one outward wave from node `0` over `states` until the cheapest open node lies
    /// in `goal`, and returns the path to it.
    fn march(
        &self,
        states: Vec<Sp::State>,
        goal: &GoalRegion<Sp::State>,
    ) -> Option<Path<Sp::State>> {
        let mut batch = Batch::new(
            &self.space,
            &self.checker,
            self.config.gamma,
            states,
            self.nn.clone(),
        );
        let mut wave = Wave::new(batch.len(), 0, Direction::Outward);

        while let Some(z) = wave.peek_open() {
            if goal.contains(&self.space, &batch.states[z]) {
                let mut states: Vec<_> = wave
                    .branch(z)
                    .into_iter()
                    .map(|node| batch.states[node].clone())
                    .collect();
                states.reverse();

                return Some(Path {
                    states,
                    cost: wave.cost[z],
                });
            }

            wave.expand(&mut batch, z);
        }

        None
    }

    /// The start, the valid samples and, when valid, the goal center, in that order.
    fn draw_batch<R: Rng + ?Sized>(
        &mut self,
        start: &Sp::State,
        goal: &GoalRegion<Sp::State>,
        samples: usize,
        rng: &mut R,
    ) -> Vec<Sp::State> {
        let mut states = alloc::vec![start.clone()];

        for _ in 0..samples {
            let state = self.sampler.sample(rng);

            if self.checker.check_state(&state).is_valid() {
                states.push(state);
            }
        }

        if self.checker.check_state(goal.center()).is_valid() {
            states.push(goal.center().clone());
        }

        states
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Euclidean, Validity};

    /// Replays a fixed list of states, cycling.
    struct Replay(Vec<[f64; 2]>, usize);

    impl Sampler<[f64; 2]> for Replay {
        fn sample<R: Rng + ?Sized>(&mut self, _: &mut R) -> [f64; 2] {
            self.1 += 1;
            self.0[(self.1 - 1) % self.0.len()]
        }
    }

    /// Free space, except around the state (5, 5).
    struct Hole;

    impl ValidityChecker<Euclidean<2>> for Hole {
        fn check_state(&self, s: &[f64; 2]) -> Validity {
            if libm::hypot(s[0] - 5.0, s[1] - 5.0) < 0.1 {
                Validity::Invalid
            } else {
                Validity::Valid
            }
        }

        fn check_motion(&self, _: &Euclidean<2>, _: &[f64; 2], to: &[f64; 2]) -> Validity {
            self.check_state(to)
        }
    }

    #[test]
    fn hand_checked_batch_stops_at_the_cheapest_goal_node() {
        let sampler = Replay(alloc::vec![[5.0, 5.0], [8.8, 9.0], [3.0, 3.0]], 0);
        let mut fmt = FmtStar::new(
            Euclidean::<2>,
            sampler,
            Hole,
            FmtStarConfig { gamma: 100.0 },
        )
        .expect("valid");
        let goal = GoalRegion::new([9.0, 9.0], 0.5);
        let mut rng = rand_pcg::Pcg64Mcg::new(1);

        // Every node connects straight to the start; (8.8, 9) is the cheapest goal node, and the
        // invalid (5, 5) never enters the batch.
        let path = fmt
            .plan(&[1.0, 1.0], &goal, 3, &mut rng)
            .expect("valid")
            .expect("solved");
        assert_eq!(path.states, [[1.0, 1.0], [8.8, 9.0]]);
        assert!((path.cost - libm::hypot(7.8, 8.0)).abs() < 1e-12);

        assert_eq!(
            fmt.plan(&[5.0, 5.0], &goal, 3, &mut rng),
            Err(Error::InvalidStart(Validity::Invalid))
        );
    }
}
