//! FMT*: the fast marching tree (Janson, Schmerling, Clark and Pavone 2015).

use alloc::{collections::BinaryHeap, vec::Vec};

use rand_core::Rng;

use super::{Error, graph::Entry};
use crate::radius::{dimension_f64, shrinking_radius, unit_ball_volume};
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

/// Where a node stands in the march.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Status {
    /// Not connected yet (the set `W`).
    Unvisited,
    /// Connected during the current expansion; joins the open set after it (`H_new`).
    Fresh,
    /// Connected and on the wavefront (the set `H`).
    Open,
    /// Expanded; never touched again.
    Closed,
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

        Ok(March::new(self, states).run(goal))
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

/// The state of one FMT* run over a fixed batch. Node `0` is the start.
struct March<'a, Sp: StateSpace, Sa, C, N> {
    /// The planner, for its space, checker and config.
    planner: &'a FmtStar<Sp, Sa, C, N>,
    /// The batch states, indexed by node id.
    states: Vec<Sp::State>,
    /// Index over all batch states.
    nn: N,
    /// The connection radius for this batch.
    radius: f64,
    /// Neighbors within `radius` per node, excluding the node; filled on first use.
    near: Vec<Option<Vec<usize>>>,
    /// Where each node stands.
    status: Vec<Status>,
    /// Cost-to-come of connected nodes.
    cost: Vec<f64>,
    /// Parent of connected nodes; `None` for the start and unconnected nodes.
    parent: Vec<Option<usize>>,
    /// The open set ordered by cost, then id; closed entries are skipped when popped.
    open: BinaryHeap<Entry>,
}

impl<'a, Sp, Sa, C, N> March<'a, Sp, Sa, C, N>
where
    Sp: StateSpace,
    C: ValidityChecker<Sp>,
    N: NearestNeighbors<Sp> + Clone,
{
    /// Indexes the batch and puts the start on the wavefront.
    fn new(planner: &'a FmtStar<Sp, Sa, C, N>, states: Vec<Sp::State>) -> Self {
        let n = states.len();
        let mut nn = planner.nn.clone();

        for (id, state) in states.iter().enumerate() {
            nn.insert(id, state.clone());
        }

        let d = dimension_f64(planner.space.dimension());
        let mut march = Self {
            planner,
            radius: shrinking_radius(planner.config.gamma, n, d),
            states,
            nn,
            near: alloc::vec![None; n],
            status: alloc::vec![Status::Unvisited; n],
            cost: alloc::vec![f64::INFINITY; n],
            parent: alloc::vec![None; n],
            open: BinaryHeap::new(),
        };
        march.status[0] = Status::Open;
        march.cost[0] = 0.0;
        march.open.push(Entry { cost: 0.0, node: 0 });

        march
    }

    /// Expands the cheapest open node until it lies in `goal` or the open set is empty.
    fn run(mut self, goal: &GoalRegion<Sp::State>) -> Option<Path<Sp::State>> {
        while let Some(Entry { node: z, .. }) = self.open.pop() {
            if self.status[z] == Status::Closed {
                continue;
            }

            if goal.contains(&self.planner.space, &self.states[z]) {
                return Some(self.path_to(z));
            }

            self.expand(z);
        }

        None
    }

    /// Connects each unvisited neighbor of `z` to its cheapest open neighbor, if that motion
    /// is valid, then moves the new nodes to the open set and closes `z`.
    fn expand(&mut self, z: usize) {
        let mut fresh = Vec::new();
        self.ensure_neighbors(z);
        // Taken out for the loop; only neighbor lists of other nodes are needed meanwhile.
        let near_z = self.near[z].take().unwrap_or_default();

        for &x in &near_z {
            if self.status[x] != Status::Unvisited {
                continue;
            }

            let Some((y, cost)) = self.cheapest_open_neighbor(x) else {
                continue;
            };

            // Only the locally best parent is checked; a failure leaves `x` for later waves.
            if self.is_motion_valid(y, x) {
                self.status[x] = Status::Fresh;
                self.cost[x] = cost;
                self.parent[x] = Some(y);
                fresh.push(x);
            }
        }

        self.near[z] = Some(near_z);

        for x in fresh {
            self.status[x] = Status::Open;
            self.open.push(Entry {
                cost: self.cost[x],
                node: x,
            });
        }

        self.status[z] = Status::Closed;
    }

    /// The open neighbor `y` of `x` minimizing `cost(y) + d(y, x)`, lowest id on ties, with that
    /// cost.
    fn cheapest_open_neighbor(&mut self, x: usize) -> Option<(usize, f64)> {
        let mut best: Option<(usize, f64)> = None;
        self.ensure_neighbors(x);

        for &y in self.near[x].as_deref().unwrap_or_default() {
            if self.status[y] != Status::Open {
                continue;
            }

            let cost = self.cost[y]
                + self
                    .planner
                    .space
                    .distance(&self.states[y], &self.states[x]);
            if best.is_none_or(|(_, best_cost)| cost < best_cost) {
                best = Some((y, cost));
            }
        }

        best
    }

    /// Computes the neighbors of `node` within the radius, in id order, unless cached.
    fn ensure_neighbors(&mut self, node: usize) {
        if self.near[node].is_some() {
            return;
        }

        let mut near = Vec::new();
        self.nn.within_radius(
            &self.planner.space,
            &self.states[node],
            self.radius,
            &mut near,
        );
        near.retain(|&other| other != node);
        self.near[node] = Some(near);
    }

    /// Whether the motion from node `from` to node `to` checks as valid.
    fn is_motion_valid(&self, from: usize, to: usize) -> bool {
        self.planner
            .checker
            .check_motion(&self.planner.space, &self.states[from], &self.states[to])
            .is_valid()
    }

    /// The path from the start to the connected node `id`.
    fn path_to(&self, id: usize) -> Path<Sp::State> {
        let mut states = Vec::new();
        let mut current = Some(id);

        while let Some(node) = current {
            states.push(self.states[node].clone());
            current = self.parent[node];
        }

        states.reverse();

        Path {
            states,
            cost: self.cost[id],
        }
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
