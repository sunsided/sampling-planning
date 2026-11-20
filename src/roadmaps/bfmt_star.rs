//! BFMT*: the bidirectional fast marching tree (Starek et al. 2015).

use alloc::vec::Vec;

use rand_core::Rng;

use super::{
    Error,
    march::{Batch, Direction, Status, Wave},
};
use crate::radius::{dimension_f64, unit_ball_volume};
use crate::{GoalRegion, LinearScan, NearestNeighbors, Path, Sampler, StateSpace, ValidityChecker};

/// When [`BfmtStar`] stops marching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Termination {
    /// Stop at the first expansion in which the two trees touch, and return the cheapest
    /// connection found in it (Algorithm 1 of Starek et al. as printed). Fastest, but in general
    /// not the cheapest path through the batch.
    FirstMeeting,
    /// Keep marching until the node just expanded in one tree was already expanded in the
    /// other, and return the cheapest connection found by then (the "best path" criterion of
    /// Starek et al., Section III-C, and OMPL's default).
    #[default]
    Optimal,
}

/// Settings for [`BfmtStar`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BfmtStarConfig {
    /// The connection radius constant: with `n` nodes in `d` dimensions, nodes within
    /// `gamma * (ln n / n)^(1/d)` are neighbors.
    ///
    /// Asymptotic optimality needs `gamma` above [`BfmtStarConfig::gamma_lower_bound`].
    /// Positive and finite.
    pub gamma: f64,
    /// When to stop; [`Termination::Optimal`] by default.
    pub termination: Termination,
}

impl BfmtStarConfig {
    /// A config with radius constant `gamma` and [`Termination::Optimal`].
    #[must_use]
    pub const fn new(gamma: f64) -> Self {
        Self {
            gamma,
            termination: Termination::Optimal,
        }
    }

    /// The radius constant of Starek et al. 2015, Theorem 4.1 with `eta = 0`:
    /// `4 (1/d)^(1/d) (free_volume / zeta_d)^(1/d)`, twice FMT*'s.
    ///
    /// The theorem needs `(1 + eta)^(1/d)` times this for some `eta >= 0`, so this value is
    /// already admissible; OMPL uses FMT*'s smaller constant instead. `free_volume` is the
    /// Lebesgue measure of the obstacle-free space; the volume of the sampling box is a safe
    /// over-estimate.
    #[must_use]
    pub fn gamma_lower_bound(dimension: usize, free_volume: f64) -> f64 {
        let d = dimension_f64(dimension);

        4.0 * libm::pow(1.0 / d, 1.0 / d) * libm::pow(free_volume / unit_ball_volume(d), 1.0 / d)
    }
}

/// Which of the two trees an operation works on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    /// The tree rooted at the start; edges traveled away from the root.
    Start,
    /// The tree rooted at the goal center; edges traveled toward the root.
    Goal,
}

/// The cheapest solution found so far.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Connection {
    /// The node where the solution leaves the start tree.
    node: usize,
    /// The total cost.
    cost: f64,
    /// Whether the path continues through the goal tree to the goal center; `false` when
    /// `node` itself lies in the goal region.
    through_goal_tree: bool,
}

/// BFMT*: marches one tree outward from the start and one inward from the goal center over the
/// same batch, alternating expansions, and joins them where they meet (Starek, Gomez,
/// Schmerling, Janson, Moreno and Pavone 2015).
///
/// Each expansion is an FMT* step on the current tree: connect the unvisited neighbors of its
/// cheapest open node `z` to their locally cheapest open neighbor. A newly connected node that
/// the other tree already holds is a connection with cost `cost_start(x) + cost_goal(x)`; the
/// cheapest one is kept. After each expansion the trees swap ("swap every time"), unless the
/// other tree's wavefront is empty. [`Termination`] decides when to stop.
///
/// Goal-tree edges are traveled toward the goal, so they are checked from child to parent,
/// which keeps directional checkers sound. The goal center must be valid. A start-tree node
/// inside the goal region is also a solution, ending there.
///
/// Like [`FmtStar`](super::FmtStar) it is asymptotically optimal and checks each motion at most
/// once; growing two trees from both ends typically settles fewer nodes.
///
/// # Examples
///
/// ```
/// use rand_core::SeedableRng;
/// use sampling_planning::{
///     Euclidean, GoalRegion, UniformBox, Validity, ValidityChecker,
///     roadmaps::{BfmtStar, BfmtStarConfig},
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
/// let config = BfmtStarConfig::new(BfmtStarConfig::gamma_lower_bound(2, 100.0));
/// let mut bfmt = BfmtStar::new(Euclidean::<2>, sampler, Free, config).expect("valid");
///
/// let goal = GoalRegion::new([9.0, 9.0], 0.5);
/// let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(1);
/// let path = bfmt.plan(&[1.0, 1.0], &goal, 1000, &mut rng).expect("valid endpoints");
///
/// assert!(path.expect("free space").cost < 11.5);
/// ```
#[derive(Debug, Clone)]
pub struct BfmtStar<Sp: StateSpace, Sa, C, N = LinearScan<<Sp as StateSpace>::State>> {
    /// The configuration space.
    space: Sp,
    /// Draws the sample batch.
    sampler: Sa,
    /// Decides which states and motions are allowed.
    checker: C,
    /// The settings; `gamma` is positive and finite.
    config: BfmtStarConfig,
    /// An empty index, cloned for every batch.
    nn: N,
}

impl<Sp, Sa, C> BfmtStar<Sp, Sa, C>
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
    pub fn new(space: Sp, sampler: Sa, checker: C, config: BfmtStarConfig) -> Result<Self, Error> {
        Self::with_nearest_neighbors(space, sampler, checker, config, LinearScan::default())
    }
}

impl<Sp, Sa, C, N> BfmtStar<Sp, Sa, C, N>
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
        config: BfmtStarConfig,
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

    /// Draws `samples` states, keeps the valid ones, and marches from both `start` and the
    /// center of `goal` until [`Termination`] says to stop.
    ///
    /// Returns `None` if both wavefronts die out without a connection.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidStart`] if `start` does not check as valid.
    /// - [`Error::InvalidGoal`] if the goal center does not check as valid.
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

        let validity = self.checker.check_state(goal.center());
        if !validity.is_valid() {
            return Err(Error::InvalidGoal(validity));
        }

        if goal.contains(&self.space, start) {
            return Ok(Some(Path {
                states: alloc::vec![start.clone()],
                cost: 0.0,
            }));
        }

        let mut states = alloc::vec![start.clone()];
        states.extend(
            (0..samples)
                .map(|_| self.sampler.sample(rng))
                .filter(|state| self.checker.check_state(state).is_valid()),
        );
        states.push(goal.center().clone());

        Ok(self.march(states, goal))
    }

    /// Marches both trees over `states`: node `0` is the start, the last node the goal center.
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
        let n = batch.len();
        let mut march = Bidirectional {
            start: Wave::new(n, 0, Direction::Outward),
            goal: Wave::new(n, n - 1, Direction::Inward),
            best: None,
        };
        let mut side = Side::Start;

        while let Some(z) = march.wave_mut(side).peek_open() {
            let fresh = march.wave_mut(side).expand(&mut batch, z);
            march.record_connections(&batch, goal, side, &fresh);

            if self.should_stop(&march, side, z) {
                break;
            }

            side = march.next_side(side);
        }

        march.best.map(|best| march.path(&batch, best))
    }

    /// Whether to stop after expanding `z` in the tree on `side`.
    fn should_stop(&self, march: &Bidirectional, side: Side, z: usize) -> bool {
        match self.config.termination {
            Termination::FirstMeeting => march.best.is_some(),
            Termination::Optimal => march.wave(other(side)).status[z] == Status::Closed,
        }
    }
}

/// The two waves of one BFMT* run and the best connection so far.
struct Bidirectional {
    /// The tree rooted at the start.
    start: Wave,
    /// The tree rooted at the goal center.
    goal: Wave,
    /// The cheapest solution found so far.
    best: Option<Connection>,
}

impl Bidirectional {
    /// The wave on `side`.
    const fn wave(&self, side: Side) -> &Wave {
        match side {
            Side::Start => &self.start,
            Side::Goal => &self.goal,
        }
    }

    /// The wave on `side`, mutably.
    const fn wave_mut(&mut self, side: Side) -> &mut Wave {
        match side {
            Side::Start => &mut self.start,
            Side::Goal => &mut self.goal,
        }
    }

    /// The side to expand next: the other one, unless its wavefront is empty.
    fn next_side(&mut self, side: Side) -> Side {
        if self.wave_mut(other(side)).peek_open().is_some() {
            other(side)
        } else {
            side
        }
    }

    /// Records the solutions through the nodes just connected on `side`: nodes held by both
    /// trees, and start-tree nodes inside the goal region.
    fn record_connections<Sp: StateSpace, C, N>(
        &mut self,
        batch: &Batch<'_, Sp, C, N>,
        goal: &GoalRegion<Sp::State>,
        side: Side,
        fresh: &[usize],
    ) {
        for &x in fresh {
            if self.goal.status[x] != Status::Unvisited && self.start.status[x] != Status::Unvisited
            {
                self.offer(x, self.start.cost[x] + self.goal.cost[x], true);
            }

            if side == Side::Start && goal.contains(batch.space, &batch.states[x]) {
                self.offer(x, self.start.cost[x], false);
            }
        }
    }

    /// Keeps the solution through `node` if it is cheaper than the best so far.
    fn offer(&mut self, node: usize, cost: f64, through_goal_tree: bool) {
        if self.best.is_none_or(|best| cost < best.cost) {
            self.best = Some(Connection {
                node,
                cost,
                through_goal_tree,
            });
        }
    }

    /// The states of the solution `best`: the start-tree branch to its node, then, if it goes
    /// through the goal tree, the goal-tree branch to the goal center.
    fn path<Sp: StateSpace, C, N>(
        &self,
        batch: &Batch<'_, Sp, C, N>,
        best: Connection,
    ) -> Path<Sp::State> {
        let mut nodes = self.start.branch(best.node);
        nodes.reverse();

        if best.through_goal_tree {
            nodes.extend(self.goal.branch(best.node).into_iter().skip(1));
        }

        Path {
            states: nodes
                .iter()
                .map(|&node| batch.states[node].clone())
                .collect(),
            cost: best.cost,
        }
    }
}

/// The other side.
const fn other(side: Side) -> Side {
    match side {
        Side::Start => Side::Goal,
        Side::Goal => Side::Start,
    }
}
