//! RRT*: the asymptotically optimal RRT (Karaman and Frazzoli 2011).

use alloc::vec::Vec;

use rand_core::Rng;

use super::{Error, growth::Growth, targets::draw_target};
use crate::{
    LinearScan, NearestNeighbors, Path, Problem, Sampler, StateSpace, Steering, ValidityChecker,
    radius::{
        dimension_f64, k_constant_lower_bound, k_nearest_count, shrinking_radius, unit_ball_volume,
    },
};

/// Settings for [`RrtStar`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RrtStarConfig {
    /// Probability in `[0, 1]` of extending toward the goal center instead of a random sample.
    pub goal_bias: f64,
    /// The connection radius constant: with `n` nodes in a `d`-dimensional space, nodes within
    /// `gamma * (ln n / n)^(1/d)` of a new node are candidate parents and rewiring targets.
    ///
    /// Asymptotic optimality needs `gamma` above [`RrtStarConfig::gamma_lower_bound`].
    /// Positive and finite. Unused when [`RrtStarConfig::k_nearest`] is set.
    pub gamma: f64,
    /// RRT*-k: when `Some(k_constant)`, the candidate parents and rewiring targets of a new node
    /// are its `ceil(k_constant * ln n)` nearest nodes (`n` counting the new node) instead of
    /// the nodes within the `gamma` radius. Neighbors beyond `max_radius` are still dropped.
    ///
    /// Asymptotic optimality needs `k_constant` above
    /// [`RrtStarConfig::k_constant_lower_bound`]. Positive and finite. `None` by default.
    pub k_nearest: Option<f64>,
    /// Upper bound on the connection radius, `eta` in Karaman and Frazzoli. Usually the
    /// steering step, so that no edge is longer than one steering step. Positive; may be
    /// infinite.
    pub max_radius: f64,
    /// Informed sampling (Gammell et al. 2014): once a solution of cost `c` exists, only keep
    /// samples `x` with `d(start, x) + d(x, goal center) < c + goal tolerance`, the states that
    /// can still lie on a cheaper path.
    ///
    /// Samplers that support it draw from the informed set directly
    /// ([`Sampler::sample_informed`]); [`UniformBox`](crate::UniformBox) in a Euclidean space
    /// samples the prolate hyperspheroid around start and goal center whenever it is smaller
    /// than the box. Other samplers fall back to rejection, which works for every [`Sampler`]
    /// and [`StateSpace`] but costs up to [`INFORMED_ATTEMPTS`] sampler calls per iteration when
    /// the informed set is small. Goal bias still applies. Until the first solution the planner
    /// behaves exactly as without it. Off by default.
    pub informed: bool,
}

/// The most samples a rejection-sampled informed iteration draws before it settles for an
/// uninformed one.
pub const INFORMED_ATTEMPTS: usize = 64;

impl RrtStarConfig {
    /// Creates a config with the given radius constant and radius cap, and a goal bias of 5 %.
    #[must_use]
    pub const fn new(gamma: f64, max_radius: f64) -> Self {
        Self {
            goal_bias: 0.05,
            gamma,
            k_nearest: None,
            max_radius,
            informed: false,
        }
    }

    /// An RRT*-k config: `ceil(k_constant * ln n)` nearest neighbors, capped at `max_radius`,
    /// and a goal bias of 5 %.
    #[must_use]
    pub const fn k_nearest(k_constant: f64, max_radius: f64) -> Self {
        Self {
            k_nearest: Some(k_constant),
            ..Self::new(1.0, max_radius)
        }
    }

    /// The smallest k-nearest constant for which RRT*-k is asymptotically optimal (Karaman and
    /// Frazzoli 2011, Theorem 39): `2^(d + 1) e (1 + 1/d)`. Unlike the radius bound it does not
    /// depend on the free volume.
    ///
    /// This is `2^(d + 1)` times the k-nearest PRM* and RRG bound `e (1 + 1/d)` (Theorems 35 and
    /// 37), which OMPL uses for RRT* too and which works well in practice; the theorem only
    /// guarantees the larger one.
    #[must_use]
    pub fn k_constant_lower_bound(dimension: usize) -> f64 {
        libm::pow(2.0, dimension_f64(dimension) + 1.0) * k_constant_lower_bound(dimension)
    }

    /// This config with [`RrtStarConfig::informed`] switched on.
    #[must_use]
    pub const fn informed(self) -> Self {
        Self {
            informed: true,
            ..self
        }
    }

    /// Checks `gamma` (or the k-nearest constant) and `max_radius`; the goal bias is checked by
    /// the tree.
    pub(super) fn validate(&self) -> Result<(), Error> {
        let constant = self.k_nearest.unwrap_or(self.gamma);
        if !(constant.is_finite() && constant > 0.0) {
            return Err(Error::InvalidGamma);
        }

        if self.max_radius.is_nan() || self.max_radius <= 0.0 {
            return Err(Error::InvalidMaxRadius);
        }

        Ok(())
    }

    /// Replaces `out` with the neighbors of a new node at `state` in a graph that will have `n`
    /// nodes: within `min(gamma (ln n / n)^(1/d), max_radius)`, or the `ceil(k ln n)` nearest
    /// within `max_radius`. `state_of` looks up stored states by id.
    pub(super) fn neighbors<'a, Sp, N>(
        &self,
        space: &Sp,
        nn: &N,
        state: &Sp::State,
        n: usize,
        state_of: impl Fn(usize) -> &'a Sp::State,
        out: &mut Vec<usize>,
    ) where
        Sp: StateSpace,
        Sp::State: 'a,
        N: NearestNeighbors<Sp>,
    {
        let Some(k_constant) = self.k_nearest else {
            let radius = shrinking_radius(self.gamma, n, dimension_f64(space.dimension()));
            nn.within_radius(space, state, radius.min(self.max_radius), out);
            return;
        };

        nn.k_nearest(space, state, k_nearest_count(k_constant, n), out);

        if self.max_radius.is_finite() {
            out.retain(|&id| space.distance(state_of(id), state) <= self.max_radius);
        }
    }

    /// The smallest `gamma` for which RRT* is asymptotically optimal (Karaman and Frazzoli
    /// 2011, Theorem 38): `(2 (1 + 1/d))^(1/d) (free_volume / zeta_d)^(1/d)`, where `zeta_d` is
    /// the volume of the unit `d`-ball.
    ///
    /// `free_volume` is the Lebesgue measure of the obstacle-free space; the volume of the
    /// sampling box is a safe over-estimate. Pick `gamma` somewhat above this value.
    #[must_use]
    pub fn gamma_lower_bound(dimension: usize, free_volume: f64) -> f64 {
        let d = dimension_f64(dimension);

        libm::pow(
            2.0 * (1.0 + 1.0 / d) * free_volume / unit_ball_volume(d),
            1.0 / d,
        )
    }
}

/// RRT*: like [`Rrt`](super::Rrt), but each new node picks the cheapest valid parent among its
/// neighbors and then rewires neighbors through itself when that is cheaper.
///
/// Costs only ever decrease, so the cost of [`RrtStar::best_path`] never increases as
/// iterations run, and it converges to the optimum as the number of iterations grows.
#[derive(Clone)]
pub struct RrtStar<Sp: StateSpace, Sa, C, St, N = LinearScan<<Sp as StateSpace>::State>> {
    /// The tree and problem.
    growth: Growth<Sp, Sa, C, St, N>,
    /// The validated settings.
    config: RrtStarConfig,
    /// Reused buffer for neighbor queries, to avoid an allocation per iteration.
    near: Vec<usize>,
}

impl<Sp, Sa, C, St> RrtStar<Sp, Sa, C, St>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    St: Steering<Sp>,
{
    /// Creates a planner with a [`LinearScan`] nearest-neighbor index.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidStart`] if the start state is not valid.
    /// - [`Error::GoalBiasOutOfRange`] if the goal bias is not in `[0, 1]`.
    /// - [`Error::InvalidGamma`] if `gamma` is not positive and finite while no k-nearest
    ///   constant is set, or the k-nearest constant is not positive and finite.
    /// - [`Error::InvalidMaxRadius`] if `max_radius` is not positive.
    pub fn new(problem: Problem<Sp, Sa, C, St>, config: RrtStarConfig) -> Result<Self, Error> {
        Self::with_nearest_neighbors(problem, config, LinearScan::default())
    }
}

impl<Sp, Sa, C, St, N> RrtStar<Sp, Sa, C, St, N>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    St: Steering<Sp>,
    N: NearestNeighbors<Sp>,
{
    /// Creates a planner with the given empty nearest-neighbor index.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidStart`] if the start state is not valid.
    /// - [`Error::GoalBiasOutOfRange`] if the goal bias is not in `[0, 1]`.
    /// - [`Error::InvalidGamma`] if `gamma` is not positive and finite while no k-nearest
    ///   constant is set, or the k-nearest constant is not positive and finite.
    /// - [`Error::InvalidMaxRadius`] if `max_radius` is not positive.
    pub fn with_nearest_neighbors(
        problem: Problem<Sp, Sa, C, St>,
        config: RrtStarConfig,
        nn: N,
    ) -> Result<Self, Error> {
        config.validate()?;

        Ok(Self {
            growth: Growth::new(problem, config.goal_bias, nn)?,
            config,
            near: Vec::new(),
        })
    }

    /// Runs one iteration. Returns whether a node was added.
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> bool {
        let target = self.draw_target(rng);
        let (nearest, new) = self.growth.propose_toward(&target);
        if !self.growth.is_motion_valid(nearest, &new) {
            return false;
        }

        let mut near = core::mem::take(&mut self.near);
        self.neighbors(&new, &mut near);

        let (parent, cost) = self.choose_parent(nearest, &new, &near);
        let id = self.growth.add(new, parent, cost);
        self.rewire(id, &near);

        self.near = near;
        true
    }

    /// Draws the next growth target: a goal-biased sample, restricted to the informed set when
    /// informed sampling is on and a solution exists.
    fn draw_target<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Sp::State {
        let bound = self.informed_bound();

        draw_target(&mut self.growth.problem, self.growth.goal_bias, bound, rng)
    }

    /// The informed-set bound `c_best + tolerance`, or `None` when informed sampling is off or
    /// no solution exists yet.
    fn informed_bound(&self) -> Option<f64> {
        if !self.config.informed {
            return None;
        }

        let best = self.growth.best_cost()?;

        Some(best + self.growth.problem.goal.tolerance())
    }

    /// Replaces `near` with the candidate parents and rewiring targets of the new node at
    /// `state`.
    fn neighbors(&self, state: &Sp::State, near: &mut Vec<usize>) {
        let tree = &self.growth.tree;

        self.config.neighbors(
            &self.growth.problem.space,
            &self.growth.nn,
            state,
            tree.len() + 1,
            |id| tree.state(id),
            near,
        );
    }

    /// The cheapest parent for `new` among `nearest` and `near`, with its resulting cost.
    ///
    /// `nearest` is known to connect validly. Other candidates are motion-checked only when
    /// they would be cheaper, since checks dominate the run time.
    fn choose_parent(&self, nearest: usize, new: &Sp::State, near: &[usize]) -> (usize, f64) {
        let mut best = (nearest, self.growth.cost_via(nearest, new));

        for &candidate in near {
            let cost = self.growth.cost_via(candidate, new);

            if cost < best.1 && self.growth.is_motion_valid(candidate, new) {
                best = (candidate, cost);
            }
        }

        best
    }

    /// Moves each node in `near` under node `id` when that lowers its cost and the motion is
    /// valid.
    fn rewire(&mut self, id: usize, near: &[usize]) {
        let space = &self.growth.problem.space;

        for &node in near {
            let tree = &self.growth.tree;
            let cost = self.growth.cost_via(id, tree.state(node));

            if cost < tree.cost(node) && self.is_rewire_valid(id, node) {
                self.growth.tree.reparent(space, node, id, cost);
            }
        }
    }

    /// Whether the motion from node `from` to node `to` checks as valid.
    fn is_rewire_valid(&self, from: usize, to: usize) -> bool {
        self.growth
            .is_motion_valid(from, self.growth.tree.state(to))
    }

    /// Runs `iterations` iterations.
    pub fn run<R: Rng + ?Sized>(&mut self, iterations: usize, rng: &mut R) {
        for _ in 0..iterations {
            self.step(rng);
        }
    }

    /// The cheapest path found so far, or `None` if the goal was not reached yet.
    #[must_use]
    pub fn best_path(&self) -> Option<Path<Sp::State>> {
        self.growth.best_path()
    }

    /// The number of tree nodes, including the root.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.growth.tree.len()
    }
}
