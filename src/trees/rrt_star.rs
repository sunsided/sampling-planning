//! RRT*: the asymptotically optimal RRT (Karaman and Frazzoli 2011).

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use rand_core::Rng;

use super::{Error, growth::Growth, targets::draw_target};
use crate::{
    CostObjective, LinearScan, NearestNeighbors, Path, PathLength, Problem, RemovableNeighbors,
    Sampler, StateSpace, Steering, ValidityChecker,
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
    /// 37). OMPL's RRT* uses this bound times a rewire factor of 1.1.
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
pub struct RrtStar<
    Sp: StateSpace,
    Sa,
    C,
    St,
    N = LinearScan<<Sp as StateSpace>::State>,
    O = PathLength,
> {
    /// The tree and problem.
    growth: Growth<Sp, Sa, C, St, N>,
    /// The validated settings.
    config: RrtStarConfig,
    /// Prices motions; tree costs are sums of its motion costs.
    objective: O,
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
        Self::with_objective(problem, config, nn, PathLength)
    }
}

impl<Sp, Sa, C, St, N, O> RrtStar<Sp, Sa, C, St, N, O>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    St: Steering<Sp>,
    N: NearestNeighbors<Sp>,
    O: CostObjective<Sp>,
{
    /// Creates a planner with the given empty nearest-neighbor index that minimizes
    /// `objective` instead of path length.
    ///
    /// Neighbors are still found by [`StateSpace::distance`] and the connection radius; only
    /// costs come from [`CostObjective::motion_cost`]. Informed sampling uses the informed set
    /// with bound `c_best / w + tolerance`, `w` = [`CostObjective::distance_weight`], and is
    /// off when `w` is zero. RRT*'s asymptotic optimality is proven for metric costs such as
    /// path length, not for every objective.
    ///
    /// # Errors
    ///
    /// The same as [`RrtStar::with_nearest_neighbors`].
    pub fn with_objective(
        problem: Problem<Sp, Sa, C, St>,
        config: RrtStarConfig,
        nn: N,
        objective: O,
    ) -> Result<Self, Error> {
        config.validate()?;

        Ok(Self {
            growth: Growth::new(problem, config.goal_bias, nn)?,
            config,
            objective,
            near: Vec::new(),
        })
    }

    /// The objective's cost of reaching `to` from node `from` by a direct motion.
    fn cost_via(&self, from: usize, to: &Sp::State) -> f64 {
        let tree = &self.growth.tree;

        tree.cost(from)
            + self
                .objective
                .motion_cost(&self.growth.problem.space, tree.state(from), to)
    }

    /// Runs one iteration. Returns whether a node was added.
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> bool {
        let target = self.draw_target(rng);
        let (nearest, new) = self.growth.propose_toward(&target);
        let valid = self.growth.is_motion_valid(nearest, &new);

        if valid {
            let mut near = core::mem::take(&mut self.near);
            self.neighbors(&new, &mut near);

            let (parent, cost) = self.choose_parent(nearest, &new, &near);
            let id = self.growth.add(new.clone(), parent, cost);
            self.rewire(id, &near);
            self.near = near;
        }

        self.growth.record_extension(nearest, &new, valid);

        valid
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
        let weight = self.objective.distance_weight();

        // A path of cost c through x has d(start, x) + d(x, goal) <= c / w.
        (weight > 0.0).then(|| best / weight + self.growth.problem.goal.tolerance())
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

    #[cfg(feature = "parallel")]
    /// Adds the states of another tree over the same problem, in that tree's insertion order
    /// (parents before children), each like a new RRT* node: connected to the cheapest of its
    /// neighbors with a valid motion, then rewiring them. States without such a neighbor are
    /// dropped. Costs never increase, so the best path never gets worse.
    pub(super) fn absorb(&mut self, states: impl IntoIterator<Item = Sp::State>) {
        let mut near = core::mem::take(&mut self.near);

        for state in states {
            self.neighbors(&state, &mut near);

            if let Some((parent, cost)) = self.cheapest_valid_parent(&state, &near) {
                let id = self.growth.add(state, parent, cost);
                self.rewire(id, &near);
            }
        }

        self.near = near;
    }

    #[cfg(feature = "parallel")]
    /// The neighbor in `near` that reaches `state` most cheaply by a valid motion, with that
    /// cost; motions are checked in order of cost until one is valid.
    fn cheapest_valid_parent(&self, state: &Sp::State, near: &[usize]) -> Option<(usize, f64)> {
        let mut candidates: Vec<(usize, f64)> = near
            .iter()
            .map(|&id| (id, self.cost_via(id, state)))
            .collect();
        candidates.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));

        candidates
            .into_iter()
            .find(|&(id, _)| self.growth.is_motion_valid(id, state))
    }

    #[cfg(feature = "parallel")]
    /// The states of the tree in insertion order, the root first.
    pub(super) fn tree_states(&self) -> Vec<Sp::State> {
        self.growth.tree.states().cloned().collect()
    }

    /// The cheapest parent for `new` among `nearest` and `near`, with its resulting cost.
    ///
    /// `nearest` is known to connect validly. Other candidates are motion-checked only when
    /// they would be cheaper, since checks dominate the run time.
    fn choose_parent(&self, nearest: usize, new: &Sp::State, near: &[usize]) -> (usize, f64) {
        let mut best = (nearest, self.cost_via(nearest, new));

        for &candidate in near {
            let cost = self.cost_via(candidate, new);

            if cost < best.1 && self.growth.is_motion_valid(candidate, new) {
                best = (candidate, cost);
            }
        }

        best
    }

    /// Moves each node in `near` under node `id` when that lowers its cost and the motion is
    /// valid.
    fn rewire(&mut self, id: usize, near: &[usize]) {
        for &node in near {
            let tree = &self.growth.tree;
            let cost = self.cost_via(id, tree.state(node));

            if cost < tree.cost(node) && self.is_rewire_valid(id, node) {
                let (space, objective) = (&self.growth.problem.space, &self.objective);
                let edge_cost = |a: &Sp::State, b: &Sp::State| objective.motion_cost(space, a, b);
                self.growth.tree.reparent(node, id, cost, edge_cost);
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

    /// The number of vertices retired from the nearest-neighbor index; always zero unless
    /// [`RrtStar::with_retirement`] enabled retirement.
    #[must_use]
    pub const fn retired_count(&self) -> usize {
        self.growth.retirement.retired()
    }
}

impl<Sp, Sa, C, St, N, O> RrtStar<Sp, Sa, C, St, N, O>
where
    Sp: StateSpace,
    N: RemovableNeighbors<Sp> + Clone,
{
    /// Retires exhausted vertices from the nearest-neighbor index: with finite steering once
    /// all their successors are in the tree, and in any case after `failures` consecutive
    /// failed extensions; see [the module docs](crate::trees#vertex-retirement).
    ///
    /// RRT* finds parent candidates and rewiring targets in the same index, so a retired vertex
    /// is no longer a parent candidate or rewired. Its cost still drops when an ancestor is
    /// rewired.
    #[must_use]
    pub fn with_retirement(mut self, failures: NonZeroUsize) -> Self {
        let reached = self.growth.nn.clone();
        self.growth.retirement.enable(failures, N::remove, reached);
        self
    }
}
