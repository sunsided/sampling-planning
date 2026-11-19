//! RRT*: the asymptotically optimal RRT (Karaman and Frazzoli 2011).

use alloc::vec::Vec;
use core::f64::consts::PI;

use rand_core::Rng;

use super::{Error, count_to_f64, growth::Growth};
use crate::{
    LinearScan, NearestNeighbors, Path, Problem, Sampler, StateSpace, Steering, ValidityChecker,
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
    /// Positive and finite.
    pub gamma: f64,
}

impl RrtStarConfig {
    /// Creates a config with the given radius constant and a goal bias of 5 %.
    #[must_use]
    pub const fn new(gamma: f64) -> Self {
        Self {
            goal_bias: 0.05,
            gamma,
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
        let d = count_to_f64(dimension.max(1));
        let unit_ball = libm::pow(PI, d / 2.0) / libm::tgamma(d / 2.0 + 1.0);

        libm::pow(2.0 * (1.0 + 1.0 / d) * free_volume / unit_ball, 1.0 / d)
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
    /// The connection radius constant; positive and finite.
    gamma: f64,
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
    /// - [`Error::InvalidGamma`] if `gamma` is not positive and finite.
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
    /// - [`Error::InvalidGamma`] if `gamma` is not positive and finite.
    pub fn with_nearest_neighbors(
        problem: Problem<Sp, Sa, C, St>,
        config: RrtStarConfig,
        nn: N,
    ) -> Result<Self, Error> {
        if !(config.gamma.is_finite() && config.gamma > 0.0) {
            return Err(Error::InvalidGamma);
        }

        Ok(Self {
            growth: Growth::new(problem, config.goal_bias, nn)?,
            gamma: config.gamma,
            near: Vec::new(),
        })
    }

    /// Runs one iteration. Returns whether a node was added.
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> bool {
        let (nearest, new) = self.growth.propose(rng);
        if !self.growth.is_motion_valid(nearest, &new) {
            return false;
        }

        let mut near = core::mem::take(&mut self.near);
        let space = &self.growth.problem.space;
        self.growth
            .nn
            .within_radius(space, &new, self.radius(), &mut near);

        let (parent, cost) = self.choose_parent(nearest, &new, &near);
        let id = self.growth.add(new, parent, cost);
        self.rewire(id, &near);

        self.near = near;
        true
    }

    /// The connection radius for the next node: `gamma * (ln n / n)^(1/d)`, with `n` counting
    /// the new node.
    fn radius(&self) -> f64 {
        let n = count_to_f64(self.growth.tree.len() + 1);
        let d = count_to_f64(self.growth.problem.space.dimension().max(1));

        self.gamma * libm::pow(libm::log(n) / n, 1.0 / d)
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
