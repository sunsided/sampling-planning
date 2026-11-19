//! RRT: the rapidly-exploring random tree (LaValle 1998).

use rand_core::Rng;

use super::{Error, growth::Growth};
use crate::{
    LinearScan, NearestNeighbors, Path, Problem, Sampler, StateSpace, Steering, ValidityChecker,
};

/// Settings for [`Rrt`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RrtConfig {
    /// Probability in `[0, 1]` of extending toward the goal center instead of a random sample.
    pub goal_bias: f64,
}

impl Default for RrtConfig {
    /// A goal bias of 5 %.
    fn default() -> Self {
        Self { goal_bias: 0.05 }
    }
}

/// The rapidly-exploring random tree: grows toward random samples, finds a feasible path fast,
/// but does not improve it.
///
/// Each iteration extends the nearest node toward a goal-biased sample by one steering step
/// and keeps the new node if the motion is [`Validity::Valid`](crate::Validity::Valid).
#[derive(Clone)]
pub struct Rrt<Sp: StateSpace, Sa, C, St, N = LinearScan<<Sp as StateSpace>::State>> {
    /// The tree and problem.
    growth: Growth<Sp, Sa, C, St, N>,
}

impl<Sp, Sa, C, St> Rrt<Sp, Sa, C, St>
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
    pub fn new(problem: Problem<Sp, Sa, C, St>, config: RrtConfig) -> Result<Self, Error> {
        Self::with_nearest_neighbors(problem, config, LinearScan::default())
    }
}

impl<Sp, Sa, C, St, N> Rrt<Sp, Sa, C, St, N>
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
    pub fn with_nearest_neighbors(
        problem: Problem<Sp, Sa, C, St>,
        config: RrtConfig,
        nn: N,
    ) -> Result<Self, Error> {
        Ok(Self {
            growth: Growth::new(problem, config.goal_bias, nn)?,
        })
    }

    /// Runs one iteration. Returns whether a node was added.
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> bool {
        let (nearest, new) = self.growth.propose(rng);
        if !self.growth.is_motion_valid(nearest, &new) {
            return false;
        }

        let cost = self.growth.cost_via(nearest, &new);
        self.growth.add(new, nearest, cost);

        true
    }

    /// Runs `iterations` iterations.
    pub fn run<R: Rng + ?Sized>(&mut self, iterations: usize, rng: &mut R) {
        for _ in 0..iterations {
            self.step(rng);
        }
    }

    /// Runs until the first solution or at most `max_iterations` iterations.
    ///
    /// Returns the solution, or `None` if none was found within the budget.
    pub fn solve<R: Rng + ?Sized>(
        &mut self,
        max_iterations: usize,
        rng: &mut R,
    ) -> Option<Path<Sp::State>> {
        for _ in 0..max_iterations {
            if self.growth.has_solution() {
                break;
            }

            self.step(rng);
        }

        self.growth.best_path()
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
