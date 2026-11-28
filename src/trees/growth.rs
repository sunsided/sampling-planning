//! Machinery shared by the tree planners: goal-biased sampling, extension and solutions.

use alloc::vec::Vec;

use rand_core::Rng;

use super::{Error, retirement::Retirement, targets::draw_target, tree::Tree};
use crate::{NearestNeighbors, Path, Problem, Sampler, StateSpace, Steering, ValidityChecker};

/// A tree with its nearest-neighbor index and vertex retirement: one RRT-family search tree.
#[derive(Clone)]
pub(super) struct IndexedTree<S, N> {
    /// The tree.
    pub(super) tree: Tree<S>,
    /// Index over all tree nodes, using tree node ids.
    pub(super) nn: N,
    /// Removes vertices that keep failing to extend from `nn`; disabled by default.
    pub(super) retirement: Retirement<N>,
}

impl<S: Clone, N> IndexedTree<S, N> {
    /// A tree holding only `root`, indexed as node `0` in the empty index `nn`.
    pub(super) fn rooted<Sp>(root: S, mut nn: N) -> Self
    where
        Sp: StateSpace<State = S>,
        N: NearestNeighbors<Sp>,
    {
        nn.insert(0, root.clone());

        Self {
            tree: Tree::new(root),
            nn,
            retirement: Retirement::disabled(),
        }
    }

    /// Adds a node to the tree and the index; returns its id.
    pub(super) fn add<Sp>(&mut self, state: S, parent: usize, cost: f64) -> usize
    where
        Sp: StateSpace<State = S>,
        N: NearestNeighbors<Sp>,
    {
        let id = self.tree.add(state.clone(), parent, cost);
        self.retirement.note_added::<Sp>(id, state.clone());
        self.nn.insert(id, state);

        id
    }

    /// Records whether the extension from node `from` to the steered state `to` made progress
    /// (the motion was valid and moved away from `from`), and retires `from` once that failed
    /// too often or, for finite steering, once all its successors are in the tree. Call it
    /// after adding the new node, if any.
    pub(super) fn record_extension<Sp, St>(
        &mut self,
        space: &Sp,
        steering: &St,
        from: usize,
        to: &S,
        valid: bool,
    ) where
        Sp: StateSpace<State = S>,
        St: Steering<Sp>,
        N: NearestNeighbors<Sp>,
    {
        if !self.retirement.is_enabled() {
            return;
        }

        let from_state = self.tree.state(from);
        let progressed = valid && space.distance(from_state, to) > 0.0;
        self.retirement.record(&mut self.nn, from, progressed);

        if let Some(successors) = steering.successors(space, from_state) {
            self.retirement
                .retire_if_exhausted(&mut self.nn, space, from, &successors);
        }
    }
}

/// Checks that `goal_bias` is a probability and the start is valid.
pub(super) fn validate<Sp, Sa, C, St>(
    problem: &Problem<Sp, Sa, C, St>,
    goal_bias: f64,
) -> Result<(), Error>
where
    Sp: StateSpace,
    C: ValidityChecker<Sp>,
{
    if !(0.0..=1.0).contains(&goal_bias) {
        return Err(Error::GoalBiasOutOfRange);
    }

    let validity = problem.checker.check_state(&problem.start);
    if !validity.is_valid() {
        return Err(Error::InvalidStart(validity));
    }

    Ok(())
}

/// Whether the extension from `from` to `to` moves and checks as valid. A zero-length step
/// (the target is already in the tree) would only add a duplicate node.
pub(super) fn is_extension_valid<Sp, C>(
    space: &Sp,
    checker: &C,
    from: &Sp::State,
    to: &Sp::State,
) -> bool
where
    Sp: StateSpace,
    C: ValidityChecker<Sp>,
{
    space.distance(from, to) > 0.0 && checker.check_motion(space, from, to).is_valid()
}

/// A growing tree over a [`Problem`], with its nearest-neighbor index and goal nodes.
#[derive(Clone)]
pub(super) struct Growth<Sp: StateSpace, Sa, C, St, N> {
    /// The problem being solved.
    pub(super) problem: Problem<Sp, Sa, C, St>,
    /// The search tree, rooted at the start, with its index.
    pub(super) indexed: IndexedTree<Sp::State, N>,
    /// Ids of tree nodes inside the goal region, in insertion order.
    goal_nodes: Vec<usize>,
    /// Probability in `[0, 1]` of sampling the goal center instead of the sampler.
    pub(super) goal_bias: f64,
}

impl<Sp, Sa, C, St, N> Growth<Sp, Sa, C, St, N>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    St: Steering<Sp>,
    N: NearestNeighbors<Sp>,
{
    /// Checks the start and goal bias and plants the root.
    ///
    /// `nn` must be empty; it receives the root as node `0`.
    pub(super) fn new(
        problem: Problem<Sp, Sa, C, St>,
        goal_bias: f64,
        nn: N,
    ) -> Result<Self, Error> {
        validate(&problem, goal_bias)?;

        let goal_nodes = if problem.goal.contains(&problem.space, &problem.start) {
            alloc::vec![0]
        } else {
            Vec::new()
        };

        Ok(Self {
            indexed: IndexedTree::rooted(problem.start.clone(), nn),
            problem,
            goal_nodes,
            goal_bias,
        })
    }

    /// [`IndexedTree::record_extension`] on the search tree.
    pub(super) fn record_extension(&mut self, from: usize, to: &Sp::State, valid: bool) {
        let (space, steering) = (&self.problem.space, &self.problem.steering);
        self.indexed
            .record_extension(space, steering, from, to, valid);
    }

    /// Picks the nearest node to a goal-biased sample and steers from it.
    ///
    /// Returns the nearest node id and the steered state; the motion is not checked yet.
    pub(super) fn propose<R: Rng + ?Sized>(&mut self, rng: &mut R) -> (usize, Sp::State) {
        let target = self.draw(rng);

        self.propose_toward(&target)
    }

    /// Picks the nearest node to `target` and steers from it toward `target`.
    ///
    /// Returns the nearest node id and the steered state; the motion is not checked yet.
    pub(super) fn propose_toward(&self, target: &Sp::State) -> (usize, Sp::State) {
        let nearest = self
            .indexed
            .nn
            .nearest(&self.problem.space, target)
            .expect(crate::nearest::NO_NEAREST);
        let new = self.problem.steering.steer(
            &self.problem.space,
            self.indexed.tree.state(nearest),
            target,
        );

        (nearest, new)
    }

    /// Draws the goal center with probability `goal_bias`, otherwise a sampler state.
    pub(super) fn draw<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Sp::State {
        draw_target(&mut self.problem, self.goal_bias, None, rng)
    }

    /// Whether the extension from node `from` to `to` moves and checks as valid. A zero-length
    /// step (the target is already in the tree) would only add a duplicate node.
    pub(super) fn is_extension_valid(&self, from: usize, to: &Sp::State) -> bool {
        let from = self.indexed.tree.state(from);

        is_extension_valid(&self.problem.space, &self.problem.checker, from, to)
    }

    /// Whether the motion from node `from` to `to` checks as valid. Unknown counts as invalid.
    pub(super) fn is_motion_valid(&self, from: usize, to: &Sp::State) -> bool {
        self.problem
            .checker
            .check_motion(&self.problem.space, self.indexed.tree.state(from), to)
            .is_valid()
    }

    /// The path cost of reaching `to` from node `from` by a direct motion.
    pub(super) fn cost_via(&self, from: usize, to: &Sp::State) -> f64 {
        self.indexed.tree.cost(from)
            + self
                .problem
                .space
                .distance(self.indexed.tree.state(from), to)
    }

    /// Adds a node to the tree, the index and, if it reaches the goal, the goal nodes.
    pub(super) fn add(&mut self, state: Sp::State, parent: usize, cost: f64) -> usize {
        if self.problem.goal.contains(&self.problem.space, &state) {
            self.goal_nodes.push(self.indexed.tree.len());
        }

        self.indexed.add(state, parent, cost)
    }

    /// Whether any node reached the goal region.
    pub(super) fn has_solution(&self) -> bool {
        !self.goal_nodes.is_empty()
    }

    /// The cost of the cheapest path to a goal node, or `None` if no node reached the goal yet.
    pub(super) fn best_cost(&self) -> Option<f64> {
        self.best_goal_node().map(|id| self.indexed.tree.cost(id))
    }

    /// The cheapest path to a goal node, or `None` if no node reached the goal yet.
    pub(super) fn best_path(&self) -> Option<Path<Sp::State>> {
        self.best_goal_node().map(|id| Path {
            states: self.indexed.tree.path_to(id),
            cost: self.indexed.tree.cost(id),
        })
    }

    /// The goal node with the lowest cost; the earliest one on ties.
    fn best_goal_node(&self) -> Option<usize> {
        let mut best: Option<usize> = None;

        for &id in &self.goal_nodes {
            if best.is_none_or(|b| self.indexed.tree.cost(id) < self.indexed.tree.cost(b)) {
                best = Some(id);
            }
        }

        best
    }
}
