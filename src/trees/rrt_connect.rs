//! RRT-Connect: bidirectional RRT with a greedy connect step (Kuffner and LaValle 2000).

use rand_core::Rng;

use super::{Error, tree::Tree};
use crate::{
    LinearScan, NearestNeighbors, Path, Problem, Sampler, StateSpace, Steering, ValidityChecker,
};

/// Which of the two trees an operation works on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Side {
    /// The tree rooted at the start; its edges are traveled from parent to child.
    Start,
    /// The tree rooted at the goal center; its edges are traveled from child to parent.
    Goal,
}

impl Side {
    /// The other tree.
    const fn other(self) -> Self {
        match self {
            Self::Start => Self::Goal,
            Self::Goal => Self::Start,
        }
    }
}

/// One tree with its nearest-neighbor index.
#[derive(Clone)]
struct Half<S, N> {
    /// The tree, rooted at the start or the goal center.
    tree: Tree<S>,
    /// Index over the tree's nodes, by node id.
    nn: N,
}

/// RRT-Connect: grows one tree from the start and one from the goal center, and after every
/// extension of one tree greedily steers the other tree toward the new node until it either
/// reaches it or hits an obstacle.
///
/// Finds feasible paths much faster than [`Rrt`](super::Rrt) in open spaces with narrow
/// passages near the endpoints, but, like RRT, does not improve them; run
/// [`shortcut`](crate::shortcut) on the result.
///
/// The goal tree is rooted at [`GoalRegion::center`](crate::GoalRegion::center), so the
/// returned path ends there. Its edges are traveled toward the root, so they are checked in that
/// direction: a new goal-tree state gets [`ValidityChecker::check_state`] and then
/// [`ValidityChecker::check_motion`] from itself to its parent.
///
/// The trees meet when a steered state is at distance zero from the target state, so
/// [`Steering::steer`] must return the target itself when it can reach it, as
/// [`StraightLine`](crate::StraightLine) does.
///
/// # Examples
///
/// ```
/// use rand_core::SeedableRng;
/// use sampling_planning::{
///     Euclidean, GoalRegion, Problem, StraightLine, UniformBox, Validity, ValidityChecker,
///     trees::RrtConnect,
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
/// let problem = Problem {
///     space: Euclidean::<2>,
///     sampler: UniformBox::new([0.0, 0.0], [10.0, 10.0]),
///     checker: Free,
///     steering: StraightLine::new(1.0),
///     start: [1.0, 1.0],
///     goal: GoalRegion::new([9.0, 9.0], 0.5),
/// };
///
/// let mut planner = RrtConnect::new(problem).expect("valid start and goal");
/// let path = planner.solve(1000, &mut rand_pcg::Pcg64Mcg::seed_from_u64(1));
///
/// assert_eq!(path.expect("free space").states.last(), Some(&[9.0, 9.0]));
/// ```
#[derive(Clone)]
pub struct RrtConnect<Sp: StateSpace, Sa, C, St, N = LinearScan<<Sp as StateSpace>::State>> {
    /// The problem being solved.
    problem: Problem<Sp, Sa, C, St>,
    /// The tree rooted at the start.
    start: Half<Sp::State, N>,
    /// The tree rooted at the goal center.
    goal: Half<Sp::State, N>,
    /// The tree that extends toward the next sample; alternates every iteration.
    active: Side,
    /// The first solution found, if any.
    solution: Option<Path<Sp::State>>,
}

impl<Sp, Sa, C, St> RrtConnect<Sp, Sa, C, St>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    St: Steering<Sp>,
{
    /// Creates a planner with [`LinearScan`] nearest-neighbor indices.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidStart`] if the start state is not valid.
    /// - [`Error::InvalidGoal`] if the goal center is not valid.
    pub fn new(problem: Problem<Sp, Sa, C, St>) -> Result<Self, Error> {
        Self::with_nearest_neighbors(problem, LinearScan::default())
    }
}

impl<Sp, Sa, C, St, N> RrtConnect<Sp, Sa, C, St, N>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    St: Steering<Sp>,
    N: NearestNeighbors<Sp> + Clone,
{
    /// Creates a planner; each tree gets a clone of the given empty nearest-neighbor index.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidStart`] if the start state is not valid.
    /// - [`Error::InvalidGoal`] if the goal center is not valid.
    pub fn with_nearest_neighbors(problem: Problem<Sp, Sa, C, St>, nn: N) -> Result<Self, Error> {
        let start_validity = problem.checker.check_state(&problem.start);
        if !start_validity.is_valid() {
            return Err(Error::InvalidStart(start_validity));
        }

        let goal_validity = problem.checker.check_state(problem.goal.center());
        if !goal_validity.is_valid() {
            return Err(Error::InvalidGoal(goal_validity));
        }

        let solution = problem
            .goal
            .contains(&problem.space, &problem.start)
            .then(|| Path {
                states: alloc::vec![problem.start.clone()],
                cost: 0.0,
            });

        Ok(Self {
            start: Half::rooted(problem.start.clone(), nn.clone()),
            goal: Half::rooted(problem.goal.center().clone(), nn),
            problem,
            active: Side::Start,
            solution,
        })
    }

    /// Runs one iteration: extends the active tree toward a sample, then connects the other
    /// tree toward the new node. Returns whether a solution exists afterward.
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> bool {
        if self.solution.is_some() {
            return true;
        }

        let side = self.active;
        self.active = side.other();

        let target = self.problem.sampler.sample(rng);
        if let Some(new) = self.extend(side, &target) {
            self.check_direct_goal(side, new);

            if self.solution.is_none() {
                self.connect(side.other(), new, side);
            }
        }

        self.solution.is_some()
    }

    /// Runs until a solution is found or `max_iterations` iterations have run.
    ///
    /// Returns the solution, or `None` if none was found within the budget.
    pub fn solve<R: Rng + ?Sized>(
        &mut self,
        max_iterations: usize,
        rng: &mut R,
    ) -> Option<Path<Sp::State>> {
        for _ in 0..max_iterations {
            if self.step(rng) {
                break;
            }
        }

        self.solution.clone()
    }

    /// The solution, or `None` if the trees have not met yet.
    #[must_use]
    pub const fn path(&self) -> Option<&Path<Sp::State>> {
        self.solution.as_ref()
    }

    /// The number of nodes in both trees, including both roots.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.start.tree.len() + self.goal.tree.len()
    }

    /// The tree on `side`.
    const fn half(&self, side: Side) -> &Half<Sp::State, N> {
        match side {
            Side::Start => &self.start,
            Side::Goal => &self.goal,
        }
    }

    /// Takes one steering step on `side` from the node nearest to `target`.
    ///
    /// Returns the new node id, or `None` if the motion was not valid.
    fn extend(&mut self, side: Side, target: &Sp::State) -> Option<usize> {
        let nearest = self.nearest(side, target);

        self.step_from(side, nearest, target)
    }

    /// Takes one steering step on `side` from node `from` toward `target`. Returns the new node
    /// id, or `None` if the edge was not valid.
    fn step_from(&mut self, side: Side, from: usize, target: &Sp::State) -> Option<usize> {
        let space = &self.problem.space;
        let tree = &self.half(side).tree;
        let new = self.problem.steering.steer(space, tree.state(from), target);

        if !self.is_edge_valid(side, from, &new) {
            return None;
        }

        let cost = tree.cost(from) + space.distance(tree.state(from), &new);
        let half = match side {
            Side::Start => &mut self.start,
            Side::Goal => &mut self.goal,
        };
        let id = half.tree.add(new.clone(), from, cost);
        half.nn.insert(id, new);

        Some(id)
    }

    /// Greedily steers `side` toward node `target` of tree `target_side` until it reaches the
    /// node (and records the solution) or an edge is not valid.
    fn connect(&mut self, side: Side, target: usize, target_side: Side) {
        let goal = self.half(target_side).tree.state(target).clone();
        let mut current = self.nearest(side, &goal);

        while let Some(next) = self.step_from(side, current, &goal) {
            let reached = self.half(side).tree.state(next);

            if self.problem.space.distance(reached, &goal) <= 0.0 {
                self.record_meeting(side, next, target);
                return;
            }

            current = next;
        }
    }

    /// Records the solution if start-tree node `id`, just added on `side`, is in the goal
    /// region; the start tree can reach the region before it meets the goal tree.
    fn check_direct_goal(&mut self, side: Side, id: usize) {
        let tree = &self.start.tree;

        if side == Side::Start
            && self
                .problem
                .goal
                .contains(&self.problem.space, tree.state(id))
        {
            self.solution = Some(Path {
                states: tree.path_to(id),
                cost: tree.cost(id),
            });
        }
    }

    /// Records the path through the meeting node pair: node `id` on `side` and node `other` on
    /// the opposite tree hold the same state.
    fn record_meeting(&mut self, side: Side, id: usize, other: usize) {
        let (start_id, goal_id) = match side {
            Side::Start => (id, other),
            Side::Goal => (other, id),
        };

        let mut states = self.start.tree.path_to(start_id);
        let mut to_goal = self.goal.tree.path_to(goal_id);
        to_goal.reverse();
        states.extend(to_goal.into_iter().skip(1));

        self.solution = Some(Path {
            states,
            cost: self.start.tree.cost(start_id) + self.goal.tree.cost(goal_id),
        });
    }

    /// The node on `side` nearest to `target`.
    fn nearest(&self, side: Side, target: &Sp::State) -> usize {
        self.half(side)
            .nn
            .nearest(&self.problem.space, target)
            .expect("each tree always holds its root")
    }

    /// Whether the edge between node `parent` on `side` and the new state `new` is valid in
    /// travel direction: parent to child in the start tree, child to parent in the goal tree.
    fn is_edge_valid(&self, side: Side, parent: usize, new: &Sp::State) -> bool {
        let checker = &self.problem.checker;
        let space = &self.problem.space;
        let parent_state = self.half(side).tree.state(parent);

        match side {
            Side::Start => checker.check_motion(space, parent_state, new).is_valid(),
            Side::Goal => {
                checker.check_state(new).is_valid()
                    && checker.check_motion(space, new, parent_state).is_valid()
            }
        }
    }
}

impl<S: Clone, N> Half<S, N> {
    /// A tree holding only `root`, indexed as node `0` in the empty index `nn`.
    fn rooted<Sp>(root: S, mut nn: N) -> Self
    where
        Sp: StateSpace<State = S>,
        N: NearestNeighbors<Sp>,
    {
        nn.insert(0, root.clone());

        Self {
            tree: Tree::new(root),
            nn,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Euclidean, GoalRegion, StraightLine, UniformBox, Validity, discretized_motion};
    use rand_core::SeedableRng;

    /// A checker whose wall at `x = 5` is a one-way gate: motions may cross it only while `x`
    /// increases, so a solution from left to right is valid only in travel direction.
    struct OneWay;

    impl ValidityChecker<Euclidean<1>> for OneWay {
        fn check_state(&self, _: &[f64; 1]) -> Validity {
            Validity::Valid
        }

        fn check_motion(&self, space: &Euclidean<1>, from: &[f64; 1], to: &[f64; 1]) -> Validity {
            let crosses = (from[0] < 5.0) != (to[0] < 5.0);

            if crosses && to[0] < from[0] {
                Validity::Invalid
            } else {
                discretized_motion(space, from, to, 0.1, |_| Validity::Valid)
            }
        }
    }

    #[test]
    fn paths_are_valid_in_travel_direction() {
        let problem = Problem {
            space: Euclidean::<1>,
            sampler: UniformBox::new([0.0], [10.0]),
            checker: OneWay,
            steering: StraightLine::new(0.7),
            start: [1.0],
            goal: GoalRegion::new([9.0], 0.1),
        };
        let mut planner = RrtConnect::new(problem).expect("valid");
        let path = planner
            .solve(10_000, &mut rand_pcg::Pcg64Mcg::seed_from_u64(4))
            .expect("the gate is passable left to right");
        for pair in path.states.windows(2) {
            assert_eq!(
                OneWay.check_motion(&Euclidean::<1>, &pair[0], &pair[1]),
                Validity::Valid
            );
        }

        assert!((path.cost - 8.0).abs() < 1e-9, "{}", path.cost);
    }

    #[test]
    fn one_way_gate_blocks_the_reverse_direction() {
        // A goal tree that checked parent to child would cross the gate leftward from the goal
        // and return a path that is invalid in travel direction.
        let problem = Problem {
            space: Euclidean::<1>,
            sampler: UniformBox::new([0.0], [10.0]),
            checker: OneWay,
            steering: StraightLine::new(0.7),
            start: [9.0],
            goal: GoalRegion::new([1.0], 0.1),
        };
        let mut planner = RrtConnect::new(problem).expect("valid");

        assert_eq!(
            planner.solve(3000, &mut rand_pcg::Pcg64Mcg::seed_from_u64(4)),
            None
        );
    }
}
