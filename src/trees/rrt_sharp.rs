//! RRT#: RRT* with dynamic-programming replanning (Arslan and Tsiotras 2013).

use alloc::{collections::BinaryHeap, vec::Vec};

use rand_core::Rng;

use super::{Error, RrtStarConfig, targets::draw_target};
use crate::graph::Entry;
use crate::{
    LinearScan, NearestNeighbors, Path, Problem, Sampler, StateSpace, Steering, ValidityChecker,
};

/// Whether a graph edge has been motion-checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EdgeStatus {
    /// Not checked yet.
    Unchecked,
    /// Checked and valid.
    Valid,
    /// Checked and invalid or unknown; never traveled.
    Invalid,
}

/// A directed graph edge, stored at its source vertex.
#[derive(Debug, Clone)]
struct Edge {
    /// The target vertex.
    to: usize,
    /// The edge length under [`StateSpace::distance`].
    cost: f64,
    /// Whether the motion along the edge was checked, and how.
    status: EdgeStatus,
}

/// A graph vertex with its LPA*-style cost estimates.
#[derive(Debug, Clone)]
struct Vertex<S> {
    /// The state.
    state: S,
    /// The cost-to-come committed by the last replanning; infinite until the vertex is
    /// processed. Never below `lmc`.
    g: f64,
    /// The one-step lookahead `min over valid in-edges (g(u) + c(u, v))`; zero for the root.
    lmc: f64,
    /// The admissible cost-to-go `max(0, d(v, goal center) - tolerance)`.
    h: f64,
    /// The in-neighbor that attains `lmc`; `None` for the root and unreached vertices.
    parent: Option<usize>,
    /// The outgoing edges, checked lazily.
    out: Vec<Edge>,
}

/// RRT#: grows a graph like RRG and keeps the cost-to-come of every promising vertex optimal on
/// that graph by replanning after each extension (Arslan and Tsiotras 2013).
///
/// Each iteration extends the nearest vertex toward a goal-biased sample, like
/// [`RrtStar`](super::RrtStar), and links the new vertex both ways to its neighbors (the same
/// radius or k-nearest rule, from [`RrtStarConfig`]). Instead of RRT*'s one-step rewiring, a
/// replanning step then runs an LPA*-like relaxation: vertices whose lookahead cost `lmc`
/// improved are processed cheapest key (`lmc + h`) first, until no queued vertex can beat the
/// best goal vertex. Vertices that cannot (`lmc + h >= c_best`) stay unprocessed. Because the
/// graph only grows and edge costs are fixed, costs only decrease and the underconsistent case
/// of LPA* never occurs.
///
/// After every step the cost of [`RrtSharp::best_path`] is the shortest-path cost to the goal
/// region on the current graph, so it never increases and improves faster per sample than
/// RRT*.
///
/// Edges are motion-checked lazily, in travel direction, and at most once: only when they would
/// lower a vertex's `lmc`. This is sound for directional checkers and costs fewer checks than
/// checking every graph edge eagerly. Informed sampling works as in RRT*.
///
/// The secondary key of the paper, `min(g, lmc)`, is replaced by the vertex id as the tie
/// break, which keeps runs deterministic.
///
/// # Examples
///
/// ```
/// use rand_core::SeedableRng;
/// use sampling_planning::{
///     Euclidean, GoalRegion, Problem, StraightLine, UniformBox, Validity, ValidityChecker,
///     trees::{RrtSharp, RrtStarConfig},
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
/// let gamma = 1.5 * RrtStarConfig::gamma_lower_bound(2, 100.0);
/// let mut planner = RrtSharp::new(problem, RrtStarConfig::new(gamma, 1.0)).expect("valid");
/// planner.run(1000, &mut rand_pcg::Pcg64Mcg::seed_from_u64(1));
///
/// assert!(planner.best_path().expect("free space").cost < 11.5);
/// ```
#[derive(Clone)]
pub struct RrtSharp<Sp: StateSpace, Sa, C, St, N = LinearScan<<Sp as StateSpace>::State>> {
    /// The problem being solved.
    problem: Problem<Sp, Sa, C, St>,
    /// The validated settings.
    config: RrtStarConfig,
    /// The graph vertices, indexed by vertex id; vertex `0` is the start.
    vertices: Vec<Vertex<Sp::State>>,
    /// Index over all vertex states, by vertex id.
    nn: N,
    /// Vertices with `lmc < g`, keyed by `lmc + h` at push time; stale entries are skipped.
    queue: BinaryHeap<Entry>,
    /// The goal vertex with the lowest committed cost `g`, if any is reached.
    best_goal: Option<usize>,
    /// Reused buffer for neighbor queries.
    near: Vec<usize>,
}

impl<Sp, Sa, C, St> RrtSharp<Sp, Sa, C, St>
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
    /// The same as [`RrtSharp::with_nearest_neighbors`].
    pub fn new(problem: Problem<Sp, Sa, C, St>, config: RrtStarConfig) -> Result<Self, Error> {
        Self::with_nearest_neighbors(problem, config, LinearScan::default())
    }
}

impl<Sp, Sa, C, St, N> RrtSharp<Sp, Sa, C, St, N>
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
    /// - [`Error::InvalidGamma`] if `gamma` (or the k-nearest constant) is not positive and
    ///   finite.
    /// - [`Error::InvalidMaxRadius`] if `max_radius` is not positive.
    pub fn with_nearest_neighbors(
        problem: Problem<Sp, Sa, C, St>,
        config: RrtStarConfig,
        mut nn: N,
    ) -> Result<Self, Error> {
        config.validate()?;

        if !(0.0..=1.0).contains(&config.goal_bias) {
            return Err(Error::GoalBiasOutOfRange);
        }

        let validity = problem.checker.check_state(&problem.start);
        if !validity.is_valid() {
            return Err(Error::InvalidStart(validity));
        }

        let start = problem.start.clone();
        let in_goal = problem.goal.contains(&problem.space, &start);
        nn.insert(0, start.clone());

        Ok(Self {
            vertices: alloc::vec![Vertex {
                h: heuristic(&problem, &start),
                state: start,
                g: 0.0,
                lmc: 0.0,
                parent: None,
                out: Vec::new(),
            }],
            best_goal: in_goal.then_some(0),
            problem,
            config,
            nn,
            queue: BinaryHeap::new(),
            near: Vec::new(),
        })
    }

    /// Runs one iteration: extend, then replan. Returns whether a vertex was added.
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> bool {
        let Some((nearest, new)) = self.extend_target(rng) else {
            return false;
        };

        let id = self.add_vertex(new, nearest);
        self.enqueue_if_promising(id);
        self.replan();

        true
    }

    /// Runs `iterations` iterations.
    pub fn run<R: Rng + ?Sized>(&mut self, iterations: usize, rng: &mut R) {
        for _ in 0..iterations {
            self.step(rng);
        }
    }

    /// Draws a target, steers the nearest vertex toward it and checks that motion. Returns the
    /// nearest vertex and the new state, or `None` if the motion is not valid.
    fn extend_target<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Option<(usize, Sp::State)> {
        let bound = self.informed_bound();
        let target = draw_target(&mut self.problem, self.config.goal_bias, bound, rng);
        let space = &self.problem.space;
        let nearest = self
            .nn
            .nearest(space, &target)
            .expect("the graph always holds the start");
        let from = &self.vertices[nearest].state;
        let new = self.problem.steering.steer(space, from, &target);

        // A zero-length step (the target is already a vertex) would only add a duplicate.
        (space.distance(from, &new) > 0.0
            && self
                .problem
                .checker
                .check_motion(space, from, &new)
                .is_valid())
        .then_some((nearest, new))
    }

    /// The informed-set bound `c_best + tolerance`, or `None` when informed sampling is off or
    /// no solution exists yet.
    fn informed_bound(&self) -> Option<f64> {
        if !self.config.informed {
            return None;
        }

        let best = self.best_goal?;

        Some(self.vertices[best].g + self.problem.goal.tolerance())
    }

    /// Adds a vertex at `state`, linked both ways to its neighbors and to `nearest` (whose edge
    /// to `state` is known to be valid), and sets its lookahead cost from the cheapest valid
    /// in-edge.
    fn add_vertex(&mut self, state: Sp::State, nearest: usize) -> usize {
        let id = self.vertices.len();
        let mut near = core::mem::take(&mut self.near);
        self.config.neighbors(
            &self.problem.space,
            &self.nn,
            &state,
            id + 1,
            |other| &self.vertices[other].state,
            &mut near,
        );

        if !near.contains(&nearest) {
            near.push(nearest);
        }

        self.vertices.push(Vertex {
            h: heuristic(&self.problem, &state),
            state: state.clone(),
            g: f64::INFINITY,
            lmc: f64::INFINITY,
            parent: None,
            out: Vec::new(),
        });
        self.link(id, nearest, &near);
        self.choose_parent(id);

        self.nn.insert(id, state);
        self.near = near;

        id
    }

    /// Adds unchecked edges between the new vertex `id` and every vertex in `near`, both ways;
    /// the edge from `nearest` is already known to be valid.
    fn link(&mut self, id: usize, nearest: usize, near: &[usize]) {
        for &other in near {
            let cost = self
                .problem
                .space
                .distance(&self.vertices[other].state, &self.vertices[id].state);
            let into = if other == nearest {
                EdgeStatus::Valid
            } else {
                EdgeStatus::Unchecked
            };

            self.vertices[other].out.push(Edge {
                to: id,
                cost,
                status: into,
            });
            let back = self
                .problem
                .space
                .distance(&self.vertices[id].state, &self.vertices[other].state);
            self.vertices[id].out.push(Edge {
                to: other,
                cost: back,
                status: EdgeStatus::Unchecked,
            });
        }
    }

    /// Sets `lmc` and the parent of the new vertex `id` from its cheapest valid in-edge, checking
    /// candidates in order of cost until one is valid.
    fn choose_parent(&mut self, id: usize) {
        let mut candidates: Vec<(f64, usize, usize)> = self.vertices[id]
            .out
            .iter()
            .map(|edge| edge.to)
            .filter_map(|from| {
                let slot = self.vertices[from].out.iter().rposition(|e| e.to == id)?;
                let cost = self.vertices[from].g + self.vertices[from].out[slot].cost;

                cost.is_finite().then_some((cost, from, slot))
            })
            .collect();
        candidates.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));

        for (cost, from, slot) in candidates {
            if self.is_edge_valid(from, slot) {
                self.vertices[id].lmc = cost;
                self.vertices[id].parent = Some(from);
                return;
            }
        }
    }

    /// Whether edge `slot` of vertex `from` is valid, checking and caching it on first use.
    fn is_edge_valid(&mut self, from: usize, slot: usize) -> bool {
        let edge = &self.vertices[from].out[slot];

        match edge.status {
            EdgeStatus::Valid => true,
            EdgeStatus::Invalid => false,
            EdgeStatus::Unchecked => {
                let valid = self
                    .problem
                    .checker
                    .check_motion(
                        &self.problem.space,
                        &self.vertices[from].state,
                        &self.vertices[edge.to].state,
                    )
                    .is_valid();
                self.vertices[from].out[slot].status = if valid {
                    EdgeStatus::Valid
                } else {
                    EdgeStatus::Invalid
                };

                valid
            }
        }
    }

    /// Queues vertex `id` if it is inconsistent (`lmc < g`) and could still improve the best
    /// solution (`lmc + h < c_best`).
    fn enqueue_if_promising(&mut self, id: usize) {
        let vertex = &self.vertices[id];
        let key = vertex.lmc + vertex.h;

        if vertex.lmc < vertex.g && key < self.best_cost() {
            self.queue.push(Entry {
                cost: key,
                node: id,
            });
        }
    }

    /// The committed cost of the best goal vertex, or infinity.
    fn best_cost(&self) -> f64 {
        self.best_goal
            .map_or(f64::INFINITY, |id| self.vertices[id].g)
    }

    /// Processes queued vertices cheapest key first until none can beat the best goal vertex:
    /// commits `g = lmc` and relaxes the outgoing edges.
    fn replan(&mut self) {
        while let Some(&Entry { cost: key, node }) = self.queue.peek() {
            if key >= self.best_cost() {
                return;
            }

            self.queue.pop();
            if self.is_stale(key, node) {
                continue;
            }

            self.vertices[node].g = self.vertices[node].lmc;
            self.record_goal(node);
            self.relax_out_edges(node);
        }
    }

    /// Whether a queue entry with `key` for `node` is outdated: the vertex was processed since,
    /// or its key changed.
    #[expect(
        clippy::float_cmp,
        reason = "a current entry carries exactly the key computed from the current lmc"
    )]
    fn is_stale(&self, key: f64, node: usize) -> bool {
        let vertex = &self.vertices[node];

        vertex.g <= vertex.lmc || key != vertex.lmc + vertex.h
    }

    /// Makes the just-processed `node` the best goal vertex if it is a goal vertex and cheaper.
    fn record_goal(&mut self, node: usize) {
        let in_goal = self
            .problem
            .goal
            .contains(&self.problem.space, &self.vertices[node].state);

        if in_goal && self.vertices[node].g < self.best_cost() {
            self.best_goal = Some(node);
        }
    }

    /// Lowers `lmc` of every successor of `node` that is cheaper through `node` and reachable by
    /// a valid edge, and queues it.
    fn relax_out_edges(&mut self, node: usize) {
        let g = self.vertices[node].g;

        for slot in 0..self.vertices[node].out.len() {
            let Edge { to, cost, .. } = self.vertices[node].out[slot];
            let candidate = g + cost;

            if candidate < self.vertices[to].lmc && self.is_edge_valid(node, slot) {
                self.vertices[to].lmc = candidate;
                self.vertices[to].parent = Some(node);
                self.enqueue_if_promising(to);
            }
        }
    }

    /// The cheapest path to the goal region on the current graph, or `None` if the goal was
    /// not reached yet. Its cost never increases as iterations run.
    #[must_use]
    pub fn best_path(&self) -> Option<Path<Sp::State>> {
        let goal = self.best_goal?;
        let mut states = Vec::new();
        let mut cost = 0.0;
        let mut current = Some(goal);

        while let Some(id) = current {
            let vertex = &self.vertices[id];
            states.push(vertex.state.clone());

            if let Some(parent) = vertex.parent {
                cost += self
                    .problem
                    .space
                    .distance(&self.vertices[parent].state, &vertex.state);
            }

            current = vertex.parent;
        }

        states.reverse();

        Some(Path { states, cost })
    }

    /// The number of graph vertices, including the start.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.vertices.len()
    }
}

/// The admissible cost-to-go `max(0, d(state, goal center) - tolerance)`.
fn heuristic<Sp: StateSpace, Sa, C, St>(
    problem: &Problem<Sp, Sa, C, St>,
    state: &Sp::State,
) -> f64 {
    (problem.space.distance(state, problem.goal.center()) - problem.goal.tolerance()).max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Arc, shortest_path};
    use crate::{Euclidean, GoalRegion, StraightLine, UniformBox, Validity, discretized_motion};
    use rand_core::SeedableRng;

    /// A wall at `x` in `[4, 6]` for `y < 8` in a 10 x 10 box.
    struct Wall;

    impl ValidityChecker<Euclidean<2>> for Wall {
        fn check_state(&self, s: &[f64; 2]) -> Validity {
            if (4.0..=6.0).contains(&s[0]) && s[1] < 8.0 {
                Validity::Invalid
            } else {
                Validity::Valid
            }
        }

        fn check_motion(&self, space: &Euclidean<2>, from: &[f64; 2], to: &[f64; 2]) -> Validity {
            discretized_motion(space, from, to, 0.02, |s| self.check_state(s))
        }
    }

    /// An RRT# planner on the wall world.
    fn planner() -> RrtSharp<Euclidean<2>, UniformBox<2>, Wall, StraightLine> {
        let problem = Problem {
            space: Euclidean::<2>,
            sampler: UniformBox::new([0.0; 2], [10.0; 2]),
            checker: Wall,
            steering: StraightLine::new(1.0),
            start: [1.0, 1.0],
            goal: GoalRegion::new([9.0, 1.0], 0.5),
        };
        let gamma = 1.5 * RrtStarConfig::gamma_lower_bound(2, 100.0);

        RrtSharp::new(problem, RrtStarConfig::new(gamma, 1.0)).expect("valid")
    }

    /// Dijkstra over every graph edge that is valid when checked eagerly.
    fn eager_optimum(planner: &RrtSharp<Euclidean<2>, UniformBox<2>, Wall, StraightLine>) -> f64 {
        let space = Euclidean::<2>;
        let vertices = &planner.vertices;
        let route = shortest_path(
            vertices.len(),
            0,
            |id| planner.problem.goal.contains(&space, &vertices[id].state),
            |id, out: &mut Vec<Arc>| {
                out.clear();
                out.extend(
                    vertices[id]
                        .out
                        .iter()
                        .filter(|e| {
                            Wall.check_motion(&space, &vertices[id].state, &vertices[e.to].state)
                                .is_valid()
                        })
                        .map(|e| Arc {
                            to: e.to,
                            cost: e.cost,
                            edge: None,
                        }),
                );
            },
        );

        route.map_or(f64::INFINITY, |route| route.cost)
    }

    #[test]
    fn replanning_keeps_the_graph_optimum_with_lazy_checks() {
        let mut planner = planner();
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(2);
        let mut checked = 0;

        for _ in 0..8 {
            planner.run(250, &mut rng);

            let best = planner.best_path().map_or(f64::INFINITY, |path| path.cost);
            let optimum = eager_optimum(&planner);
            assert!((best - optimum).abs() < 1e-9, "{best} vs {optimum}");
            checked += usize::from(best.is_finite());
        }

        assert!(checked > 4, "solved early enough");

        let unchecked = planner
            .vertices
            .iter()
            .flat_map(|v| &v.out)
            .filter(|e| e.status == EdgeStatus::Unchecked)
            .count();
        assert!(unchecked > 0, "lazy checking skips edges");
    }
}
