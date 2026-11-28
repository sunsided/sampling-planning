//! EIT*: effort informed trees (Strub and Gammell 2022).

mod checks;
mod forward;
mod queue;
mod reverse;

use alloc::{
    collections::{BTreeMap, BTreeSet, BinaryHeap},
    vec::Vec,
};

use rand_core::Rng;

use super::batch::{Informed, Neighborhood};
use super::queues::EdgeEntry;
use super::{BitStarConfig, Error};
use crate::{GoalRegion, LinearScan, NearestNeighbors, Path, Sampler, StateSpace, ValidityChecker};
use queue::ForwardQueue;

/// Settings for [`EitStar`]: BIT*'s batch settings plus the effort model.
///
/// EIT* computes the batch radius as OMPL does: `q` counts every stored state, the newest
/// batch included, and `gamma` shrinks with the informed set's share of the sampled domain
/// ([`Sampler::informed_fraction`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EitStarConfig {
    /// The batch size, neighbor rule and edge cap, as for [`BitStar`](super::BitStar).
    pub bit: BitStarConfig,
    /// The distance between two state checks of a full-resolution edge check, in units of
    /// [`StateSpace::distance`]: an edge of length `d` costs `ceil(d / resolution)` checks of
    /// effort. Match the resolution of the checker's
    /// [`check_motion`](ValidityChecker::check_motion), for example the one passed to
    /// [`discretized_motion`](crate::discretized_motion). Positive and finite.
    pub resolution: f64,
    /// The number of states the reverse search checks per edge at first (OMPL's
    /// `initial_num_sparse_collision_checks`, 1 by default). At least one.
    pub initial_sparse_checks: usize,
    /// How much costlier than the best lower bound an edge may be for the forward search to
    /// prefer it for its lower effort (OMPL's `suboptimality_factor`). At least one; infinite
    /// by default, which always takes the least-effort edge.
    pub suboptimality: f64,
}

impl EitStarConfig {
    /// A radius-based config with 100 samples per batch, the given check resolution and
    /// OMPL's EIT* defaults: one initial sparse check and infinite suboptimality.
    #[must_use]
    pub const fn new(gamma: f64, resolution: f64) -> Self {
        Self::from_bit_star(BitStarConfig::new(gamma), resolution)
    }

    /// OMPL's EIT* defaults on top of the given batch settings.
    #[must_use]
    pub const fn from_bit_star(bit: BitStarConfig, resolution: f64) -> Self {
        Self {
            bit,
            resolution,
            initial_sparse_checks: 1,
            suboptimality: f64::INFINITY,
        }
    }

    /// Panics unless the effort settings are in their documented ranges.
    fn assert_effort(&self) {
        assert!(
            self.resolution.is_finite() && self.resolution > 0.0,
            "check resolution must be positive and finite"
        );
        assert!(
            self.initial_sparse_checks > 0,
            "initial sparse checks must be at least one"
        );
        assert!(
            self.suboptimality >= 1.0,
            "suboptimality factor must be at least one"
        );
    }
}

/// What a stored state currently is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// A forward-tree vertex.
    Vertex,
    /// A sample not connected to the forward tree.
    Sample,
    /// Pruned for good: it cannot improve the solution.
    Pruned,
}

/// One stored state with its forward-tree and reverse-search data.
#[derive(Debug, Clone)]
struct Node<S> {
    /// The state.
    state: S,
    /// What the state currently is.
    kind: Kind,
    /// The forward cost-to-come through the tree; infinite for samples.
    g: f64,
    /// The geometric cost-to-come estimate `d(start, x)`.
    g_hat: f64,
    /// The geometric cost-to-go estimate `max(0, d(x, goal center) - tolerance)`.
    h_hat: f64,
    /// The forward-tree parent; `None` for the start and samples.
    parent: Option<usize>,
    /// The forward-tree children.
    children: Vec<usize>,
    /// The batch graph's neighbors, by id, sorted. Valid while `adjacent_batch` is the
    /// current batch; see [`EitStar::ensure_adjacent`].
    adjacent: Vec<usize>,
    /// The batch whose graph `adjacent` holds; `0` before the first batch.
    adjacent_batch: usize,
    /// The forward-tree parent and children at the start of the batch, which join the
    /// lazily built neighbor list.
    tree_links: Vec<usize>,
    /// The reverse search's cost-to-go over the batch graph with sparsely checked edges;
    /// infinite until reached. Admissible for the current graph only.
    h: f64,
    /// The reverse search's estimate of the checks still needed to reach the goal.
    effort: usize,
    /// The reverse-tree parent, toward the goal; `None` for goals and unreached states.
    reverse_parent: Option<usize>,
    /// The reverse search in which this state was last expanded; equal to the planner's tag
    /// while it is closed.
    closed_tag: usize,
    /// The sources of the queued forward edges into this state.
    incoming: Vec<usize>,
}

impl<S> Node<S> {
    /// The admissible solution cost through this state, `g_hat + h_hat`.
    fn f_hat(&self) -> f64 {
        self.g_hat + self.h_hat
    }

    /// Whether the state lies in the goal region.
    fn in_goal(&self) -> bool {
        self.h_hat <= 0.0
    }
}

/// What the planner remembers about checked edges, by node id.
#[derive(Debug, Clone, Default)]
struct EdgeChecks {
    /// The number of states already checked on each undirected edge `(low, high)`.
    performed: BTreeMap<(usize, usize), usize>,
    /// Directed edges whose motion checked as valid.
    valid: BTreeSet<(usize, usize)>,
    /// Directed edges known to be invalid: from a failed motion check, or both directions
    /// after a state on the edge failed.
    invalid: BTreeSet<(usize, usize)>,
}

/// The id of the start state.
const START: usize = 0;

/// EIT*: effort informed trees (Strub and Gammell 2022, "AIT* and EIT*: Asymmetric
/// bidirectional sampling-based path planning", IJRR 41(4), arXiv 2111.01877).
///
/// EIT* searches each batch's random geometric graph twice, like [`AitStar`](super::AitStar),
/// but its reverse search also estimates **effort**: how many state checks it still takes to
/// reach the goal. The forward search uses it to check the cheapest-to-check edges first, so
/// it finds a first solution with few collision checks:
///
/// ```mermaid
/// graph LR
///     Batch["new batch: prune, sample, restart reverse search"] --> Reverse
///     Reverse["reverse A* from the goals: sparse checks, cost-to-go h, effort-to-go"] --> Forward
///     Forward["forward: pop least-effort edge within suboptimality of best bound"] -->|valid| Tree["add or rewire"]
///     Forward -->|invalid edge of the reverse tree| Refine["more sparse checks, restart reverse"]
///     Refine --> Reverse
///     Tree --> Reverse
///     Forward -->|no edge beats the solution| Batch
/// ```
///
/// This follows OMPL's `EITstar` (Strub) without multi-query planning:
///
/// - The reverse search is an A* over edges from the goal states (the goal center and every
///   sample in the goal region), keyed `(h(s) + c_hat + g_hat(t), h(s) + c_hat, effort)`. It
///   accepts an edge only if `initial_sparse_checks` evenly spaced states on it pass
///   [`check_state`](ValidityChecker::check_state), in bisection order, and records
///   `h(t) = h(s) + c_hat` and `effort(t) = effort(s) + remaining checks of the edge`.
/// - The forward queue holds edges keyed by the lower bound `g + c_hat + h` and the effort
///   `remaining checks + effort(x)`. It pops the least-effort edge among those whose bound is
///   within `suboptimality` times the smallest bound (ties by the bound), and the edge with
///   the smallest bound when that one is not strictly within it.
/// - The forward search checks an edge with [`check_motion`](ValidityChecker::check_motion)
///   and adds or rewires its target if that lowers its cost and `g + c + h` beats the
///   solution. An invalid edge that the reverse tree uses raises the sparse check count from
///   `k` to `2k + 1` (the new points interleave the old ones) and restarts the reverse search.
/// - The reverse search runs until the forward search's best edge leads to a closed state
///   with a bound below the reverse queue's, every queued forward edge leads to a closed
///   state, or, with infinite suboptimality, the best edge leads to a reached state. The
///   forward search runs while its smallest bound beats the solution; otherwise a new batch
///   starts.
///
/// Deviations: OMPL's estimated cost-to-go equals its admissible one for path length, so
/// there is only `h` here. Effort comes from [`EitStarConfig::resolution`], since
/// [`ValidityChecker`] does not count its checks. Full checks are whitelisted per direction,
/// so directional checkers stay sound; OMPL whitelists both. The forward queue is scanned on
/// every pop with a finite suboptimality, as in OMPL; with the default infinite one, the
/// queue is ordered by effort and pops in `O(log n)`. Pruning, batches and
/// the radius follow [`AitStar`](super::AitStar), using the geometric `h_hat`.
///
/// The cost of [`EitStar::best_path`] never increases.
///
/// # Examples
///
/// ```
/// use rand_core::SeedableRng;
/// use sampling_planning::{
///     Euclidean, GoalRegion, UniformBox, Validity, ValidityChecker,
///     informed::{BitStarConfig, EitStar, EitStarConfig},
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
/// let config = EitStarConfig::new(1.5 * BitStarConfig::gamma_lower_bound(2, 100.0), 0.05);
/// let goal = GoalRegion::new([9.0, 9.0], 0.5);
/// let mut eit = EitStar::new(Euclidean::<2>, sampler, Free, [1.0, 1.0], goal, config)
///     .expect("valid");
/// eit.run(3000, &mut rand_pcg::Pcg64Mcg::seed_from_u64(1));
///
/// assert!(eit.best_path().expect("free space").cost < 11.0);
/// ```
#[derive(Debug, Clone)]
pub struct EitStar<Sp: StateSpace, Sa, C, N = LinearScan<<Sp as StateSpace>::State>> {
    /// The configuration space.
    space: Sp,
    /// Draws the batches.
    sampler: Sa,
    /// Decides which states and motions are allowed.
    checker: C,
    /// The start state; node `0`.
    start: Sp::State,
    /// Where the path has to end.
    goal: GoalRegion<Sp::State>,
    /// The validated settings.
    config: EitStarConfig,
    /// All stored states, by node id; node `0` is the start.
    nodes: Vec<Node<Sp::State>>,
    /// An empty index, cloned for every batch.
    prototype: N,
    /// Index over the vertices and samples of the current batch.
    nn: N,
    /// The forward queue: directed edges `(vertex, target)` with their keys.
    forward_queue: ForwardQueue,
    /// The number of queued forward edges whose target the reverse search has not closed.
    open_targets: usize,
    /// The reverse queue; `from` is the state closer to the goal.
    reverse_queue: BinaryHeap<EdgeEntry>,
    /// Identifies the running reverse search; see [`Node::closed_tag`].
    reverse_tag: usize,
    /// The number of states the reverse search checks per edge now.
    sparse_checks: usize,
    /// Results of edge checks, kept across batches.
    checks: EdgeChecks,
    /// The forward vertices in the goal region.
    solutions: Vec<usize>,
    /// The cost of the best solution; infinite until one is found.
    best_cost: f64,
    /// The neighbor rule of the current batch.
    neighborhood: Neighborhood,
    /// The number of batches started.
    batches: usize,
}

impl<Sp, Sa, C> EitStar<Sp, Sa, C>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
{
    /// Creates a planner with a [`LinearScan`] nearest-neighbor index.
    ///
    /// # Errors
    ///
    /// The same as [`EitStar::with_nearest_neighbors`].
    ///
    /// # Panics
    ///
    /// The same as [`EitStar::with_nearest_neighbors`].
    pub fn new(
        space: Sp,
        sampler: Sa,
        checker: C,
        start: Sp::State,
        goal: GoalRegion<Sp::State>,
        config: EitStarConfig,
    ) -> Result<Self, Error> {
        Self::with_nearest_neighbors(
            space,
            sampler,
            checker,
            start,
            goal,
            config,
            LinearScan::default(),
        )
    }
}

impl<Sp, Sa, C, N> EitStar<Sp, Sa, C, N>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    N: NearestNeighbors<Sp> + Clone,
{
    /// Creates a planner; every batch indexes its states in a clone of the given empty index.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidStart`] if `start` is not valid.
    /// - [`Error::EmptyBatch`] if the batch size is zero.
    /// - [`Error::InvalidRadius`] if `gamma`, the k-nearest constant or `max_radius` is out of
    ///   range.
    ///
    /// # Panics
    ///
    /// Panics if the resolution is not positive and finite, `initial_sparse_checks` is zero or
    /// the suboptimality factor is below one.
    pub fn with_nearest_neighbors(
        space: Sp,
        sampler: Sa,
        checker: C,
        start: Sp::State,
        goal: GoalRegion<Sp::State>,
        config: EitStarConfig,
        nn: N,
    ) -> Result<Self, Error> {
        config.bit.validate()?;
        config.assert_effort();

        let validity = checker.check_state(&start);
        if !validity.is_valid() {
            return Err(Error::InvalidStart(validity));
        }

        let mut planner = Self {
            space,
            sampler,
            checker,
            start: start.clone(),
            goal,
            config,
            nodes: Vec::new(),
            nn: nn.clone(),
            prototype: nn,
            forward_queue: ForwardQueue::default(),
            open_targets: 0,
            reverse_queue: BinaryHeap::new(),
            reverse_tag: 0,
            sparse_checks: config.initial_sparse_checks,
            checks: EdgeChecks::default(),
            solutions: Vec::new(),
            best_cost: f64::INFINITY,
            neighborhood: Neighborhood::Radius(0.0),
            batches: 0,
        };
        planner.plant_root(start);

        Ok(planner)
    }

    /// Stores the start as vertex `0` and the goal center as the first sample, if it is valid.
    fn plant_root(&mut self, start: Sp::State) {
        let root = self.push_node(start, Kind::Vertex);
        self.nodes[root].g = 0.0;

        if self.nodes[root].in_goal() {
            self.solutions.push(root);
            self.best_cost = 0.0;
        }

        let center = self.goal.center().clone();
        if self.checker.check_state(&center).is_valid() {
            self.push_node(center, Kind::Sample);
        }
    }

    /// Stores a new state of the given kind with its heuristics and returns its id.
    fn push_node(&mut self, state: Sp::State, kind: Kind) -> usize {
        let id = self.nodes.len();
        let center_distance = self.space.distance(&state, self.goal.center());

        self.nodes.push(Node {
            g_hat: self.space.distance(&self.start, &state),
            h_hat: (center_distance - self.goal.tolerance()).max(0.0),
            g: f64::INFINITY,
            state,
            kind,
            parent: None,
            children: Vec::new(),
            adjacent: Vec::new(),
            adjacent_batch: 0,
            tree_links: Vec::new(),
            h: f64::INFINITY,
            effort: usize::MAX,
            reverse_parent: None,
            closed_tag: 0,
            incoming: Vec::new(),
        });

        id
    }

    /// Runs one iteration: advances the reverse search if the forward search needs it, else
    /// processes the forward search's next edge, else starts a new batch. Returns whether the
    /// forward tree changed.
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> bool {
        if self.should_continue_reverse() {
            self.reverse_step();
            false
        } else if self.should_continue_forward() {
            self.forward_step()
        } else {
            self.start_batch(rng);
            false
        }
    }

    /// Runs `iterations` iterations.
    pub fn run<R: Rng + ?Sized>(&mut self, iterations: usize, rng: &mut R) {
        for _ in 0..iterations {
            self.step(rng);
        }
    }

    /// Prunes, samples a batch, rebuilds the graph and restarts both searches (OMPL's
    /// `improveApproximation`).
    fn start_batch<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        self.batches += 1;

        if self.best_cost.is_finite() {
            self.prune();
        }

        self.sample_batch(rng);
        self.reindex();
        self.build_graph();
        self.sparse_checks = self.config.initial_sparse_checks;
        self.clear_forward_queue();
        self.restart_reverse_search();
        self.expand_forward(START, None);
    }

    /// Removes the samples and vertices that cannot improve the solution, as BIT*'s
    /// Algorithm 3; vertices that still could become samples again.
    fn prune(&mut self) {
        let c = self.best_cost;

        for node in &mut self.nodes {
            if node.kind == Kind::Sample && node.f_hat() >= c {
                node.kind = Kind::Pruned;
            }
        }

        let mut vertices: Vec<usize> = (0..self.nodes.len())
            .filter(|&id| self.nodes[id].kind == Kind::Vertex)
            .collect();
        vertices.sort_by(|&a, &b| self.nodes[a].g.total_cmp(&self.nodes[b].g).then(a.cmp(&b)));

        for v in vertices {
            let node = &self.nodes[v];
            if node.kind == Kind::Vertex && (node.f_hat() > c || node.g + node.h_hat > c) {
                self.disconnect(v);
            }
        }

        // Children of a pruned vertex cost more than the solution too, so the sweep above
        // removed them as well; this keeps the tree whole should rounding disagree.
        for v in 1..self.nodes.len() {
            if self.nodes[v].kind == Kind::Vertex && self.nodes[v].parent.is_none() {
                self.disconnect(v);
            }
        }

        self.solutions
            .retain(|&id| self.nodes[id].kind == Kind::Vertex);
    }

    /// Removes vertex `v` from the forward tree; it becomes a sample if it could still improve
    /// the solution and is pruned otherwise.
    fn disconnect(&mut self, v: usize) {
        if let Some(parent) = self.nodes[v].parent.take() {
            self.nodes[parent].children.retain(|&child| child != v);
        }

        let recycle = self.nodes[v].f_hat() < self.best_cost;
        let node = &mut self.nodes[v];
        node.kind = if recycle { Kind::Sample } else { Kind::Pruned };
        node.g = f64::INFINITY;

        for child in core::mem::take(&mut node.children) {
            self.nodes[child].parent = None;
        }
    }

    /// Draws up to `batch_size` valid samples that can improve the solution.
    fn sample_batch<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        for _ in 0..self.config.bit.batch_size {
            let informed = Informed {
                space: &self.space,
                checker: &self.checker,
                start: &self.start,
                goal: &self.goal,
                best_cost: self.best_cost,
            };

            if let Some(state) = informed.draw(&mut self.sampler, rng) {
                self.push_node(state, Kind::Sample);
            }
        }
    }

    /// Rebuilds the index over all live states and fixes this batch's neighbor rule.
    fn reindex(&mut self) {
        self.nn = self.prototype.clone();
        let mut live: usize = 0;

        for (id, node) in self.nodes.iter().enumerate() {
            if node.kind != Kind::Pruned {
                self.nn.insert(id, node.state.clone());
                live += 1;
            }
        }

        // As OMPL: every informed state counts, the new batch included, and the radius
        // shrinks with the informed set's measure.
        let q = live.max(2);
        let bound = self.best_cost + self.goal.tolerance();
        let fraction = self
            .sampler
            .informed_fraction(&self.space, &self.start, self.goal.center(), bound)
            .unwrap_or(1.0);
        self.neighborhood =
            Neighborhood::for_batch(&self.config.bit, q, self.space.dimension(), fraction);
    }

    /// Rebuilds the batch graph: symmetrized neighbors of every live state plus the forward
    /// tree's own edges, minus nothing (invalid edges are skipped when used).
    ///
    /// With a connection radius the neighbor lists are built lazily, on a state's first use
    /// in the batch ([`EitStar::ensure_adjacent`]), as OMPL does: a radius query from `x`
    /// finds exactly the states whose own query would find `x` (the distance is symmetric),
    /// so a lazy list equals the symmetrized one, and states the searches never touch cost
    /// nothing. `k` nearest neighbors are not symmetric, and neither are radius neighborhoods
    /// in a space that is not [symmetric](StateSpace::is_symmetric), such as
    /// [`Dubins`](crate::Dubins); those build the whole symmetrized graph here.
    fn build_graph(&mut self) {
        for node in &mut self.nodes {
            node.adjacent.clear();
            node.tree_links.clear();
        }

        for id in 0..self.nodes.len() {
            if self.nodes[id].kind == Kind::Pruned {
                continue;
            }

            if let Some(parent) = self.nodes[id].parent {
                self.nodes[id].tree_links.push(parent);
                self.nodes[parent].tree_links.push(id);
            }
        }

        if matches!(self.neighborhood, Neighborhood::Nearest(_)) || !self.space.is_symmetric() {
            self.build_symmetrized_graph();
        }
    }

    /// Builds every live state's neighbor list for the `k`-nearest rule: its own neighbors,
    /// those that have it as a neighbor, and its tree links.
    fn build_symmetrized_graph(&mut self) {
        let mut near = Vec::new();

        for id in 0..self.nodes.len() {
            if self.nodes[id].kind == Kind::Pruned {
                continue;
            }

            self.query_neighbors(id, &mut near);
            for &other in &near {
                self.nodes[id].adjacent.push(other);
                self.nodes[other].adjacent.push(id);
            }
        }

        for id in 0..self.nodes.len() {
            let links = core::mem::take(&mut self.nodes[id].tree_links);
            let node = &mut self.nodes[id];
            node.adjacent.extend_from_slice(&links);
            node.adjacent.sort_unstable();
            node.adjacent.dedup();
            node.adjacent_batch = self.batches;
            node.tree_links = links;
        }
    }

    /// Builds the neighbor list of `x` for this batch unless it is built already: the states
    /// within the connection radius and the tree links, sorted. Pruned states have none.
    pub(super) fn ensure_adjacent(&mut self, x: usize) {
        if self.nodes[x].adjacent_batch == self.batches {
            return;
        }

        let mut near = core::mem::take(&mut self.nodes[x].adjacent);
        near.clear();

        if self.nodes[x].kind != Kind::Pruned {
            self.query_neighbors(x, &mut near);
            near.extend_from_slice(&self.nodes[x].tree_links);
            near.sort_unstable();
            near.dedup();
        }

        let node = &mut self.nodes[x];
        node.adjacent = near;
        node.adjacent_batch = self.batches;
    }

    /// Replaces `out` with the batch neighbors of `id` under the batch rule, without `id`
    /// itself and without edges longer than `max_radius`.
    fn query_neighbors(&self, id: usize, out: &mut Vec<usize>) {
        let cap = self.config.bit.max_radius;
        let keep = |x| x != id && (cap.is_infinite() || self.cost_estimate(id, x) <= cap);

        self.neighborhood
            .query(&self.space, &self.nn, &self.nodes[id].state, keep, out);
    }

    /// The admissible edge cost estimate `c_hat = d(from, to)`.
    fn cost_estimate(&self, from: usize, to: usize) -> f64 {
        self.space
            .distance(&self.nodes[from].state, &self.nodes[to].state)
    }

    /// Whether node `id` is a goal state of the reverse search.
    fn is_goal(&self, id: usize) -> bool {
        self.nodes[id].kind != Kind::Pruned && self.nodes[id].in_goal()
    }

    /// The cheapest path found so far, or `None` if no vertex reached the goal yet. Its cost
    /// never increases as iterations run.
    #[must_use]
    pub fn best_path(&self) -> Option<Path<Sp::State>> {
        let best = self
            .solutions
            .iter()
            .copied()
            .min_by(|&a, &b| self.nodes[a].g.total_cmp(&self.nodes[b].g).then(a.cmp(&b)))?;
        let mut states = Vec::new();
        let mut current = Some(best);

        while let Some(id) = current {
            states.push(self.nodes[id].state.clone());
            current = self.nodes[id].parent;
        }

        states.reverse();

        Some(Path {
            states,
            cost: self.nodes[best].g,
        })
    }

    /// The number of forward-tree vertices, including the start.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.count(Kind::Vertex)
    }

    /// The number of samples not connected to the forward tree.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.count(Kind::Sample)
    }

    /// The number of batches started so far.
    #[must_use]
    pub const fn batch_count(&self) -> usize {
        self.batches
    }

    /// The number of states the reverse search currently checks per edge.
    #[must_use]
    pub const fn sparse_check_count(&self) -> usize {
        self.sparse_checks
    }

    /// The number of stored states of `kind`.
    fn count(&self, kind: Kind) -> usize {
        self.nodes.iter().filter(|node| node.kind == kind).count()
    }
}
