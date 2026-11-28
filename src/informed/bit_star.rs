//! BIT*: batch informed trees (Gammell, Barfoot and Srinivasa 2020).

use alloc::{
    collections::{BTreeSet, BinaryHeap},
    vec::Vec,
};

use rand_core::Rng;

use core::fmt;

use super::Error;
use super::abit_star::{AbitStarConfig, Search};
use super::batch::Neighborhood;
use super::batch_graph::{BatchGraph, Extension, Kind, START};
use super::queues::{EdgeEntry, VertexEntry};
use crate::radius::k_constant_lower_bound;
#[cfg(doc)]
use crate::trees::INFORMED_ATTEMPTS;
use crate::trees::RrtStarConfig;
use crate::{GoalRegion, LinearScan, NearestNeighbors, Path, Sampler, StateSpace, ValidityChecker};

/// Settings for [`BitStar`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BitStarConfig {
    /// The number of samples per batch, `m` in the paper. At least one.
    pub batch_size: usize,
    /// The connection radius constant: with `q` states in the graph (vertices plus unconnected
    /// samples, minus the newest batch), states within `gamma * (ln q / q)^(1/d)` are
    /// neighbors. Asymptotic optimality needs `gamma` above
    /// [`BitStarConfig::gamma_lower_bound`]. Positive and finite. Unused when
    /// [`BitStarConfig::k_nearest`] is set.
    pub gamma: f64,
    /// When `Some(k_constant)`, the neighbors are the `ceil(k_constant * ln q)` nearest states
    /// instead (eq. 2 of the paper); asymptotic optimality needs `k_constant` above
    /// [`BitStarConfig::k_constant_lower_bound`]. Positive and finite.
    pub k_nearest: Option<f64>,
    /// Upper bound on the connection radius and on every edge, `r_max` in the paper. Positive;
    /// may be infinite.
    pub max_radius: f64,
}

impl BitStarConfig {
    /// A radius-based config with 100 samples per batch and no edge-length cap.
    #[must_use]
    pub const fn new(gamma: f64) -> Self {
        Self {
            batch_size: 100,
            gamma,
            k_nearest: None,
            max_radius: f64::INFINITY,
        }
    }

    /// The smallest `gamma` for which BIT* is asymptotically optimal (Gammell et al. 2020,
    /// eq. 1): `(2 (1 + 1/d) free_volume / zeta_d)^(1/d)`, the RRT* bound.
    ///
    /// The paper uses the smaller of the free volume and the informed-set volume, which
    /// shrinks the radius once a solution exists; using the free volume throughout keeps the
    /// bound admissible at the price of larger neighborhoods.
    #[must_use]
    pub fn gamma_lower_bound(dimension: usize, free_volume: f64) -> f64 {
        RrtStarConfig::gamma_lower_bound(dimension, free_volume)
    }

    /// The smallest k-nearest constant for which BIT* is asymptotically optimal (Gammell et
    /// al. 2020, eq. 2): `e (1 + 1/d)`.
    #[must_use]
    pub fn k_constant_lower_bound(dimension: usize) -> f64 {
        k_constant_lower_bound(dimension)
    }

    /// Checks every field against its documented range.
    pub(super) fn validate(&self) -> Result<(), Error> {
        let constant = self.k_nearest.unwrap_or(self.gamma);

        if self.batch_size == 0 {
            return Err(Error::EmptyBatch);
        }

        if !(constant.is_finite() && constant > 0.0)
            || self.max_radius.is_nan()
            || self.max_radius <= 0.0
        {
            return Err(Error::InvalidRadius);
        }

        Ok(())
    }
}

/// BIT*'s batch bookkeeping of one stored state.
#[derive(Debug, Clone)]
struct BitExt {
    /// A vertex that has not been expanded yet (`V_unexpnd`).
    unexpanded: bool,
    /// A sample added in the current batch, sampled or recycled (`X_new`).
    fresh: bool,
    /// Whether the vertex is waiting in the vertex queue.
    queued: bool,
}

impl Default for BitExt {
    fn default() -> Self {
        Self {
            unexpanded: false,
            fresh: true,
            queued: false,
        }
    }
}

impl Extension for BitExt {
    /// A recycled vertex is a fresh sample of the new batch and no longer waits for expansion.
    fn disconnected(&mut self, recycled: bool) {
        self.fresh = recycled;
        self.unexpanded = false;
    }
}

/// BIT*: batch informed trees (Gammell, Srinivasa and Barfoot 2015; Gammell, Barfoot and
/// Srinivasa 2020).
///
/// BIT* treats each batch of samples as an implicit random geometric graph and builds a tree
/// over it in order of the estimated solution cost through each edge,
/// `g_T(v) + c_hat(v, x) + h_hat(x)`, with an edge queue fed by expanding vertices. Edges are
/// motion-checked only when they come out on top and could still improve the tree and the
/// solution. When the best edge cannot improve the current solution, the batch is over: the
/// planner prunes every state that cannot improve it (Algorithm 3), samples a new batch from
/// the informed set and starts again. With one batch of infinitely many samples it would be
/// Lifelong-Planning A* on the RGG; with batches of one sample it behaves like RRT*.
///
/// This follows Algorithms 1 to 3 of the 2020 paper, with these choices:
///
/// - The heuristics are `g_hat(x) = d(start, x)`, `h_hat(x) = max(0, d(x, goal center) -
///   tolerance)` and `c_hat = d`. The goal center is the first sample when it is valid; any
///   vertex inside the goal region is a solution.
/// - Batches are drawn through [`Sampler::sample_informed`] with foci start and goal center
///   and bound `c_best + tolerance`, falling back to rejection, and then filtered to valid
///   states with `f_hat < c_best`; a batch can come out smaller than `batch_size` when
///   [`INFORMED_ATTEMPTS`] draws per sample do not find one.
/// - Queue keys go stale when a rewiring lowers the costs of a subtree; both queues are then
///   re-sorted before the next pop, as OMPL does.
/// - Edges are checked in travel direction, from tree parent to child.
///
/// The cost of [`BitStar::best_path`] never increases. Pruned states stay in memory until the
/// planner is dropped.
///
/// # Examples
///
/// ```
/// use rand_core::SeedableRng;
/// use sampling_planning::{
///     Euclidean, GoalRegion, UniformBox, Validity, ValidityChecker,
///     informed::{BitStar, BitStarConfig},
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
/// let config = BitStarConfig::new(1.5 * BitStarConfig::gamma_lower_bound(2, 100.0));
/// let goal = GoalRegion::new([9.0, 9.0], 0.5);
/// let mut bit = BitStar::new(Euclidean::<2>, sampler, Free, [1.0, 1.0], goal, config)
///     .expect("valid");
/// bit.run(2000, &mut rand_pcg::Pcg64Mcg::seed_from_u64(1));
///
/// assert!(bit.best_path().expect("free space").cost < 11.0);
/// ```
#[derive(Clone)]
pub struct BitStar<Sp: StateSpace, Sa, C, N = LinearScan<<Sp as StateSpace>::State>> {
    /// The problem and the stored states with the forward tree; BIT* uses no batch graph.
    graph: BatchGraph<Sp, Sa, C, N, BitExt, ()>,
    /// The vertex queue `Q_V`.
    vertex_queue: BinaryHeap<VertexEntry>,
    /// The edge queue `Q_E`.
    edge_queue: BinaryHeap<EdgeEntry>,
    /// Whether a rewiring changed costs since the queues were keyed.
    stale_keys: bool,
    /// Reused buffer for neighbor queries.
    near: Vec<usize>,
    /// ABIT*'s inflation and truncation state; `None` for plain BIT*.
    search: Option<Search>,
    /// ABIT* only: edges `(from, to)` found invalid, so that the final search on a batch does
    /// not check them again. Ids are stable because pruned states stay stored.
    invalid: BTreeSet<(usize, usize)>,
}

impl<Sp, Sa, C, N> fmt::Debug for BitStar<Sp, Sa, C, N>
where
    Sp: StateSpace + fmt::Debug,
    Sp::State: fmt::Debug,
    Sa: fmt::Debug,
    C: fmt::Debug,
    N: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BitStar")
            .field("graph", &self.graph)
            .field("vertex_queue", &self.vertex_queue)
            .field("edge_queue", &self.edge_queue)
            .field("stale_keys", &self.stale_keys)
            .field("near", &self.near)
            .field("search", &self.search)
            .field("invalid", &self.invalid)
            .finish()
    }
}

impl<Sp, Sa, C> BitStar<Sp, Sa, C>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
{
    /// Creates a planner with a [`LinearScan`] nearest-neighbor index.
    ///
    /// # Errors
    ///
    /// The same as [`BitStar::with_nearest_neighbors`].
    pub fn new(
        space: Sp,
        sampler: Sa,
        checker: C,
        start: Sp::State,
        goal: GoalRegion<Sp::State>,
        config: BitStarConfig,
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

impl<Sp, Sa, C, N> BitStar<Sp, Sa, C, N>
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
    pub fn with_nearest_neighbors(
        space: Sp,
        sampler: Sa,
        checker: C,
        start: Sp::State,
        goal: GoalRegion<Sp::State>,
        config: BitStarConfig,
        nn: N,
    ) -> Result<Self, Error> {
        let mut graph: BatchGraph<_, _, _, _, BitExt, ()> =
            BatchGraph::new(space, sampler, checker, start, goal, config, nn)?;
        graph.nodes[START].ext.unexpanded = true;

        Ok(Self {
            graph,
            vertex_queue: BinaryHeap::new(),
            edge_queue: BinaryHeap::new(),
            stale_keys: false,
            near: Vec::new(),
            search: None,
            invalid: BTreeSet::new(),
        })
    }

    /// Turns this planner into ABIT* with the given factors; see [`AbitStarConfig`].
    pub(super) fn advanced(mut self, config: &AbitStarConfig) -> Self {
        self.search = Some(Search::new(config));

        self
    }

    /// Runs one iteration: starts a new batch if both queues are empty, expands vertices while
    /// the vertex queue is ahead, then processes the best edge. Returns whether the tree
    /// changed.
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> bool {
        let truncated = self.search.is_some_and(|search| search.is_truncated());
        if truncated || (self.vertex_queue.is_empty() && self.edge_queue.is_empty()) {
            self.start_search(rng);
        }

        self.resort_if_stale();

        while self.best_vertex_value() <= self.best_edge_value() {
            if !self.expand_next_vertex() {
                break;
            }
        }

        match self.edge_queue.pop() {
            Some(edge) => self.process_edge(edge),
            None => false,
        }
    }

    /// Runs `iterations` iterations.
    pub fn run<R: Rng + ?Sized>(&mut self, iterations: usize, rng: &mut R) {
        for _ in 0..iterations {
            self.step(rng);
        }
    }

    /// Starts the next search: a new batch, or for ABIT* with a solution the final, less
    /// inflated search on the current batch after its inflated one.
    fn start_search<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        let has_solution = self.graph.best_cost.is_finite();
        let Some(mut search) = self.search else {
            self.start_batch(rng);
            return;
        };

        if search.wants_final_search(has_solution) {
            // The queues survive the inflated search; re-keying them deflates the heuristic.
            search.begin_final_search(self.live_states());
            self.search = Some(search);
            self.stale_keys = true;
        } else {
            self.vertex_queue.clear();
            self.edge_queue.clear();
            search.begin_batch();
            self.search = Some(search);
            self.start_batch(rng);
            search.truncate_for(self.live_states());
            self.search = Some(search);
        }
    }

    /// The number of vertices and samples that are not pruned, `|V| + |X_unconn|`.
    fn live_states(&self) -> usize {
        self.graph
            .nodes
            .iter()
            .filter(|node| node.kind != Kind::Pruned)
            .count()
    }

    /// Prunes, samples a batch, re-indexes and queues every vertex (Algorithm 1, lines 7-12).
    fn start_batch<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        self.graph.batches += 1;

        for node in &mut self.graph.nodes {
            node.ext.fresh = false;
        }

        if self.graph.best_cost.is_finite() {
            self.graph.prune();
        }

        let sampled = self.graph.sample_batch(rng);
        self.reindex(sampled);

        for id in 0..self.graph.nodes.len() {
            if self.graph.nodes[id].kind == Kind::Vertex {
                self.queue_vertex(id);
            }
        }
    }

    /// Rebuilds the index over all vertices and samples and fixes this batch's neighbor rule
    /// from `q = |V| + |X_unconn| - m` (eqs. 1 and 2).
    fn reindex(&mut self, sampled: usize) {
        let live = self.graph.rebuild_index();
        let q = live.saturating_sub(sampled).max(2);
        self.graph.neighborhood =
            Neighborhood::for_batch(&self.graph.config, q, self.graph.space.dimension(), 1.0);
    }

    /// Queues vertex `v` for expansion.
    fn queue_vertex(&mut self, v: usize) {
        let inflation = self.inflation();
        let node = &mut self.graph.nodes[v];
        node.ext.queued = true;
        self.vertex_queue.push(VertexEntry {
            key: node.g + inflation * node.h_hat,
            cost: node.g,
            vertex: v,
        });
    }

    /// Re-keys both queues after a rewiring lowered the costs of a subtree.
    fn resort_if_stale(&mut self) {
        if !self.stale_keys {
            return;
        }

        self.stale_keys = false;

        let vertices: Vec<usize> = self
            .vertex_queue
            .drain()
            .map(|entry| entry.vertex)
            .collect();
        for v in vertices {
            if self.graph.nodes[v].ext.queued && self.graph.nodes[v].kind == Kind::Vertex {
                self.graph.nodes[v].ext.queued = false;
                self.queue_vertex(v);
            }
        }

        let edges: Vec<EdgeEntry> = self.edge_queue.drain().collect();
        for edge in edges {
            self.queue_edge(edge.from, edge.to);
        }
    }

    /// The best vertex-queue value `g_T(v) + h_hat(v)`, or infinity, dropping stale entries.
    fn best_vertex_value(&mut self) -> f64 {
        while let Some(entry) = self.vertex_queue.peek() {
            let node = &self.graph.nodes[entry.vertex];

            if node.ext.queued && node.kind == Kind::Vertex {
                return entry.key;
            }

            self.vertex_queue.pop();
        }

        f64::INFINITY
    }

    /// The best edge-queue value `g_T(v) + c_hat(v, x) + h_hat(x)`, or infinity.
    fn best_edge_value(&self) -> f64 {
        self.edge_queue
            .peek()
            .map_or(f64::INFINITY, |edge| edge.key[0])
    }

    /// Pops the best vertex and queues its promising edges (Algorithm 2). Returns `false` when
    /// the vertex queue is empty.
    fn expand_next_vertex(&mut self) -> bool {
        let Some(entry) = self.vertex_queue.pop() else {
            return false;
        };

        let v = entry.vertex;
        if !self.graph.nodes[v].ext.queued || self.graph.nodes[v].kind != Kind::Vertex {
            return true;
        }

        self.graph.nodes[v].ext.queued = false;

        let mut near = core::mem::take(&mut self.near);
        self.neighbors(v, &mut near);

        let unexpanded = self.graph.nodes[v].ext.unexpanded;
        // ABIT* cascades rewirings: every expansion of a vertex may rewire, not just its first.
        let rewires = unexpanded || self.search.is_some();
        for &x in &near {
            if self.is_sample_edge_candidate(v, x, unexpanded)
                || self.is_rewire_candidate(v, x, rewires)
            {
                self.queue_edge(v, x);
            }
        }

        self.graph.nodes[v].ext.unexpanded = false;
        self.near = near;

        true
    }

    /// Whether the edge from vertex `v` to `x` is a candidate edge to an unconnected sample:
    /// a fresh one, or any if `v` was never expanded, whose estimated solution cost beats the
    /// current one.
    fn is_sample_edge_candidate(&self, v: usize, x: usize, unexpanded: bool) -> bool {
        let target = &self.graph.nodes[x];

        target.kind == Kind::Sample
            && (unexpanded || target.ext.fresh)
            && self.graph.nodes[v].g_hat + self.edge_estimate(v, x) + target.h_hat
                < self.graph.best_cost
    }

    /// Whether the edge from vertex `v` to vertex `w` is a rewiring candidate: `v` may rewire
    /// (it was never expanded, or ABIT* cascades rewirings), the edge is not a tree edge, and
    /// it could improve both `w` and the solution.
    fn is_rewire_candidate(&self, v: usize, w: usize, rewires: bool) -> bool {
        let target = &self.graph.nodes[w];
        let through = self.graph.nodes[v].g_hat + self.edge_estimate(v, w);

        rewires
            && target.kind == Kind::Vertex
            && w != v
            && target.parent != Some(v)
            && through + target.h_hat < self.graph.best_cost
            && through < target.g
    }

    /// Replaces `out` with the vertices and samples near node `v` under this batch's rule,
    /// excluding `v`.
    fn neighbors(&self, v: usize, out: &mut Vec<usize>) {
        let cap = self.graph.config.max_radius;
        let keep = |x| x != v && (cap.is_infinite() || self.edge_estimate(v, x) <= cap);

        self.graph.neighborhood.query(
            &self.graph.space,
            &self.graph.nn,
            &self.graph.nodes[v].state,
            keep,
            out,
        );
    }

    /// The admissible edge cost estimate `c_hat(v, x) = d(v, x)`.
    fn edge_estimate(&self, v: usize, x: usize) -> f64 {
        self.graph
            .space
            .distance(&self.graph.nodes[v].state, &self.graph.nodes[x].state)
    }

    /// Queues the edge from vertex `v` to `x` with its current keys.
    fn queue_edge(&mut self, v: usize, x: usize) {
        let g = self.graph.nodes[v].g;
        let through = g + self.edge_estimate(v, x);

        self.edge_queue.push(EdgeEntry {
            key: [
                through + self.inflation() * self.graph.nodes[x].h_hat,
                through,
                g,
            ],
            from: v,
            to: x,
        });
    }

    /// Processes the best edge (Algorithm 1, lines 15-34): adds or rewires its target when
    /// the checked edge improves both the target and the solution, or ends the batch when even
    /// the estimate cannot improve the solution. Returns whether the tree changed.
    fn process_edge(&mut self, edge: EdgeEntry) -> bool {
        let (v, x) = (edge.from, edge.to);

        if self.graph.nodes[v].kind != Kind::Vertex || self.graph.nodes[x].kind == Kind::Pruned {
            return false;
        }

        let g_v = self.graph.nodes[v].g;
        let estimate = g_v + self.edge_estimate(v, x);

        if self.truncation() * (estimate + self.graph.nodes[x].h_hat) >= self.graph.best_cost {
            self.end_search(edge);
            return false;
        }

        if estimate >= self.graph.nodes[x].g || !self.is_edge_valid(v, x) {
            return false;
        }

        // With c = c_hat for a valid edge, the checked conditions equal the estimated ones.
        self.connect(v, x, estimate);

        true
    }

    /// Ends the current search at `edge`, the first that cannot improve the solution enough.
    ///
    /// BIT* clears both queues, which ends the batch. ABIT*'s inflated search keeps them, with
    /// `edge` put back, for the final search on the same batch to continue from.
    fn end_search(&mut self, edge: EdgeEntry) {
        let search = self
            .search
            .as_mut()
            .filter(|search| search.wants_final_search(true));

        if let Some(search) = search {
            search.truncate();
            self.edge_queue.push(edge);
        } else {
            self.vertex_queue.clear();
            self.edge_queue.clear();
        }
    }

    /// Whether the motion from node `from` to node `to` checks as valid; ABIT* remembers
    /// invalid edges and never checks them twice.
    fn is_edge_valid(&mut self, from: usize, to: usize) -> bool {
        if self.search.is_some() && self.invalid.contains(&(from, to)) {
            return false;
        }

        let valid = self
            .graph
            .checker
            .check_motion(
                &self.graph.space,
                &self.graph.nodes[from].state,
                &self.graph.nodes[to].state,
            )
            .is_valid();

        if !valid && self.search.is_some() {
            self.invalid.insert((from, to));
        }

        valid
    }

    /// The heuristic inflation factor of the current search; `1` for BIT*.
    pub(super) fn inflation(&self) -> f64 {
        self.search.map_or(1.0, |search| search.inflation())
    }

    /// The truncation factor of the current search; `1` for BIT*.
    pub(super) fn truncation(&self) -> f64 {
        self.search.map_or(1.0, |search| search.truncation())
    }

    /// Makes `v` the parent of `x` with cost `g`: adds the sample `x` to the tree, or rewires
    /// the vertex `x` and lowers the costs of its subtree. Updates the solution.
    fn connect(&mut self, v: usize, x: usize, g: f64) {
        if let Some(old) = self.graph.nodes[x].parent {
            self.graph.nodes[old].children.retain(|&child| child != x);
            self.stale_keys = true;
        }

        let was_sample = self.graph.nodes[x].kind == Kind::Sample;
        let node = &mut self.graph.nodes[x];
        node.parent = Some(v);
        node.g = g;
        self.graph.nodes[v].children.push(x);

        if was_sample {
            let node = &mut self.graph.nodes[x];
            node.kind = Kind::Vertex;
            node.ext.unexpanded = true;
            self.queue_vertex(x);

            if self.graph.nodes[x].in_goal() {
                self.graph.solutions.push(x);
            }
        } else {
            self.propagate_costs(x);
        }

        self.graph.update_best_cost();
    }

    /// Recomputes the costs of all descendants of `x` from their parents' costs.
    fn propagate_costs(&mut self, x: usize) {
        let mut stack = self.graph.nodes[x].children.clone();

        while let Some(child) = stack.pop() {
            let parent = self.graph.nodes[child]
                .parent
                .expect("a child has a parent");
            self.graph.nodes[child].g =
                self.graph.nodes[parent].g + self.edge_estimate(parent, child);
            stack.extend_from_slice(&self.graph.nodes[child].children);
        }
    }

    /// The cheapest path found so far, or `None` if no vertex reached the goal yet. Its cost
    /// never increases as iterations run.
    #[must_use]
    pub fn best_path(&self) -> Option<Path<Sp::State>> {
        self.graph.best_path()
    }

    /// The number of tree vertices, including the start.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.graph.count(Kind::Vertex)
    }

    /// The number of samples not connected to the tree.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.graph.count(Kind::Sample)
    }

    /// The number of batches started so far.
    #[must_use]
    pub const fn batch_count(&self) -> usize {
        self.graph.batches
    }
}
