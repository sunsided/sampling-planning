//! BIT*: batch informed trees (Gammell, Barfoot and Srinivasa 2020).

use alloc::{collections::BinaryHeap, vec::Vec};

use rand_core::Rng;

use super::Error;
use super::batch::{Informed, Neighborhood};
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

/// What a stored state currently is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// A tree vertex.
    Vertex,
    /// A sample not connected to the tree (`X_unconn`).
    Sample,
    /// Pruned for good: it cannot improve the solution.
    Pruned,
}

/// One stored state: a vertex or a sample.
#[derive(Debug, Clone)]
struct Node<S> {
    /// The state.
    state: S,
    /// What the state currently is.
    kind: Kind,
    /// The cost-to-come through the tree, `g_T`; infinite for samples.
    g: f64,
    /// The admissible cost-to-come `g_hat = d(start, x)`.
    g_hat: f64,
    /// The admissible cost-to-go `h_hat = max(0, d(x, goal center) - tolerance)`.
    h_hat: f64,
    /// The tree parent; `None` for the root and samples.
    parent: Option<usize>,
    /// The tree children.
    children: Vec<usize>,
    /// A vertex that has not been expanded yet (`V_unexpnd`).
    unexpanded: bool,
    /// A sample added in the current batch, sampled or recycled (`X_new`).
    fresh: bool,
    /// Whether the vertex is waiting in the vertex queue.
    queued: bool,
}

impl<S> Node<S> {
    /// The admissible solution cost through this state, `f_hat = g_hat + h_hat`.
    fn f_hat(&self) -> f64 {
        self.g_hat + self.h_hat
    }

    /// Whether the state lies in the goal region, where the cost-to-go estimate is zero.
    fn in_goal(&self) -> bool {
        self.h_hat <= 0.0
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
#[derive(Debug, Clone)]
pub struct BitStar<Sp: StateSpace, Sa, C, N = LinearScan<<Sp as StateSpace>::State>> {
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
    config: BitStarConfig,
    /// All stored states, by node id; node `0` is the start.
    nodes: Vec<Node<Sp::State>>,
    /// An empty index, cloned for every batch.
    prototype: N,
    /// Index over the vertices and samples of the current batch.
    nn: N,
    /// The vertex queue `Q_V`.
    vertex_queue: BinaryHeap<VertexEntry>,
    /// The edge queue `Q_E`.
    edge_queue: BinaryHeap<EdgeEntry>,
    /// Whether a rewiring changed costs since the queues were keyed.
    stale_keys: bool,
    /// The vertices inside the goal region (`V_sol'n`).
    solutions: Vec<usize>,
    /// The cost of the best solution, `c_i`; infinite until one is found.
    best_cost: f64,
    /// The connection radius of the current batch, or the neighbor count for k-nearest.
    neighborhood: Neighborhood,
    /// The number of batches started.
    batches: usize,
    /// Reused buffer for neighbor queries.
    near: Vec<usize>,
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
        config.validate()?;

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
            vertex_queue: BinaryHeap::new(),
            edge_queue: BinaryHeap::new(),
            stale_keys: false,
            solutions: Vec::new(),
            best_cost: f64::INFINITY,
            neighborhood: Neighborhood::Radius(0.0),
            batches: 0,
            near: Vec::new(),
        };
        planner.plant_root(start);

        Ok(planner)
    }

    /// Stores the start as vertex `0` and the goal center as the first sample, if it is valid.
    fn plant_root(&mut self, start: Sp::State) {
        let root = self.push_node(start, Kind::Vertex);
        self.nodes[root].g = 0.0;
        self.nodes[root].unexpanded = true;

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
            unexpanded: false,
            fresh: true,
            queued: false,
        });

        id
    }

    /// Runs one iteration: starts a new batch if both queues are empty, expands vertices while
    /// the vertex queue is ahead, then processes the best edge. Returns whether the tree
    /// changed.
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> bool {
        if self.vertex_queue.is_empty() && self.edge_queue.is_empty() {
            self.start_batch(rng);
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

    /// Prunes, samples a batch, re-indexes and queues every vertex (Algorithm 1, lines 7-12).
    fn start_batch<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        self.batches += 1;

        for node in &mut self.nodes {
            node.fresh = false;
        }

        if self.best_cost.is_finite() {
            self.prune();
        }

        let sampled = self.sample_batch(rng);
        self.reindex(sampled);

        for id in 0..self.nodes.len() {
            if self.nodes[id].kind == Kind::Vertex {
                self.queue_vertex(id);
            }
        }
    }

    /// Removes the samples and vertices that cannot improve the solution; vertices that still
    /// could as samples are recycled into the new batch (Algorithm 3).
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

    /// Removes vertex `v` from the tree; it becomes a fresh sample if it could still improve
    /// the solution and is pruned otherwise. Its children are pruned by the caller's loop,
    /// since their costs exceed the solution cost too.
    fn disconnect(&mut self, v: usize) {
        if let Some(parent) = self.nodes[v].parent.take() {
            self.nodes[parent].children.retain(|&child| child != v);
        }

        let recycle = self.nodes[v].f_hat() < self.best_cost;
        let node = &mut self.nodes[v];
        node.kind = if recycle { Kind::Sample } else { Kind::Pruned };
        node.g = f64::INFINITY;
        node.fresh = recycle;
        node.unexpanded = false;

        for child in core::mem::take(&mut node.children) {
            self.nodes[child].parent = None;
        }
    }

    /// Draws up to `batch_size` valid samples that can improve the solution and returns how
    /// many were added.
    fn sample_batch<R: Rng + ?Sized>(&mut self, rng: &mut R) -> usize {
        let mut added = 0;

        for _ in 0..self.config.batch_size {
            let informed = Informed {
                space: &self.space,
                checker: &self.checker,
                start: &self.start,
                goal: &self.goal,
                best_cost: self.best_cost,
            };

            if let Some(state) = informed.draw(&mut self.sampler, rng) {
                self.push_node(state, Kind::Sample);
                added += 1;
            }
        }

        added
    }

    /// Rebuilds the index over all vertices and samples and fixes this batch's neighbor rule
    /// from `q = |V| + |X_unconn| - m` (eqs. 1 and 2).
    fn reindex(&mut self, sampled: usize) {
        self.nn = self.prototype.clone();
        let mut live: usize = 0;

        for (id, node) in self.nodes.iter().enumerate() {
            if node.kind != Kind::Pruned {
                self.nn.insert(id, node.state.clone());
                live += 1;
            }
        }

        let q = live.saturating_sub(sampled).max(2);
        self.neighborhood = Neighborhood::for_batch(&self.config, q, self.space.dimension());
    }

    /// Queues vertex `v` for expansion.
    fn queue_vertex(&mut self, v: usize) {
        let node = &mut self.nodes[v];
        node.queued = true;
        self.vertex_queue.push(VertexEntry {
            key: node.g + node.h_hat,
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
            if self.nodes[v].queued && self.nodes[v].kind == Kind::Vertex {
                self.nodes[v].queued = false;
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
            let node = &self.nodes[entry.vertex];

            if node.queued && node.kind == Kind::Vertex {
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
        if !self.nodes[v].queued || self.nodes[v].kind != Kind::Vertex {
            return true;
        }

        self.nodes[v].queued = false;

        let mut near = core::mem::take(&mut self.near);
        self.neighbors(v, &mut near);

        let unexpanded = self.nodes[v].unexpanded;
        for &x in &near {
            if self.is_sample_edge_candidate(v, x, unexpanded)
                || self.is_rewire_candidate(v, x, unexpanded)
            {
                self.queue_edge(v, x);
            }
        }

        self.nodes[v].unexpanded = false;
        self.near = near;

        true
    }

    /// Whether the edge from vertex `v` to `x` is a candidate edge to an unconnected sample:
    /// a fresh one, or any if `v` was never expanded, whose estimated solution cost beats the
    /// current one.
    fn is_sample_edge_candidate(&self, v: usize, x: usize, unexpanded: bool) -> bool {
        let target = &self.nodes[x];

        target.kind == Kind::Sample
            && (unexpanded || target.fresh)
            && self.nodes[v].g_hat + self.edge_estimate(v, x) + target.h_hat < self.best_cost
    }

    /// Whether the edge from the never-expanded vertex `v` to vertex `w` is a rewiring
    /// candidate: not a tree edge, and it could improve both `w` and the solution.
    fn is_rewire_candidate(&self, v: usize, w: usize, unexpanded: bool) -> bool {
        let target = &self.nodes[w];
        let through = self.nodes[v].g_hat + self.edge_estimate(v, w);

        unexpanded
            && target.kind == Kind::Vertex
            && w != v
            && target.parent != Some(v)
            && through + target.h_hat < self.best_cost
            && through < target.g
    }

    /// Replaces `out` with the vertices and samples near node `v` under this batch's rule,
    /// excluding `v`.
    fn neighbors(&self, v: usize, out: &mut Vec<usize>) {
        let cap = self.config.max_radius;
        let keep = |x| x != v && (cap.is_infinite() || self.edge_estimate(v, x) <= cap);

        self.neighborhood
            .query(&self.space, &self.nn, &self.nodes[v].state, keep, out);
    }

    /// The admissible edge cost estimate `c_hat(v, x) = d(v, x)`.
    fn edge_estimate(&self, v: usize, x: usize) -> f64 {
        self.space
            .distance(&self.nodes[v].state, &self.nodes[x].state)
    }

    /// Queues the edge from vertex `v` to `x` with its current keys.
    fn queue_edge(&mut self, v: usize, x: usize) {
        let g = self.nodes[v].g;
        let through = g + self.edge_estimate(v, x);

        self.edge_queue.push(EdgeEntry {
            key: [through + self.nodes[x].h_hat, through, g],
            from: v,
            to: x,
        });
    }

    /// Processes the best edge (Algorithm 1, lines 15-34): adds or rewires its target when
    /// the checked edge improves both the target and the solution, or ends the batch when even
    /// the estimate cannot improve the solution. Returns whether the tree changed.
    fn process_edge(&mut self, edge: EdgeEntry) -> bool {
        let (v, x) = (edge.from, edge.to);

        if self.nodes[v].kind != Kind::Vertex || self.nodes[x].kind == Kind::Pruned {
            return false;
        }

        let g_v = self.nodes[v].g;
        let estimate = g_v + self.edge_estimate(v, x);

        if estimate + self.nodes[x].h_hat >= self.best_cost {
            self.vertex_queue.clear();
            self.edge_queue.clear();
            return false;
        }

        if estimate >= self.nodes[x].g || !self.is_motion_valid(v, x) {
            return false;
        }

        // With c = c_hat for a valid edge, the checked conditions equal the estimated ones.
        self.connect(v, x, estimate);

        true
    }

    /// Whether the motion from node `from` to node `to` checks as valid.
    fn is_motion_valid(&self, from: usize, to: usize) -> bool {
        self.checker
            .check_motion(&self.space, &self.nodes[from].state, &self.nodes[to].state)
            .is_valid()
    }

    /// Makes `v` the parent of `x` with cost `g`: adds the sample `x` to the tree, or rewires
    /// the vertex `x` and lowers the costs of its subtree. Updates the solution.
    fn connect(&mut self, v: usize, x: usize, g: f64) {
        if let Some(old) = self.nodes[x].parent {
            self.nodes[old].children.retain(|&child| child != x);
            self.stale_keys = true;
        }

        let was_sample = self.nodes[x].kind == Kind::Sample;
        let node = &mut self.nodes[x];
        node.parent = Some(v);
        node.g = g;
        self.nodes[v].children.push(x);

        if was_sample {
            let node = &mut self.nodes[x];
            node.kind = Kind::Vertex;
            node.unexpanded = true;
            self.queue_vertex(x);

            if self.nodes[x].in_goal() {
                self.solutions.push(x);
            }
        } else {
            self.propagate_costs(x);
        }

        self.update_best_cost();
    }

    /// Recomputes the costs of all descendants of `x` from their parents' costs.
    fn propagate_costs(&mut self, x: usize) {
        let mut stack = self.nodes[x].children.clone();

        while let Some(child) = stack.pop() {
            let parent = self.nodes[child].parent.expect("a child has a parent");
            self.nodes[child].g = self.nodes[parent].g + self.edge_estimate(parent, child);
            stack.extend_from_slice(&self.nodes[child].children);
        }
    }

    /// Sets `c_best` to the cost of the cheapest solution vertex.
    fn update_best_cost(&mut self) {
        self.best_cost = self
            .solutions
            .iter()
            .map(|&id| self.nodes[id].g)
            .fold(f64::INFINITY, f64::min);
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

    /// The number of tree vertices, including the start.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.count(Kind::Vertex)
    }

    /// The number of samples not connected to the tree.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.count(Kind::Sample)
    }

    /// The number of batches started so far.
    #[must_use]
    pub const fn batch_count(&self) -> usize {
        self.batches
    }

    /// The number of stored states of `kind`.
    fn count(&self, kind: Kind) -> usize {
        self.nodes.iter().filter(|node| node.kind == kind).count()
    }
}
