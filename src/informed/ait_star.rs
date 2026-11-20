//! AIT*: adaptively informed trees (Strub and Gammell 2020).

mod forward;
mod reverse;
#[cfg(test)]
mod tests;

use alloc::{collections::BinaryHeap, vec::Vec};

use rand_core::Rng;

use super::batch::{Informed, Neighborhood};
use super::queues::{EdgeEntry, VertexEntry};
use super::{BitStarConfig, Error};
#[cfg(doc)]
use crate::trees::INFORMED_ATTEMPTS;
use crate::{GoalRegion, LinearScan, NearestNeighbors, Path, Sampler, StateSpace, ValidityChecker};

/// Settings for [`AitStar`]: the same batch size, radius constant, k-nearest constant and
/// edge cap as BIT*, with the same meaning, ranges and lower bounds
/// ([`BitStarConfig::gamma_lower_bound`], [`BitStarConfig::k_constant_lower_bound`]).
pub type AitStarConfig = BitStarConfig;

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

/// One stored state: a forward-tree vertex or a sample, with its reverse-search values.
#[derive(Debug, Clone)]
struct Node<S> {
    /// The state.
    state: S,
    /// What the state currently is.
    kind: Kind,
    /// The cost-to-come through the forward tree, `g_F`; infinite for samples.
    g: f64,
    /// The admissible cost-to-come `g_hat = d(start, x)`, also the reverse search's heuristic.
    g_hat: f64,
    /// The geometric cost-to-go `max(0, d(x, goal center) - tolerance)`, used for pruning and
    /// to recognize goal states (zero inside the goal region).
    h_hat: f64,
    /// The forward-tree parent; `None` for the root and samples.
    parent: Option<usize>,
    /// The forward-tree children.
    children: Vec<usize>,
    /// The reverse search's cost-to-go when last connected, `h_con` (LPA*'s `rhs`); the
    /// forward search's heuristic. Infinite until the reverse search reaches the state.
    h_con: f64,
    /// The reverse search's cost-to-go when last expanded, `h_exp` (LPA*'s `g`).
    h_exp: f64,
    /// The key of this state's live entry in the reverse queue, or `None` if not queued.
    reverse_key: Option<[f64; 2]>,
    /// The forward-queue edges into this state: each source vertex with the key of its live
    /// entry. Edges leave this list when popped.
    incoming: Vec<(usize, [f64; 3])>,
    /// The neighbors in this batch's graph: the RGG neighbors, symmetrized, plus the forward
    /// tree edges at the start of the batch. Sorted ascending, no duplicates.
    adjacent: Vec<usize>,
    /// The targets of motions from this state that checked as not valid, `E_invalid`; kept
    /// across batches.
    invalid: Vec<usize>,
}

impl<S> Node<S> {
    /// The admissible solution cost through this state, `f_hat = g_hat + h_hat`.
    fn f_hat(&self) -> f64 {
        self.g_hat + self.h_hat
    }

    /// Whether the state lies in the goal region: a root of the reverse search.
    fn in_goal(&self) -> bool {
        self.h_hat <= 0.0
    }
}

/// AIT*: adaptively informed trees (Strub and Gammell 2020).
///
/// AIT* is [BIT*](super::BitStar) with a problem-specific heuristic. Each batch of samples is
/// an implicit random geometric graph (RGG). An asymmetric bidirectional search runs on it: a
/// lazy reverse search from the goal states, Lifelong Planning A* (LPA*) without collision
/// checks, computes the cost-to-go `h_con` of every state over the RGG; the forward search
/// grows a tree from the start like BIT*, ordered by `g_F(v) + c_hat(v, x) + h_con(x)`, and
/// checks the edges that come out on top. Every edge it finds invalid is removed from the
/// reverse search's graph, and LPA* repairs the heuristic incrementally, so that the forward
/// search stops being drawn into obstacles.
///
/// ```mermaid
/// graph LR
///     Batch["new batch: prune, sample, restart reverse search"] --> Reverse
///     Reverse["reverse search (LPA*) until the best edge is settled"] --> Edge["pop best edge"]
///     Edge -->|invalid| Repair["remove edge, update_state of its source"]
///     Repair --> Reverse
///     Edge -->|improves| Connect["add or rewire, expand"]
///     Connect --> Reverse
///     Edge -->|cannot improve| Batch
/// ```
///
/// This follows Algorithms 2 to 6 of the ICRA 2020 paper (arXiv 2002.06599), with these
/// choices:
///
/// - The heuristics are `g_hat(x) = d(start, x)` and `c_hat = d`. The goal center is the first
///   sample when it is valid; every state in the goal region is a goal of the reverse search,
///   and any forward-tree vertex in it is a solution.
/// - The reverse search runs on a fixed graph per batch: the RGG neighbors of every state
///   (radius, or k-nearest made symmetric) plus the forward-tree edges kept from earlier
///   batches, minus the invalid edges. `h_con[x]` uses the motion from `x` to its neighbor
///   (travel direction), and invalid edges are stored directed, so directional checkers stay
///   sound; the paper and OMPL blacklist both directions.
/// - Algorithm 6, line 1 reads "`x ≠ x_init`" in the paper; the states whose `h_con` must not
///   be recomputed are the reverse roots, so this checks "`x ∉ X_goal`", as OMPL's
///   `updateReverseSearchVertex` does.
/// - Algorithm 5's stopping rule (lines 10 to 12, including the printed "<" on line 11) is
///   replaced by OMPL's edge-based rule: the reverse search pauses when the child of the best
///   forward edge is consistent (`h_con == h_exp`) and that edge's first key is below the
///   reverse queue's first key. This is sound because `g_hat` is consistent: by the triangle
///   inequality an edge's first key is at least its child's first LPA* key, so the child is
///   settled and every edge with an unsettled child has a larger true key.
/// - The forward queue keys are `(g_F(v) + c_hat + h_con(x), g_F(v) + c_hat, g_F(v))` as in
///   the paper; OMPL uses `c_hat + h_con(x)` as the second key. Keys follow `h_con` as the
///   reverse search changes it, and are re-sorted after a rewiring lowered tree costs.
/// - A new batch prunes like BIT* (with the geometric `h_hat`, not `h_con`, which is not
///   admissible beyond the current graph), samples through [`Sampler::sample_informed`] with a
///   rejection fallback, and restarts the reverse search from the goals, as OMPL does. Batches
///   can come out smaller than `batch_size` when [`INFORMED_ATTEMPTS`] draws per sample do not
///   find a useful one.
/// - An invalid edge triggers the paper's LPA* repair (`update_state` of its source); OMPL
///   instead resets the whole reverse branch below it.
/// - Edges are checked in travel direction, from tree parent to child.
///
/// The cost of [`AitStar::best_path`] never increases. Pruned states stay in memory until the
/// planner is dropped.
///
/// # Examples
///
/// ```
/// use rand_core::SeedableRng;
/// use sampling_planning::{
///     Euclidean, GoalRegion, UniformBox, Validity, ValidityChecker,
///     informed::{AitStar, AitStarConfig},
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
/// let config = AitStarConfig::new(1.5 * AitStarConfig::gamma_lower_bound(2, 100.0));
/// let goal = GoalRegion::new([9.0, 9.0], 0.5);
/// let mut ait = AitStar::new(Euclidean::<2>, sampler, Free, [1.0, 1.0], goal, config)
///     .expect("valid");
/// ait.run(50, &mut rand_pcg::Pcg64Mcg::seed_from_u64(1));
///
/// assert!(ait.best_path().expect("free space").cost < 11.0);
/// ```
#[derive(Debug, Clone)]
pub struct AitStar<Sp: StateSpace, Sa, C, N = LinearScan<<Sp as StateSpace>::State>> {
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
    config: AitStarConfig,
    /// All stored states, by node id; node `0` is the start.
    nodes: Vec<Node<Sp::State>>,
    /// An empty index, cloned for every batch.
    prototype: N,
    /// Index over the vertices and samples of the current batch.
    nn: N,
    /// The forward queue `Q_F`; an entry is live while its target lists it in
    /// [`Node::incoming`] with the same key.
    forward_queue: BinaryHeap<EdgeEntry>,
    /// The reverse queue `Q_R`, keyed `(min(h_con, h_exp) + g_hat, min(h_con, h_exp))`; an
    /// entry is live while its state's [`Node::reverse_key`] equals its key.
    reverse_queue: BinaryHeap<VertexEntry>,
    /// Whether a rewiring lowered tree costs since the forward queue was keyed.
    stale_keys: bool,
    /// The forward-tree vertices inside the goal region.
    solutions: Vec<usize>,
    /// The cost of the best solution, `c_current`; infinite until one is found.
    best_cost: f64,
    /// The connection radius of the current batch, or the neighbor count for k-nearest.
    neighborhood: Neighborhood,
    /// The number of batches started.
    batches: usize,
}

/// The node id of the start state.
const START: usize = 0;

impl<Sp, Sa, C> AitStar<Sp, Sa, C>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
{
    /// Creates a planner with a [`LinearScan`] nearest-neighbor index.
    ///
    /// # Errors
    ///
    /// The same as [`AitStar::with_nearest_neighbors`].
    pub fn new(
        space: Sp,
        sampler: Sa,
        checker: C,
        start: Sp::State,
        goal: GoalRegion<Sp::State>,
        config: AitStarConfig,
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

impl<Sp, Sa, C, N> AitStar<Sp, Sa, C, N>
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
        config: AitStarConfig,
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
            forward_queue: BinaryHeap::new(),
            reverse_queue: BinaryHeap::new(),
            stale_keys: false,
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
            h_con: f64::INFINITY,
            h_exp: f64::INFINITY,
            reverse_key: None,
            incoming: Vec::new(),
            adjacent: Vec::new(),
            invalid: Vec::new(),
        });

        id
    }

    /// Runs one iteration of Algorithm 2: starts a new batch if the forward queue is empty,
    /// advances the reverse search as far as the best edge needs, then processes that edge.
    /// Returns whether the forward tree changed.
    ///
    /// The step that starts a batch builds the batch's graph over all stored states. While
    /// the reverse search cannot connect the start to a goal, every step is a new batch, so
    /// budget iterations by [`AitStar::batch_count`] as well.
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> bool {
        self.resort_if_stale();

        if self.peek_edge().is_none() {
            self.start_batch(rng);
        }

        self.advance_reverse_search();

        match self.pop_edge() {
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

    /// Prunes, samples a batch, rebuilds the graph, restarts the reverse search and expands
    /// the start (Algorithm 2, lines 2, 3 and 23 to 25).
    fn start_batch<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        self.batches += 1;
        self.clear_forward_queue();

        if self.best_cost.is_finite() {
            self.prune();
        }

        let sampled = self.sample_batch(rng);
        self.reindex(sampled);
        self.build_graph();
        self.restart_reverse_search();
        self.expand(START);
    }

    /// Removes the samples and vertices that cannot improve the solution, as BIT* does
    /// (Gammell et al. 2020, Algorithm 3); vertices that still could are kept as samples.
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

    /// Removes vertex `v` from the tree; it becomes a sample if it could still improve the
    /// solution and is pruned otherwise. Its children are handled by the caller's loop.
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
    /// from `q = |V| + |X_unconn| - m`, as BIT* does.
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

    /// Fixes this batch's graph: every state's RGG neighbors, symmetrized, plus the forward
    /// tree edges, which may be longer than this batch's radius.
    fn build_graph(&mut self) {
        let mut near = Vec::new();

        for node in &mut self.nodes {
            node.adjacent.clear();
        }

        for id in 0..self.nodes.len() {
            if self.nodes[id].kind == Kind::Pruned {
                continue;
            }

            self.neighborhood.query(
                (&self.space, &self.nn),
                (id, &self.nodes[id].state),
                (self.config.max_radius, |x| self.cost_estimate(id, x)),
                &mut near,
            );
            self.link_all(id, &near);

            if let Some(parent) = self.nodes[id].parent {
                self.link_all(id, &[parent]);
            }
        }

        for node in &mut self.nodes {
            node.adjacent.sort_unstable();
            node.adjacent.dedup();
        }
    }

    /// Adds the undirected graph edges between `id` and each of `others`.
    fn link_all(&mut self, id: usize, others: &[usize]) {
        for &other in others {
            self.nodes[id].adjacent.push(other);
            self.nodes[other].adjacent.push(id);
        }
    }

    /// The admissible edge cost estimate `c_hat(from, to) = d(from, to)`, in travel direction.
    fn cost_estimate(&self, from: usize, to: usize) -> f64 {
        self.space
            .distance(&self.nodes[from].state, &self.nodes[to].state)
    }

    /// Whether the motion from `from` to `to` is known to be invalid.
    fn is_invalid(&self, from: usize, to: usize) -> bool {
        self.nodes[from].invalid.contains(&to)
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

    /// The number of stored states of `kind`.
    fn count(&self, kind: Kind) -> usize {
        self.nodes.iter().filter(|node| node.kind == kind).count()
    }
}
