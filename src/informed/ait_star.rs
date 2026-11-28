//! AIT*: adaptively informed trees (Strub and Gammell 2020).

mod forward;
mod reverse;
#[cfg(test)]
mod tests;

use alloc::{collections::BinaryHeap, vec::Vec};
use core::fmt;

use rand_core::Rng;

use super::batch_graph::{BatchGraph, Kind, START};
use super::queues::{EdgeEntry, VertexEntry};
use super::{BitStarConfig, Error};
#[cfg(doc)]
use crate::trees::INFORMED_ATTEMPTS;
use crate::{GoalRegion, LinearScan, NearestNeighbors, Path, Sampler, StateSpace, ValidityChecker};

/// Settings for [`AitStar`]: the same batch size, radius constant, k-nearest constant and
/// edge cap as BIT*, with the same ranges and lower bounds
/// ([`BitStarConfig::gamma_lower_bound`], [`BitStarConfig::k_constant_lower_bound`]).
///
/// AIT* computes the batch radius as OMPL does: `q` counts every stored state, the newest
/// batch included, and `gamma` shrinks with the informed set's share of the sampled domain
/// ([`Sampler::informed_fraction`]), so the batch graph stays sparse once a solution shrinks
/// the informed set.
pub type AitStarConfig = BitStarConfig;

/// AIT*'s reverse-search values of one stored state.
#[derive(Debug, Clone)]
struct AitExt {
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
    /// The targets of motions from this state that checked as not valid, `E_invalid`; kept
    /// across batches.
    invalid: Vec<usize>,
}

impl Default for AitExt {
    fn default() -> Self {
        Self {
            h_con: f64::INFINITY,
            h_exp: f64::INFINITY,
            reverse_key: None,
            incoming: Vec::new(),
            invalid: Vec::new(),
        }
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
/// ait.run(200, &mut rand_pcg::Pcg64Mcg::seed_from_u64(1));
///
/// assert!(ait.best_path().expect("free space").cost < 11.0);
/// ```
#[derive(Clone)]
pub struct AitStar<Sp: StateSpace, Sa, C, N = LinearScan<<Sp as StateSpace>::State>> {
    /// The problem, the stored states with the forward tree, and the batch graph.
    graph: BatchGraph<Sp, Sa, C, N, AitExt>,
    /// The forward queue `Q_F`; an entry is live while its target lists it in
    /// [`AitExt::incoming`] with the same key.
    forward_queue: BinaryHeap<EdgeEntry>,
    /// The reverse queue `Q_R`, keyed `(min(h_con, h_exp) + g_hat, min(h_con, h_exp))`; an
    /// entry is live while its state's [`AitExt::reverse_key`] equals its key.
    reverse_queue: BinaryHeap<VertexEntry>,
    /// Whether a rewiring lowered tree costs since the forward queue was keyed.
    stale_keys: bool,
}

impl<Sp, Sa, C, N> fmt::Debug for AitStar<Sp, Sa, C, N>
where
    Sp: StateSpace + fmt::Debug,
    Sp::State: fmt::Debug,
    Sa: fmt::Debug,
    C: fmt::Debug,
    N: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AitStar")
            .field("graph", &self.graph)
            .field("forward_queue", &self.forward_queue)
            .field("reverse_queue", &self.reverse_queue)
            .field("stale_keys", &self.stale_keys)
            .finish()
    }
}

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
        Ok(Self {
            graph: BatchGraph::new(space, sampler, checker, start, goal, config, nn)?,
            forward_queue: BinaryHeap::new(),
            reverse_queue: BinaryHeap::new(),
            stale_keys: false,
        })
    }

    /// Runs one iteration of Algorithm 2: starts a new batch if the forward queue is empty,
    /// advances the reverse search as far as the best edge needs, then processes that edge.
    /// Returns whether the forward tree changed.
    ///
    /// A batch ends at the first popped edge whose estimate cannot beat the current solution.
    /// When the batch's graph holds no better path than the incumbent (or none at all), that
    /// is the first edge, so every step starts a new batch: it adds up to `batch_size` states
    /// and restarts the reverse search over all stored states, and the neighbor queries of
    /// the states it expands are linear with the [`LinearScan`] default of [`AitStar::new`].
    /// Use a [`KdTree`](crate::KdTree) or [`VpTree`](crate::VpTree) and budget by
    /// [`AitStar::batch_count`] or collision checks rather than steps. In free space a
    /// near-optimal run can reach 100 batches in a few hundred steps.
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
        self.clear_forward_queue();
        self.graph.next_batch(rng);
        self.restart_reverse_search();
        self.expand(START);
    }

    /// Whether the motion from `from` to `to` is known to be invalid.
    fn is_invalid(&self, from: usize, to: usize) -> bool {
        self.graph.nodes[from].ext.invalid.contains(&to)
    }

    /// The cheapest path found so far, or `None` if no vertex reached the goal yet. Its cost
    /// never increases as iterations run.
    #[must_use]
    pub fn best_path(&self) -> Option<Path<Sp::State>> {
        self.graph.best_path()
    }

    /// The number of forward-tree vertices, including the start.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.graph.count(Kind::Vertex)
    }

    /// The number of samples not connected to the forward tree.
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
