//! EIT*: effort informed trees (Strub and Gammell 2022).

mod checks;
mod forward;
mod queue;
mod reverse;

use alloc::{
    collections::{BTreeMap, BTreeSet, BinaryHeap},
    vec::Vec,
};

use core::fmt;

use rand_core::Rng;

use super::batch_graph::{BatchGraph, Extension, Kind, START};
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

/// EIT*'s reverse-search values of one stored state.
#[derive(Debug, Clone)]
struct EitExt {
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

impl Extension for EitExt {}

impl Default for EitExt {
    fn default() -> Self {
        Self {
            h: f64::INFINITY,
            effort: usize::MAX,
            reverse_parent: None,
            closed_tag: 0,
            incoming: Vec::new(),
        }
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
#[derive(Clone)]
pub struct EitStar<Sp: StateSpace, Sa, C, N = LinearScan<<Sp as StateSpace>::State>> {
    /// The problem, the stored states with the forward tree, and the batch graph; holds
    /// `config.bit`.
    graph: BatchGraph<Sp, Sa, C, N, EitExt>,
    /// The validated settings.
    config: EitStarConfig,
    /// The forward queue: directed edges `(vertex, target)` with their keys.
    forward_queue: ForwardQueue,
    /// The number of queued forward edges whose target the reverse search has not closed.
    open_targets: usize,
    /// The reverse queue; `from` is the state closer to the goal.
    reverse_queue: BinaryHeap<EdgeEntry>,
    /// Identifies the running reverse search; see [`EitExt::closed_tag`].
    reverse_tag: usize,
    /// The number of states the reverse search checks per edge now.
    sparse_checks: usize,
    /// Results of edge checks, kept across batches.
    checks: EdgeChecks,
}

impl<Sp, Sa, C, N> fmt::Debug for EitStar<Sp, Sa, C, N>
where
    Sp: StateSpace + fmt::Debug,
    Sp::State: fmt::Debug,
    Sa: fmt::Debug,
    C: fmt::Debug,
    N: fmt::Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EitStar")
            .field("graph", &self.graph)
            .field("config", &self.config)
            .field("forward_queue", &self.forward_queue)
            .field("open_targets", &self.open_targets)
            .field("reverse_queue", &self.reverse_queue)
            .field("reverse_tag", &self.reverse_tag)
            .field("sparse_checks", &self.sparse_checks)
            .field("checks", &self.checks)
            .finish()
    }
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
        // The batch settings are checked first, as `BatchGraph::new` would, so that a bad
        // radius is an error even when the effort settings would panic.
        config.bit.validate()?;
        config.assert_effort();

        Ok(Self {
            graph: BatchGraph::new(space, sampler, checker, start, goal, config.bit, nn)?,
            config,
            forward_queue: ForwardQueue::default(),
            open_targets: 0,
            reverse_queue: BinaryHeap::new(),
            reverse_tag: 0,
            sparse_checks: config.initial_sparse_checks,
            checks: EdgeChecks::default(),
        })
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
        self.graph.next_batch(rng);
        self.sparse_checks = self.config.initial_sparse_checks;
        self.clear_forward_queue();
        self.restart_reverse_search();
        self.expand_forward(START, None);
    }

    /// Whether node `id` is a goal state of the reverse search.
    fn is_goal(&self, id: usize) -> bool {
        self.graph.nodes[id].kind != Kind::Pruned && self.graph.nodes[id].in_goal()
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

    /// The number of states the reverse search currently checks per edge.
    #[must_use]
    pub const fn sparse_check_count(&self) -> usize {
        self.sparse_checks
    }
}
