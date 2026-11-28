//! ABIT*: advanced BIT* with an inflated, truncated search (Strub and Gammell 2020).

use rand_core::Rng;

use super::{BitStar, BitStarConfig, Error};
use crate::radius::count_to_f64;
use crate::{GoalRegion, LinearScan, NearestNeighbors, Path, Sampler, StateSpace, ValidityChecker};

/// Settings for [`AbitStar`]: BIT*'s settings plus the inflation and truncation factors.
///
/// With `q = |V| + |X_unconn|` states after a batch is drawn, the truncation factor of the
/// batch is `1 + truncation_scaling / q` and the inflation factor of its final search is
/// `1 + inflation_scaling / q`. Both tend to one as batches accumulate, which makes ABIT*
/// asymptotically optimal like BIT*.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AbitStarConfig {
    /// The batch size, neighbor rule and edge cap, as for [`BitStar`].
    pub bit: BitStarConfig,
    /// The inflation factor of the first search on every batch, `epsilon_infl` before it
    /// shrinks. Large values make that search greedy: it heads for the goal like a best-first
    /// search. At least one and finite.
    pub initial_inflation: f64,
    /// The scaling `lambda` of the final search's inflation factor `1 + lambda / q`.
    /// Non-negative and finite.
    pub inflation_scaling: f64,
    /// The scaling `tau` of the truncation factor `1 + tau / q`. Non-negative and finite.
    pub truncation_scaling: f64,
}

impl AbitStarConfig {
    /// A radius-based config with OMPL's ABIT* defaults: 100 samples per batch, no edge cap,
    /// initial inflation `10^6`, inflation scaling 10 and truncation scaling 5.
    #[must_use]
    pub const fn new(gamma: f64) -> Self {
        Self::from_bit_star(BitStarConfig::new(gamma))
    }

    /// OMPL's ABIT* factors on top of the given BIT* settings.
    #[must_use]
    pub const fn from_bit_star(bit: BitStarConfig) -> Self {
        Self {
            bit,
            initial_inflation: 1e6,
            inflation_scaling: 10.0,
            truncation_scaling: 5.0,
        }
    }

    /// Panics unless the factors are in their documented ranges.
    fn assert_factors(&self) {
        assert!(
            self.initial_inflation.is_finite() && self.initial_inflation >= 1.0,
            "initial inflation must be finite and at least one"
        );
        assert!(
            self.inflation_scaling.is_finite() && self.inflation_scaling >= 0.0,
            "inflation scaling must be finite and non-negative"
        );
        assert!(
            self.truncation_scaling.is_finite() && self.truncation_scaling >= 0.0,
            "truncation scaling must be finite and non-negative"
        );
    }
}

/// The search state ABIT* adds to the BIT* core: the current factors and which of the two
/// searches on a batch is running.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct Search {
    /// The inflation factor of the first search on a batch.
    initial_inflation: f64,
    /// The scaling of the final search's inflation factor.
    inflation_scaling: f64,
    /// The scaling of the truncation factor.
    truncation_scaling: f64,
    /// The inflation factor of the running search, at least one.
    inflation: f64,
    /// The truncation factor of the current batch, at least one.
    truncation: f64,
    /// Whether the running search is the final one on its batch.
    is_final_pass: bool,
    /// Whether the inflated search on this batch ended at the truncation test, so the final
    /// search starts at the next step.
    truncated: bool,
}

impl Search {
    /// The state before the first batch.
    pub(super) const fn new(config: &AbitStarConfig) -> Self {
        Self {
            initial_inflation: config.initial_inflation,
            inflation_scaling: config.inflation_scaling,
            truncation_scaling: config.truncation_scaling,
            inflation: config.initial_inflation,
            truncation: 1.0,
            is_final_pass: false,
            truncated: false,
        }
    }

    /// The inflation factor of the running search.
    pub(super) const fn inflation(&self) -> f64 {
        self.inflation
    }

    /// The truncation factor of the current batch.
    pub(super) const fn truncation(&self) -> f64 {
        self.truncation
    }

    /// Whether the inflated search ended at the truncation test.
    pub(super) const fn is_truncated(&self) -> bool {
        self.truncated
    }

    /// Records that the inflated search ended at the truncation test.
    pub(super) const fn truncate(&mut self) {
        self.truncated = true;
    }

    /// Whether the next search runs on the current batch again: only after its inflated
    /// search, and only once a solution exists (before that, every batch gets one search).
    pub(super) const fn wants_final_search(&self, has_solution: bool) -> bool {
        has_solution && !self.is_final_pass
    }

    /// Starts the inflated first search on a new batch.
    pub(super) const fn begin_batch(&mut self) {
        self.inflation = self.initial_inflation;
        self.is_final_pass = false;
    }

    /// Sets the batch's truncation factor from its `states` vertices and samples.
    pub(super) fn truncate_for(&mut self, states: usize) {
        self.truncation = 1.0 + self.truncation_scaling / count_to_f64(states.max(1));
    }

    /// Starts the final search on the current batch of `states` vertices and samples.
    pub(super) fn begin_final_search(&mut self, states: usize) {
        self.inflation = 1.0 + self.inflation_scaling / count_to_f64(states.max(1));
        self.is_final_pass = true;
        self.truncated = false;
    }
}

/// ABIT*: advanced BIT* (Strub and Gammell 2020, "Advanced BIT* (ABIT*): Sampling-based
/// planning with advanced graph-search techniques", ICRA 2020, arXiv 2002.06589).
///
/// ABIT* runs BIT*'s batches with two graph-search techniques from the search literature:
///
/// - **Inflation:** the queues order vertices by `g + epsilon h_hat` and edges by
///   `g + c_hat + epsilon h_hat`. The first search on a batch uses a large `epsilon`
///   ([`AbitStarConfig::initial_inflation`]), so it greedily finds some path through the batch
///   quickly. Once a solution exists, a second, final search on the same batch continues from
///   the first one's tree and queues, re-keyed with `epsilon = 1 + lambda / q`.
/// - **Truncation:** a search ends at the first edge with `tau (g + c_hat + h_hat) >= c_best`,
///   `tau = 1 + truncation_scaling / q`, instead of when no edge can improve the solution at
///   all; the remaining improvement is left to denser batches.
///
/// Before the first solution every batch gets only the inflated search. Both factors tend to
/// one as `q` grows. This follows OMPL's `ABITstar` (Strub), which configures its `BITstar`:
///
/// - **Cascading rewirings** are on: every expansion of a vertex may rewire its neighbors, not
///   only its first (BIT* rewires only from never-expanded vertices).
/// - Edges found invalid are remembered and never checked twice.
/// - Like OMPL, a search ends when the popped edge fails the truncation test, although with
///   inflation the queue is not ordered by the uninflated estimate.
///
/// Two deviations: OMPL's final search also queues the edges of vertices whose cost changed
/// after their expansion; here such vertices are expanded again only in the next batch, as in
/// BIT*. And OMPL drops the edge that ended the inflated search; here it goes back into the
/// queue for the final search.
///
/// The cost of [`AbitStar::best_path`] never increases.
///
/// # Examples
///
/// ```
/// use rand_core::SeedableRng;
/// use sampling_planning::{
///     Euclidean, GoalRegion, UniformBox, Validity, ValidityChecker,
///     informed::{AbitStar, AbitStarConfig, BitStarConfig},
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
/// let config = AbitStarConfig::new(1.5 * BitStarConfig::gamma_lower_bound(2, 100.0));
/// let goal = GoalRegion::new([9.0, 9.0], 0.5);
/// let mut abit = AbitStar::new(Euclidean::<2>, sampler, Free, [1.0, 1.0], goal, config)
///     .expect("valid");
/// abit.run(2000, &mut rand_pcg::Pcg64Mcg::seed_from_u64(1));
///
/// assert!(abit.best_path().expect("free space").cost < 11.0);
/// ```
#[derive(Clone)]
pub struct AbitStar<Sp: StateSpace, Sa, C, N = LinearScan<<Sp as StateSpace>::State>> {
    /// The BIT* core with ABIT*'s search state switched on.
    core: BitStar<Sp, Sa, C, N>,
}

impl<Sp, Sa, C> AbitStar<Sp, Sa, C>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
{
    /// Creates a planner with a [`LinearScan`] nearest-neighbor index.
    ///
    /// # Errors
    ///
    /// The same as [`AbitStar::with_nearest_neighbors`].
    ///
    /// # Panics
    ///
    /// The same as [`AbitStar::with_nearest_neighbors`].
    pub fn new(
        space: Sp,
        sampler: Sa,
        checker: C,
        start: Sp::State,
        goal: GoalRegion<Sp::State>,
        config: AbitStarConfig,
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

impl<Sp, Sa, C, N> AbitStar<Sp, Sa, C, N>
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
    /// The same as [`BitStar::with_nearest_neighbors`] for [`AbitStarConfig::bit`].
    ///
    /// # Panics
    ///
    /// Panics if `initial_inflation` is below one or not finite, or a scaling is negative or
    /// not finite.
    pub fn with_nearest_neighbors(
        space: Sp,
        sampler: Sa,
        checker: C,
        start: Sp::State,
        goal: GoalRegion<Sp::State>,
        config: AbitStarConfig,
        nn: N,
    ) -> Result<Self, Error> {
        config.assert_factors();

        let core =
            BitStar::with_nearest_neighbors(space, sampler, checker, start, goal, config.bit, nn)?;

        Ok(Self {
            core: core.advanced(&config),
        })
    }

    /// Runs one iteration: starts the next search if the queues are empty (the final search on
    /// the batch, or a new batch), expands vertices while the vertex queue is ahead, then
    /// processes the best edge. Returns whether the tree changed.
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> bool {
        self.core.step(rng)
    }

    /// Runs `iterations` iterations.
    pub fn run<R: Rng + ?Sized>(&mut self, iterations: usize, rng: &mut R) {
        self.core.run(iterations, rng);
    }

    /// The cheapest path found so far, or `None` if no vertex reached the goal yet. Its cost
    /// never increases as iterations run.
    #[must_use]
    pub fn best_path(&self) -> Option<Path<Sp::State>> {
        self.core.best_path()
    }

    /// The number of tree vertices, including the start.
    #[must_use]
    pub fn vertex_count(&self) -> usize {
        self.core.vertex_count()
    }

    /// The number of samples not connected to the tree.
    #[must_use]
    pub fn sample_count(&self) -> usize {
        self.core.sample_count()
    }

    /// The number of batches started so far.
    #[must_use]
    pub const fn batch_count(&self) -> usize {
        self.core.batch_count()
    }

    /// The inflation factor of the running search.
    #[must_use]
    pub fn inflation_factor(&self) -> f64 {
        self.core.inflation()
    }

    /// The truncation factor of the current batch.
    #[must_use]
    pub fn truncation_factor(&self) -> f64 {
        self.core.truncation()
    }
}
