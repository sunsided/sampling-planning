//! SPARS: sparse roadmap spanners with a dense graph (Dobson and Bekris 2014).

mod dense;
mod quality;

use alloc::{
    collections::{BTreeMap, BTreeSet},
    vec::Vec,
};

use rand_core::Rng;

use super::{
    Error,
    sparse::{GuardKind, SparseGraph},
};
use crate::radius::k_constant_lower_bound;
use crate::{GoalRegion, LinearScan, NearestNeighbors, Path, Sampler, StateSpace, ValidityChecker};

/// Settings for [`Spars`]. The defaults follow OMPL's `SPARS`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SparsConfig {
    /// `Delta`: the visibility range of a guard. OMPL uses a quarter of the space's extent.
    /// Positive and finite.
    pub sparse_delta: f64,
    /// `delta`: two dense nodes closer than this, joined by a dense edge and represented by
    /// different guards, mark an interface between those guards. OMPL uses a thousandth of
    /// the extent. Positive, finite and below `sparse_delta`.
    ///
    /// The dense graph connects each node to its `k ln n` nearest neighbors, so interfaces
    /// appear only where dense edges get shorter than `dense_delta`; too small a value starves
    /// the quality criterion until the dense graph is very large.
    pub dense_delta: f64,
    /// `t`: the stretch factor the quality criterion enforces. Above one; 3 by default.
    pub stretch: f64,
    /// `M`: construction is complete after this many consecutive valid samples for which all
    /// four criteria failed. 1000 by default.
    pub max_failures: usize,
    /// Check every sparse and dense edge in both directions, for directional checkers (see
    /// [`PrmConfig::check_both_directions`](super::PrmConfig::check_both_directions)). Off by
    /// default.
    pub check_both_directions: bool,
}

impl SparsConfig {
    /// A config with the given visibility and interface ranges, stretch 3 and 1000 failures.
    #[must_use]
    pub const fn new(sparse_delta: f64, dense_delta: f64) -> Self {
        Self {
            sparse_delta,
            dense_delta,
            stretch: 3.0,
            max_failures: 1000,
            check_both_directions: false,
        }
    }

    /// Checks every field against its documented range.
    fn validate(&self) -> Result<(), Error> {
        let positive = |value: f64| value.is_finite() && value > 0.0;
        let valid = positive(self.sparse_delta)
            && positive(self.dense_delta)
            && self.dense_delta < self.sparse_delta
            && self.stretch.is_finite()
            && self.stretch > 1.0;

        if valid {
            Ok(())
        } else {
            Err(Error::InvalidRadius)
        }
    }
}

/// SPARS: an incremental sparse roadmap spanner that keeps a dense k-nearest PRM* graph next
/// to the sparse one (Dobson and Bekris 2014, "Sparse roadmap spanners for asymptotically
/// near-optimal motion planning"), ported from OMPL's `SPARS`.
///
/// Every valid sample `q` first joins the dense graph `D`: it connects to its
/// `ceil(e (1 + 1/d) ln n)` nearest dense nodes by valid motions, and gets a
/// *representative*, the closest guard (ties by id) it can see within `sparse_delta`. Dense
/// nodes joined by a dense edge shorter than `dense_delta` but represented by different guards
/// `v` and `v'` lie on the *interface* of `v` and `v'`; each guard keeps, per other guard, the
/// list of its dense nodes on that interface. Then `q` is tested against four criteria, in
/// order, using the guards it can see within `sparse_delta`:
///
/// 1. **Coverage:** it sees no guard, so it becomes a guard.
/// 2. **Connectivity:** it sees guards of different components, so it becomes a guard joining
///    them.
/// 3. **Interface:** its two closest guards are both visible but not adjacent, so they are
///    connected directly, or through `q`.
/// 4. **Quality:** `q` lies on the interface of its representative `v` with a guard `v'`. For
///    each neighbor `v''` of `v` not adjacent to `v'`, the estimate `s_max` of the sparse path
///    from `v'` over `v` toward `v''` is compared with the shortest dense path from `q` to a
///    dense node on the interface of `v` and `v''`; if `s_max` exceeds `stretch` times its
///    length, that dense path (extended by one dense node on each side) is added to the sparse
///    graph as quality guards.
///
/// Construction ends after `max_failures` consecutive valid samples fail all four criteria.
/// The roadmap is then, with high probability, a spanner of the dense graph: paths through it
/// cost at most about `stretch` times the optimum plus a term in `sparse_delta`, with far
/// fewer nodes than the dense graph. [`Spars::query`] searches only the sparse graph.
///
/// ```mermaid
/// graph LR
///     Sample --> Dense["add to D, update representatives"]
///     Dense --> Coverage
///     Coverage -->|fails| Connectivity
///     Connectivity -->|fails| Interface
///     Interface -->|fails| Quality["quality: interface of q, dense paths"]
///     Quality -->|fails| Failure["failures += 1"]
/// ```
///
/// The four criteria, their order, the representative and interface-list bookkeeping
/// (`addMilestone`, `calculateRepresentative`, `updateRepresentatives` over the dense nodes
/// within `sparse_delta + dense_delta` of each new guard), the loop nesting of `checkAddPath`
/// and the failure counting (incremented only when all four tests fail, reset by every new
/// guard, the direct interface edge and the zero-length path case) follow OMPL. Differences to
/// OMPL's `SPARS`, documented here so that results can be compared:
///
/// - OMPL connects the first state of an added quality path to `v''` and the last to `v'`
///   (it passes them swapped to `addPathToSpanner`); here the first, which `v'` represents,
///   connects to `v'` and the last to `v''`.
/// - When `q` itself lies on the interface of `v` and `v''`, OMPL pushes `q` in front of the
///   previous best dense path instead of replacing it; here the dense path is just `q`.
/// - OMPL throws when a dense node has no interface neighbor with the expected representative
///   (possible when the best dense path ends on another `v''` interface, or when interface
///   lists went stale); here that violation is skipped and the search goes on.
/// - Every sparse edge of a quality path is checked before anything is added, and the path is
///   skipped if one is invalid; OMPL adds them unchecked.
/// - OMPL's randomized `reduceVertices` is replaced by the deterministic greedy
///   [`shortcut`](crate::shortcut).
/// - Dense paths come from Dijkstra (ties by id) instead of A*; their lengths agree.
/// - A dense node without a visible guard has no representative (OMPL leaves an unset value);
///   it is kept out of interface lists until a guard covers it, which the coverage criterion
///   does in the same step.
/// - Start and goal are not added as guards: [`Spars::query`] connects them to visible guards,
///   so one roadmap answers many queries.
/// - Invalid samples are drawn and dropped without counting as failures, as OMPL's valid-state
///   sampler retries until it finds a valid state.
///
/// Like OMPL's, the quality test estimates the sparse path through `v` from guard distances
/// alone (`s_max`), so an added path never makes the violation it repairs disappear; only the
/// new guards taking over dense nodes do. Once many dense edges undercut `dense_delta`, every
/// new quality path creates further violations and construction may not finish in practice,
/// even at the default stretch 3 (on a 10 x 10 world with `sparse_delta` 2, from
/// `dense_delta` of about 0.085 on; at stretch 1.2 already from about 0.06). Keep
/// `dense_delta` small, as OMPL does, and bound [`Spars::construct`] by `max_samples`.
///
/// # Examples
///
/// ```
/// use rand_core::SeedableRng;
/// use sampling_planning::{
///     Euclidean, GoalRegion, UniformBox, Validity, ValidityChecker,
///     roadmaps::{Spars, SparsConfig},
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
/// let mut config = SparsConfig::new(2.5, 0.05);
/// config.max_failures = 200;
/// let mut spars = Spars::new(Euclidean::<2>, sampler, Free, config).expect("valid");
/// spars.construct(100_000, &mut rand_pcg::Pcg64Mcg::seed_from_u64(1));
///
/// assert!(spars.is_complete());
/// assert!(spars.node_count() < spars.dense_node_count());
/// let path = spars.query(&[1.0, 1.0], &GoalRegion::new([9.0, 9.0], 0.5)).expect("valid start");
/// assert!(path.expect("free space").cost < 3.0 * 11.4);
/// ```
#[derive(Debug, Clone)]
pub struct Spars<Sp: StateSpace, Sa, C, N = LinearScan<<Sp as StateSpace>::State>> {
    /// The configuration space.
    space: Sp,
    /// Draws candidate states.
    sampler: Sa,
    /// Decides which states and motions are allowed.
    checker: C,
    /// The validated settings.
    config: SparsConfig,
    /// The k-nearest PRM* constant `e (1 + 1/d)` of the dense graph.
    k_constant: f64,
    /// The sparse roadmap.
    graph: SparseGraph<Sp::State, N>,
    /// The dense node states, by dense id. All checked as valid.
    dense_states: Vec<Sp::State>,
    /// The neighbors of each dense node, in the order the edges were added.
    dense_adjacency: Vec<Vec<usize>>,
    /// Index over the dense nodes.
    dense_nn: N,
    /// The representative guard of each dense node; `None` until a guard is visible from it.
    representatives: Vec<Option<usize>>,
    /// Per guard `v`, per other guard `v'`: the dense nodes represented by `v` that have a
    /// dense neighbor closer than `dense_delta` represented by `v'`.
    interface_lists: Vec<BTreeMap<usize, BTreeSet<usize>>>,
    /// Per guard, the dense nodes it represents that lie on no interface.
    non_interface_lists: Vec<BTreeSet<usize>>,
    /// Consecutive valid samples for which all four criteria failed.
    failures: usize,
}

impl<Sp, Sa, C> Spars<Sp, Sa, C>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
{
    /// Creates an empty roadmap with [`LinearScan`] indices over its guards and dense nodes.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRadius`] if a range is not positive and finite, `dense_delta` is not
    /// below `sparse_delta`, or the stretch is not above one.
    pub fn new(space: Sp, sampler: Sa, checker: C, config: SparsConfig) -> Result<Self, Error> {
        Self::with_nearest_neighbors(space, sampler, checker, config, LinearScan::default())
    }
}

impl<Sp, Sa, C, N> Spars<Sp, Sa, C, N>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    N: NearestNeighbors<Sp> + Clone,
{
    /// Creates an empty roadmap with the given empty index; a clone of it indexes the dense
    /// nodes, the original the guards.
    ///
    /// # Errors
    ///
    /// The same as [`Spars::new`].
    pub fn with_nearest_neighbors(
        space: Sp,
        sampler: Sa,
        checker: C,
        config: SparsConfig,
        nn: N,
    ) -> Result<Self, Error> {
        config.validate()?;
        let k_constant = k_constant_lower_bound(space.dimension());

        Ok(Self {
            space,
            sampler,
            checker,
            config,
            k_constant,
            graph: SparseGraph::new(nn.clone()),
            dense_states: Vec::new(),
            dense_adjacency: Vec::new(),
            dense_nn: nn,
            representatives: Vec::new(),
            interface_lists: Vec::new(),
            non_interface_lists: Vec::new(),
            failures: 0,
        })
    }
}

impl<Sp, Sa, C, N> Spars<Sp, Sa, C, N>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    N: NearestNeighbors<Sp>,
{
    /// Processes samples until construction is complete or `max_samples` were drawn, valid or
    /// not. Returns the number of samples drawn.
    pub fn construct<R: Rng + ?Sized>(&mut self, max_samples: usize, rng: &mut R) -> usize {
        let mut drawn = 0;

        while drawn < max_samples && !self.is_complete() {
            self.step(rng);
            drawn += 1;
        }

        drawn
    }

    /// Draws one sample; if it is valid, adds it to the dense graph and applies the four
    /// criteria. Returns whether the sparse roadmap changed.
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> bool {
        let state = self.sampler.sample(rng);

        if !self.checker.check_state(&state).is_valid() {
            return false;
        }

        let before = (self.graph.len(), self.graph.edge_count());
        let q = self.add_milestone(state);

        if !self.process(q) {
            self.failures += 1;
        }

        before != (self.graph.len(), self.graph.edge_count())
    }
}

impl<Sp, Sa, C, N> Spars<Sp, Sa, C, N>
where
    Sp: StateSpace,
    C: ValidityChecker<Sp>,
    N: NearestNeighbors<Sp>,
{
    /// Applies the four criteria, in order, to the new dense node `q` (OMPL's
    /// `constructRoadmap` body). Returns whether one of them succeeded.
    fn process(&mut self, q: usize) -> bool {
        let state = self.dense_states[q].clone();
        let (near, visible) = self.neighborhood(&state);

        if visible.is_empty() {
            self.add_guard(state, GuardKind::Coverage);
            return true;
        }

        if self.check_connectivity(&state, &visible) || self.check_interface(q, &near, &visible) {
            return true;
        }

        let interface = self.interface_neighborhood(q);

        !interface.is_empty() && self.check_add_path(q, &interface)
    }

    /// The guards within `sparse_delta` of `state`, closest first (ties by id), and the ones
    /// visible from `state`.
    fn neighborhood(&self, state: &Sp::State) -> (Vec<usize>, Vec<usize>) {
        self.graph
            .neighborhood(&self.space, state, self.config.sparse_delta, |guard| {
                self.is_edge_valid(state, guard)
            })
    }

    /// Whether the motion between `a` and `b` checks as valid, both ways if configured.
    fn is_edge_valid(&self, a: &Sp::State, b: &Sp::State) -> bool {
        let forward = self.checker.check_motion(&self.space, a, b).is_valid();

        forward
            && (!self.config.check_both_directions
                || self.checker.check_motion(&self.space, b, a).is_valid())
    }

    /// Whether the motion between guards `a` and `b` checks as valid.
    fn is_guard_edge_valid(&self, a: usize, b: usize) -> bool {
        self.is_edge_valid(&self.graph.states[a], &self.graph.states[b])
    }

    /// Adds `state` as a guard, re-assigns the representatives of the dense nodes it may now
    /// represent, and resets the failure counter (OMPL's `addGuard`).
    fn add_guard(&mut self, state: Sp::State, kind: GuardKind) -> usize {
        let guard = self.graph.add_guard(state, kind);
        self.interface_lists.push(BTreeMap::new());
        self.non_interface_lists.push(BTreeSet::new());
        self.update_representatives(guard);
        self.failures = 0;

        guard
    }

    /// Connectivity: if `state` sees guards of different components, adds it as a guard
    /// joined to one guard of each. Returns whether it did.
    fn check_connectivity(&mut self, state: &Sp::State, visible: &[usize]) -> bool {
        let links = self.graph.connectivity_links(visible);

        if links.is_empty() {
            return false;
        }

        let guard = self.add_guard(state.clone(), GuardKind::Connectivity);
        self.graph.connect_links(guard, &links);

        true
    }

    /// Interface: if the two closest guards of dense node `q` are both visible but not
    /// adjacent, connects them directly (resetting the failure counter) or through `q`.
    /// Returns whether it did.
    fn check_interface(&mut self, q: usize, near: &[usize], visible: &[usize]) -> bool {
        let Some((v1, v2)) = self.graph.interface_pair(near, visible) else {
            return false;
        };

        if self.is_guard_edge_valid(v1, v2) {
            self.graph.connect(v1, v2);
            self.failures = 0;
        } else {
            let guard = self.add_guard(self.dense_states[q].clone(), GuardKind::Interface);
            self.graph.connect(guard, v1);
            self.graph.connect(guard, v2);
        }

        true
    }

    /// The cheapest path from `start` to `goal` through the sparse roadmap, or `None` if the
    /// roadmap does not connect them. Both endpoints connect to every visible guard within
    /// `sparse_delta`; the dense graph is not searched.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidStart`] if `start` does not check as valid.
    pub fn query(
        &self,
        start: &Sp::State,
        goal: &GoalRegion<Sp::State>,
    ) -> Result<Option<Path<Sp::State>>, Error> {
        let validity = self.checker.check_state(start);
        if !validity.is_valid() {
            return Err(Error::InvalidStart(validity));
        }

        Ok(self.graph.query(
            &self.space,
            &self.checker,
            start,
            goal,
            self.config.sparse_delta,
        ))
    }

    /// Whether `max_failures` consecutive valid samples failed all four criteria.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.failures >= self.config.max_failures
    }

    /// The number of guards (sparse nodes).
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.graph.len()
    }

    /// The number of dense nodes: every valid sample drawn so far.
    #[must_use]
    pub fn dense_node_count(&self) -> usize {
        self.dense_states.len()
    }

    /// The number of sparse roadmap edges.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// The number of guards added for each criterion: coverage, connectivity, interface,
    /// quality.
    #[must_use]
    pub fn guard_counts(&self) -> [usize; 4] {
        let count = |kind| self.graph.kinds.iter().filter(|&&k| k == kind).count();

        [
            count(GuardKind::Coverage),
            count(GuardKind::Connectivity),
            count(GuardKind::Interface),
            count(GuardKind::Quality),
        ]
    }
}

#[cfg(test)]
mod tests {
    use rand_core::SeedableRng;

    use super::*;
    use crate::{Euclidean, KdTree, UniformBox, Validity, discretized_motion};

    /// A wall at `x` in `[4, 6]` for `y < 7` in a 10 x 10 box.
    struct Wall;

    impl ValidityChecker<Euclidean<2>> for Wall {
        fn check_state(&self, s: &[f64; 2]) -> Validity {
            if (4.0..=6.0).contains(&s[0]) && s[1] < 7.0 {
                Validity::Invalid
            } else {
                Validity::Valid
            }
        }

        fn check_motion(&self, space: &Euclidean<2>, from: &[f64; 2], to: &[f64; 2]) -> Validity {
            discretized_motion(space, from, to, 0.02, |s| self.check_state(s))
        }
    }

    /// A SPARS roadmap over the wall world with the given interface range, after at most
    /// `max_samples` samples with a fixed seed.
    fn roadmap(
        dense_delta: f64,
        max_samples: usize,
    ) -> Spars<Euclidean<2>, UniformBox<2>, Wall, KdTree<2>> {
        let config = SparsConfig::new(2.0, dense_delta);
        let sampler = UniformBox::new([0.0, 0.0], [10.0, 10.0]);
        let mut spars =
            Spars::with_nearest_neighbors(Euclidean::<2>, sampler, Wall, config, KdTree::default())
                .expect("valid");
        spars.construct(max_samples, &mut rand_pcg::Pcg64Mcg::seed_from_u64(3));

        spars
    }

    /// The two fixtures: one constructed to completion, and one cut short whose wide interface
    /// range made the quality criterion add paths.
    fn fixtures() -> [Spars<Euclidean<2>, UniformBox<2>, Wall, KdTree<2>>; 2] {
        let complete = roadmap(0.05, 20_000);
        let with_quality = roadmap(0.25, 1500);
        assert!(complete.is_complete());
        assert!(
            with_quality.guard_counts()[3] > 0,
            "{:?}",
            with_quality.guard_counts()
        );

        [complete, with_quality]
    }

    /// Asserts that every dense node of `spars` is represented by a guard it sees within
    /// `sparse_delta`.
    fn assert_represented(spars: &Spars<Euclidean<2>, UniformBox<2>, Wall, KdTree<2>>) {
        for (q, state) in spars.dense_states.iter().enumerate() {
            let rep = spars.representatives[q].expect("every dense node is covered");
            let guard = &spars.graph.states[rep];

            assert!(spars.space.distance(state, guard) <= spars.config.sparse_delta);
            assert!(
                spars.is_edge_valid(state, guard),
                "node {q} cannot see {rep}"
            );
        }
    }

    /// Asserts that every listed dense node of `spars` is listed under its representative,
    /// and that both kinds of list are in use.
    fn assert_listed_under_representative(
        spars: &Spars<Euclidean<2>, UniformBox<2>, Wall, KdTree<2>>,
    ) {
        for (guard, lists) in spars.interface_lists.iter().enumerate() {
            let listed = lists.values().flatten();

            for &q in listed.chain(&spars.non_interface_lists[guard]) {
                assert_eq!(spars.representatives[q], Some(guard), "node {q}");
            }
        }

        let listed: usize = spars.non_interface_lists.iter().map(BTreeSet::len).sum();
        let on_interfaces = spars
            .interface_lists
            .iter()
            .any(|lists| lists.values().any(|list| !list.is_empty()));
        assert!(listed > 0 && on_interfaces && spars.guard_counts()[0] > 0);
    }

    #[test]
    fn every_dense_node_is_represented_by_a_visible_guard_within_range() {
        for spars in fixtures() {
            assert_represented(&spars);
        }
    }

    #[test]
    fn dense_nodes_are_listed_only_under_their_representative() {
        for spars in fixtures() {
            assert_listed_under_representative(&spars);
        }
    }
}
