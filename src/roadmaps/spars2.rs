//! SPARS2: sparse roadmap spanners without a dense graph (Dobson and Bekris 2013).

use alloc::{collections::BTreeMap, vec::Vec};

use rand_core::Rng;

use super::{
    Error,
    sparse::{GuardKind, SparseGraph},
};
use crate::{
    GoalRegion, LinearScan, NearestNeighbors, Path, Sampler, StateSpace, ValidityChecker, unit_f64,
};

/// Settings for [`Spars2`]. The defaults follow OMPL's `SPARStwo`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Spars2Config {
    /// `Delta`: the visibility range of a guard. OMPL uses a quarter of the space's extent.
    /// Positive and finite.
    pub sparse_delta: f64,
    /// `delta`: the range in which interface points are sampled around a sample. OMPL uses a
    /// thousandth of the extent. Positive, finite and below `sparse_delta`.
    pub dense_delta: f64,
    /// `t`: the stretch factor the quality criterion enforces. Above one; 3 by default.
    pub stretch: f64,
    /// `M`: construction is complete after this many consecutive samples that changed nothing.
    /// 5000 by default.
    pub max_failures: usize,
    /// `k`: the samples drawn near each sample to find neighboring representatives; OMPL uses
    /// twice the state dimension.
    pub near_samples: usize,
    /// Check every sparse edge in both directions, for directional checkers (see
    /// [`PrmConfig::check_both_directions`](super::PrmConfig::check_both_directions)). Off by
    /// default.
    pub check_both_directions: bool,
}

impl Spars2Config {
    /// A config with the given visibility and interface ranges, stretch 3, 5000 failures and
    /// `near_samples` four (twice a planar dimension).
    #[must_use]
    pub const fn new(sparse_delta: f64, dense_delta: f64) -> Self {
        Self {
            sparse_delta,
            dense_delta,
            stretch: 3.0,
            max_failures: 5000,
            near_samples: 4,
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

/// What a guard knows about one pair of its neighbors `(r, r')`: the closest pair of samples
/// in its visibility region seen next to each of them.
#[derive(Debug, Clone)]
struct InterfaceData<S> {
    /// `(point, sigma)` on the side of the smaller-id neighbor: `point` lies in this guard's
    /// region, `sigma` just outside, in the neighbor's.
    first: Option<(S, S)>,
    /// The same on the side of the larger-id neighbor.
    second: Option<(S, S)>,
}

impl<S> Default for InterfaceData<S> {
    fn default() -> Self {
        Self {
            first: None,
            second: None,
        }
    }
}

/// One guard's interface data, keyed by neighbor pair `(smaller id, larger id)`.
type Interfaces<S> = BTreeMap<(usize, usize), InterfaceData<S>>;

/// SPARS2: an incremental sparse roadmap spanner that needs no dense graph (Dobson and Bekris
/// 2013, "Improving sparse roadmap spanners"), ported from OMPL's `SPARStwo`.
///
/// Every valid sample `q` is tested against four criteria, in order, using the guards it can
/// see within `sparse_delta`:
///
/// 1. **Coverage:** it sees no guard, so it becomes a guard.
/// 2. **Connectivity:** it sees guards of different components, so it becomes a guard joining
///    them.
/// 3. **Interface:** its two closest guards are both visible but not adjacent, so they are
///    connected directly, or through `q`.
/// 4. **Quality:** `near_samples` states near `q` reveal neighboring guards' regions; the pair
///    points next to each interface are recorded per guard, and when the graph distance between
///    two neighbors of a guard exceeds `stretch` times the distance between their interface
///    points, a shortcut is added: the direct edge if valid, else the path through the points.
///
/// Construction ends after `max_failures` consecutive samples change nothing. The roadmap is
/// then, with high probability, a spanner: paths through it cost at most about `stretch` times
/// the optimum plus a term in `sparse_delta` (asymptotic near-optimality), with far fewer nodes
/// than a PRM.
///
/// Differences to OMPL's `SPARStwo`, documented here so that results can be compared: the
/// quality criterion uses OMPL's midpoint estimate `(d(r, v) + d(v, r'')) / 2` for the path
/// through `v`, as OMPL does (the paper takes a graph shortest path), but adds the path of
/// interface points without OMPL's smoothing step; the stretch test around a guard stops at the
/// first violation, even when no shortcut could be added (OMPL finishes the neighbor's inner
/// loop); any change to the graph resets the failure
/// counter (OMPL resets it only for new guards and direct interface edges); and sampling near
/// a state follows the geodesic toward a uniform sample, [`NEAR_ATTEMPTS`] tries at most, instead
/// of OMPL's unbounded retry.
///
/// # Examples
///
/// ```
/// use rand_core::SeedableRng;
/// use sampling_planning::{
///     Euclidean, GoalRegion, UniformBox, Validity, ValidityChecker,
///     roadmaps::{Spars2, Spars2Config},
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
/// let mut config = Spars2Config::new(2.5, 0.05);
/// config.max_failures = 500;
/// let mut spars = Spars2::new(Euclidean::<2>, sampler, Free, config).expect("valid");
/// spars.construct(100_000, &mut rand_pcg::Pcg64Mcg::seed_from_u64(1));
///
/// assert!(spars.is_complete());
/// let path = spars.query(&[1.0, 1.0], &GoalRegion::new([9.0, 9.0], 0.5)).expect("valid start");
/// assert!(path.expect("free space").cost < 3.0 * 11.4);
/// ```
#[derive(Debug, Clone)]
pub struct Spars2<Sp: StateSpace, Sa, C, N = LinearScan<<Sp as StateSpace>::State>> {
    /// The configuration space.
    space: Sp,
    /// Draws candidate states.
    sampler: Sa,
    /// Decides which states and motions are allowed.
    checker: C,
    /// The validated settings.
    config: Spars2Config,
    /// The sparse roadmap.
    graph: SparseGraph<Sp::State, N>,
    /// Per guard, the interface data of each neighbor pair `(smaller id, larger id)`.
    interfaces: Vec<Interfaces<Sp::State>>,
    /// Consecutive samples that changed nothing.
    failures: usize,
}

/// The most attempts [`Spars2`] makes to draw one valid state near a sample.
pub const NEAR_ATTEMPTS: usize = 16;

impl<Sp, Sa, C> Spars2<Sp, Sa, C>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
{
    /// Creates an empty roadmap with a [`LinearScan`] index over its guards.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRadius`] if a range is not positive and finite, `dense_delta` is not
    /// below `sparse_delta`, or the stretch is not above one.
    pub fn new(space: Sp, sampler: Sa, checker: C, config: Spars2Config) -> Result<Self, Error> {
        Self::with_nearest_neighbors(space, sampler, checker, config, LinearScan::default())
    }
}

impl<Sp, Sa, C, N> Spars2<Sp, Sa, C, N>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    N: NearestNeighbors<Sp>,
{
    /// Creates an empty roadmap with the given empty index over its guards.
    ///
    /// # Errors
    ///
    /// The same as [`Spars2::new`].
    pub fn with_nearest_neighbors(
        space: Sp,
        sampler: Sa,
        checker: C,
        config: Spars2Config,
        nn: N,
    ) -> Result<Self, Error> {
        config.validate()?;

        Ok(Self {
            space,
            sampler,
            checker,
            config,
            graph: SparseGraph::new(nn),
            interfaces: Vec::new(),
            failures: 0,
        })
    }

    /// Processes samples until construction is complete or `max_samples` were drawn. Returns
    /// the number of samples drawn.
    pub fn construct<R: Rng + ?Sized>(&mut self, max_samples: usize, rng: &mut R) -> usize {
        let mut drawn = 0;

        while drawn < max_samples && !self.is_complete() {
            self.step(rng);
            drawn += 1;
        }

        drawn
    }

    /// Draws one sample and applies the four criteria. Returns whether the roadmap changed.
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> bool {
        let before = (self.graph.len(), self.graph.edge_count());
        let q = self.sampler.sample(rng);

        if self.checker.check_state(&q).is_valid() {
            self.process(q, rng);
        }

        let changed = before != (self.graph.len(), self.graph.edge_count());
        self.failures = if changed { 0 } else { self.failures + 1 };

        changed
    }

    /// Applies the criteria to the valid sample `q`; the quality test runs only if none of the
    /// other three added anything.
    fn process<R: Rng + ?Sized>(&mut self, q: Sp::State, rng: &mut R) {
        let (near, visible) = self.neighborhood(&q);

        if visible.is_empty() {
            self.add_guard(q, GuardKind::Coverage);
        } else if !self.check_connectivity(&q, &visible)
            && !self.check_interface(&q, &near, &visible)
        {
            self.check_quality(&q, visible[0], rng);
        }
    }

    /// The guards within `sparse_delta` of `state`, closest first, and the visible ones.
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

    /// Adds `state` as a guard and forgets the interface data of every guard that can see it,
    /// since their regions just shrank.
    fn add_guard(&mut self, state: Sp::State, kind: GuardKind) -> usize {
        let mut near = Vec::new();
        self.graph
            .nn
            .within_radius(&self.space, &state, self.config.sparse_delta, &mut near);

        for guard in near {
            self.interfaces[guard].clear();
        }

        self.interfaces.push(BTreeMap::new());

        self.graph.add_guard(state, kind)
    }

    /// Connectivity: if `q` sees guards of different components, adds it as a guard joined to
    /// one guard of each. Returns whether it did.
    fn check_connectivity(&mut self, q: &Sp::State, visible: &[usize]) -> bool {
        let links = self.graph.connectivity_links(visible);

        if links.is_empty() {
            return false;
        }

        let guard = self.add_guard(q.clone(), GuardKind::Connectivity);
        self.graph.connect_links(guard, &links);

        true
    }

    /// Interface: if the two closest guards of `q` are both visible but not adjacent, connects
    /// them directly or through `q`. Returns whether it did.
    fn check_interface(&mut self, q: &Sp::State, near: &[usize], visible: &[usize]) -> bool {
        let Some((v1, v2)) = self.graph.interface_pair(near, visible) else {
            return false;
        };

        if self.is_edge_valid(&self.graph.states[v1], &self.graph.states[v2]) {
            self.graph.connect(v1, v2);
        } else {
            let guard = self.add_guard(q.clone(), GuardKind::Interface);
            self.graph.connect(guard, v1);
            self.graph.connect(guard, v2);
        }

        true
    }

    /// Quality: finds the guards of states near `q` (whose representative is `v`), records the
    /// interface points they reveal, and tests the stretch bound around `v` and around each of
    /// them.
    fn check_quality<R: Rng + ?Sized>(&mut self, q: &Sp::State, v: usize, rng: &mut R) {
        let Some(representatives) = self.close_representatives(q, v, rng) else {
            return;
        };

        for (r, sigma) in &representatives {
            self.update_pair_points(v, q, *r, sigma);
            self.update_pair_points(*r, sigma, v, q);
        }

        self.test_add_path(v);
        for (r, _) in &representatives {
            self.test_add_path(*r);
        }
    }

    /// Up to `near_samples` states `sigma` near `q` that see a guard other than `v`, one per
    /// such guard, as `(guard, sigma)`. Returns `None` if a near state sees no guard at all; it
    /// then became a coverage guard.
    fn close_representatives<R: Rng + ?Sized>(
        &mut self,
        q: &Sp::State,
        v: usize,
        rng: &mut R,
    ) -> Option<Vec<(usize, Sp::State)>> {
        let mut found: Vec<(usize, Sp::State)> = Vec::new();

        for _ in 0..self.config.near_samples {
            let Some(sigma) = self.sample_near(q, rng) else {
                continue;
            };

            let (_, visible) = self.neighborhood(&sigma);
            let Some(&representative) = visible.first() else {
                self.add_guard(sigma, GuardKind::Coverage);
                return None;
            };

            if representative != v && found.iter().all(|(r, _)| *r != representative) {
                found.push((representative, sigma));
            }
        }

        Some(found)
    }

    /// A valid state within `dense_delta` of `q` that `q` can reach: a random fraction of
    /// `dense_delta` along the geodesic toward a fresh sample. `None` after
    /// [`NEAR_ATTEMPTS`] failures.
    fn sample_near<R: Rng + ?Sized>(&mut self, q: &Sp::State, rng: &mut R) -> Option<Sp::State> {
        for _ in 0..NEAR_ATTEMPTS {
            let toward = self.sampler.sample(rng);
            let distance = self.space.distance(q, &toward);
            let step = self.config.dense_delta * unit_f64(rng);

            if distance <= 0.0 {
                continue;
            }

            let sigma = self
                .space
                .interpolate(q, &toward, (step / distance).min(1.0));

            if self.space.distance(q, &sigma) <= self.config.dense_delta
                && self.checker.check_state(&sigma).is_valid()
                && self.is_edge_valid(q, &sigma)
            {
                return Some(sigma);
            }
        }

        None
    }

    /// Records `point` (in the region of guard `v`, next to the region of its neighbor `r`,
    /// where `sigma` lies) for every pair `(r, r')` of `v`'s neighbors that are not adjacent,
    /// if it brings the pair's points closer.
    fn update_pair_points(&mut self, v: usize, point: &Sp::State, r: usize, sigma: &Sp::State) {
        for r_prime in self.graph.non_adjacent_neighbors(v, r) {
            self.distance_check(v, point, r, sigma, r_prime);
        }
    }

    /// Stores `(point, sigma)` as `v`'s interface point toward `r` for the pair `(r, r')` when
    /// the slot is empty or the new point is closer to the other side's point (OMPL's
    /// `distanceCheck`).
    fn distance_check(
        &mut self,
        v: usize,
        point: &Sp::State,
        r: usize,
        sigma: &Sp::State,
        r_prime: usize,
    ) {
        let space = &self.space;
        let data = self.interfaces[v].entry(pair(r, r_prime)).or_default();
        let (mine, other) = if r < r_prime {
            (&mut data.first, &data.second)
        } else {
            (&mut data.second, &data.first)
        };

        let replace = match (&*mine, other) {
            (None, _) => true,
            (Some(_), None) => false,
            (Some((current, _)), Some((opposite, _))) => {
                space.distance(point, opposite) < space.distance(current, opposite)
            }
        };

        if replace {
            *mine = Some((point.clone(), sigma.clone()));
        }
    }

    /// The distance between the two interface points of pair `(r, r')` at guard `v`, or
    /// infinity while one is missing.
    fn interface_distance(&self, v: usize, r: usize, r_prime: usize) -> f64 {
        match self.interfaces[v].get(&pair(r, r_prime)) {
            Some(InterfaceData {
                first: Some((a, _)),
                second: Some((b, _)),
            }) => self.space.distance(a, b),
            _ => f64::INFINITY,
        }
    }

    /// Whether guard `v` holds the interface point on `r''`'s side of the pair `(r'', x)`.
    fn has_point_toward(&self, v: usize, r_pp: usize, x: usize) -> bool {
        self.interfaces[v].get(&pair(r_pp, x)).is_some_and(|data| {
            if r_pp < x {
                data.first.is_some()
            } else {
                data.second.is_some()
            }
        })
    }

    /// The stretch test around guard `v` (OMPL's `checkAddPath`): for each neighbor `r` and
    /// each non-adjacent neighbor `r'`, compares the estimated path through `v` with the
    /// interface distance, and adds a shortcut at the first violation. Returns whether it added
    /// one.
    fn test_add_path(&mut self, v: usize) -> bool {
        let neighbors = self.graph.adjacency[v].clone();

        for r in neighbors {
            for r_prime in self.graph.non_adjacent_neighbors(v, r) {
                let through_v = self.longest_path_estimate(v, r, r_prime);

                if through_v > self.config.stretch * self.interface_distance(v, r, r_prime) {
                    self.add_shortcut(v, r, r_prime);
                    return true;
                }
            }
        }

        false
    }

    /// `max over x of (d(r, v) + d(v, x)) / 2`, where `x` is `r'` or a neighbor of `r'` adjacent
    /// to `v` but not to `r` for which `v` holds an interface point (OMPL's `computeX`).
    fn longest_path_estimate(&self, v: usize, r: usize, r_prime: usize) -> f64 {
        let states = &self.graph.states;
        let to_r = self.space.distance(&states[r], &states[v]);

        self.graph.adjacency[r_prime]
            .iter()
            .copied()
            .filter(|&x| {
                self.graph.has_edge(x, v)
                    && !self.graph.has_edge(x, r)
                    && self.has_point_toward(v, r_prime, x)
            })
            .chain([r_prime])
            .map(|x| f64::midpoint(to_r, self.space.distance(&states[v], &states[x])))
            .fold(0.0, f64::max)
    }

    /// Connects `r` and `r'`: directly if that motion is valid, otherwise through the stored
    /// interface points `sigma_r, point_r, v, point_r', sigma_r'` when every motion along that
    /// chain is valid.
    fn add_shortcut(&mut self, v: usize, r: usize, r_prime: usize) {
        if self.is_edge_valid(&self.graph.states[r], &self.graph.states[r_prime]) {
            self.graph.connect(r, r_prime);
            return;
        }

        let Some(data) = self.interfaces[v].get(&pair(r, r_prime)).cloned() else {
            return;
        };
        let (Some((point_a, sigma_a)), Some((point_b, sigma_b))) = (data.first, data.second) else {
            return;
        };
        let ((point_r, sigma_r), (point_rp, sigma_rp)) = if r < r_prime {
            ((point_a, sigma_a), (point_b, sigma_b))
        } else {
            ((point_b, sigma_b), (point_a, sigma_a))
        };

        self.add_chain(r, [sigma_r, point_r], v, [point_rp, sigma_rp], r_prime);
    }

    /// Adds the guards `before` between `r` and `v` and `after` between `v` and `r'`, chained
    /// by edges, if every motion along `r, before, v, after, r'` checks as valid.
    fn add_chain(
        &mut self,
        r: usize,
        before: [Sp::State; 2],
        v: usize,
        after: [Sp::State; 2],
        r_prime: usize,
    ) {
        let states = &self.graph.states;
        let chain: Vec<&Sp::State> = [&states[r]]
            .into_iter()
            .chain(&before)
            .chain([&states[v]])
            .chain(&after)
            .chain([&states[r_prime]])
            .collect();

        if !chain
            .windows(2)
            .all(|pair| self.is_edge_valid(pair[0], pair[1]))
        {
            return;
        }

        let mut prior = r;
        for state in before {
            let guard = self.add_guard(state, GuardKind::Quality);
            self.graph.connect(prior, guard);
            prior = guard;
        }

        self.graph.connect(prior, v);
        prior = v;

        for state in after {
            let guard = self.add_guard(state, GuardKind::Quality);
            self.graph.connect(prior, guard);
            prior = guard;
        }

        self.graph.connect(prior, r_prime);
    }

    /// The cheapest path from `start` to `goal` through the roadmap, or `None` if the roadmap
    /// does not connect them. Both endpoints connect to every visible guard within
    /// `sparse_delta`.
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

    /// Whether `max_failures` consecutive samples changed nothing.
    #[must_use]
    pub const fn is_complete(&self) -> bool {
        self.failures >= self.config.max_failures
    }

    /// The number of guards.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.graph.len()
    }

    /// The number of roadmap edges.
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

/// The interface-data key of the neighbor pair `(a, b)`: `(smaller id, larger id)`.
fn pair(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}
