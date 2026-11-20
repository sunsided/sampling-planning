//! PRM, PRM* and lazy PRM: multi-query probabilistic roadmaps.

use alloc::vec::Vec;

use rand_core::Rng;

use super::{
    Error,
    graph::{Arc, Route, shortest_path},
};
use crate::radius::{dimension_f64, shrinking_radius, unit_ball_volume};
use crate::{GoalRegion, LinearScan, NearestNeighbors, Path, Sampler, StateSpace, ValidityChecker};

/// How a new roadmap node picks the nodes it tries to connect to.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Connection {
    /// Classic PRM: every node within this fixed radius. Positive and finite.
    Radius(f64),
    /// PRM* (Karaman and Frazzoli 2011): every node within `gamma * (ln n / n)^(1/d)`, with `n`
    /// counting the new node. Asymptotic optimality needs `gamma` above
    /// [`PrmConfig::gamma_lower_bound`]. Positive and finite.
    Star {
        /// The radius constant.
        gamma: f64,
    },
}

/// Settings for [`Prm`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrmConfig {
    /// The neighbor rule for new nodes and for query endpoints.
    pub connection: Connection,
    /// Lazy PRM (Bohlin and Kavraki 2000): store edges unchecked and check only the edges of
    /// candidate shortest paths at query time, dropping invalid ones and searching again.
    ///
    /// Nodes are still state-checked when they are added. Off by default.
    pub lazy: bool,
    /// Check every roadmap edge in both directions.
    ///
    /// Roadmap edges are traveled both ways, but checked once, from the older to the newer node.
    /// That is sound only for a symmetric [`ValidityChecker::check_motion`]; switch this on for
    /// directional checkers (reachable sets, one-way constraints). Query edges from the start
    /// and to the goal are always checked in travel direction only. Off by default.
    pub check_both_directions: bool,
}

impl PrmConfig {
    /// Classic PRM with a fixed connection radius, eager checks.
    #[must_use]
    pub const fn fixed(radius: f64) -> Self {
        Self {
            connection: Connection::Radius(radius),
            lazy: false,
            check_both_directions: false,
        }
    }

    /// PRM* with the radius constant `gamma`, eager checks.
    #[must_use]
    pub const fn star(gamma: f64) -> Self {
        Self {
            connection: Connection::Star { gamma },
            lazy: false,
            check_both_directions: false,
        }
    }

    /// This config with lazy edge checking.
    #[must_use]
    pub const fn lazy(self) -> Self {
        Self { lazy: true, ..self }
    }

    /// The smallest `gamma` for which PRM* is asymptotically optimal (Karaman and Frazzoli
    /// 2011, Theorem 34): `2 (1 + 1/d)^(1/d) (free_volume / zeta_d)^(1/d)`, where `zeta_d` is the
    /// volume of the unit `d`-ball.
    ///
    /// `free_volume` is the Lebesgue measure of the obstacle-free space; the volume of the
    /// sampling box is a safe over-estimate. Pick `gamma` somewhat above this value.
    #[must_use]
    pub fn gamma_lower_bound(dimension: usize, free_volume: f64) -> f64 {
        let d = dimension_f64(dimension);

        2.0 * libm::pow(1.0 + 1.0 / d, 1.0 / d)
            * libm::pow(free_volume / unit_ball_volume(d), 1.0 / d)
    }
}

/// Whether a roadmap edge has been collision-checked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum EdgeStatus {
    /// Checked and valid.
    Valid,
    /// Checked and invalid (or unknown); never traveled. Only lazy roadmaps keep such edges.
    Invalid,
    /// Not checked yet; only in lazy roadmaps.
    Unchecked,
}

/// An undirected roadmap edge.
#[derive(Debug, Clone)]
struct Edge {
    /// The older endpoint.
    a: usize,
    /// The newer endpoint.
    b: usize,
    /// The edge length under [`StateSpace::distance`].
    cost: f64,
    /// Whether the edge was checked, and how.
    status: EdgeStatus,
}

/// A probabilistic roadmap: a graph of valid sampled states connected by valid motions, built
/// once and queried for many start and goal pairs.
///
/// [`Prm::grow`] adds samples, [`Prm::query`] connects a start and a goal region to the roadmap
/// and runs Dijkstra. Queries do not change the roadmap, except that a lazy roadmap remembers
/// which edges it has checked. With [`Connection::Star`] the roadmap is PRM*, whose query cost
/// converges to the optimum as it grows.
///
/// PRM needs no [`Steering`](crate::Steering): edges are direct motions along
/// [`StateSpace::interpolate`].
///
/// # Examples
///
/// ```
/// use rand_core::SeedableRng;
/// use sampling_planning::{
///     Euclidean, GoalRegion, UniformBox, Validity, ValidityChecker,
///     roadmaps::{Prm, PrmConfig},
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
/// let gamma = 1.5 * PrmConfig::gamma_lower_bound(2, 100.0);
/// let mut prm = Prm::new(Euclidean::<2>, sampler, Free, PrmConfig::star(gamma)).expect("valid");
/// prm.grow(500, &mut rand_pcg::Pcg64Mcg::seed_from_u64(1));
///
/// // The same roadmap answers several queries.
/// for goal in [[9.0, 9.0], [9.0, 1.0]] {
///     let path = prm.query(&[1.0, 1.0], &GoalRegion::new(goal, 0.5)).expect("valid start");
///     assert!(path.is_some());
/// }
/// ```
#[derive(Debug, Clone)]
pub struct Prm<Sp: StateSpace, Sa, C, N = LinearScan<<Sp as StateSpace>::State>> {
    /// The configuration space.
    space: Sp,
    /// Draws candidate roadmap states.
    sampler: Sa,
    /// Decides which states and motions are allowed.
    checker: C,
    /// The settings.
    config: PrmConfig,
    /// The roadmap states, indexed by node id. All checked as valid.
    states: Vec<Sp::State>,
    /// All edges, including checked-invalid ones in lazy roadmaps.
    edges: Vec<Edge>,
    /// The ids of the edges touching each node, ascending.
    incident: Vec<Vec<usize>>,
    /// Index over all roadmap nodes.
    nn: N,
    /// Reused buffer for neighbor queries.
    near: Vec<usize>,
}

impl<Sp, Sa, C> Prm<Sp, Sa, C>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
{
    /// Creates an empty roadmap with a [`LinearScan`] nearest-neighbor index.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRadius`] if the radius or `gamma` is not positive and finite.
    pub fn new(space: Sp, sampler: Sa, checker: C, config: PrmConfig) -> Result<Self, Error> {
        Self::with_nearest_neighbors(space, sampler, checker, config, LinearScan::default())
    }
}

impl<Sp, Sa, C, N> Prm<Sp, Sa, C, N>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    N: NearestNeighbors<Sp>,
{
    /// Creates an empty roadmap with the given empty nearest-neighbor index.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidRadius`] if the radius or `gamma` is not positive and finite.
    pub fn with_nearest_neighbors(
        space: Sp,
        sampler: Sa,
        checker: C,
        config: PrmConfig,
        nn: N,
    ) -> Result<Self, Error> {
        let (Connection::Radius(value) | Connection::Star { gamma: value }) = config.connection;
        if !(value.is_finite() && value > 0.0) {
            return Err(Error::InvalidRadius);
        }

        Ok(Self {
            space,
            sampler,
            checker,
            config,
            states: Vec::new(),
            edges: Vec::new(),
            incident: Vec::new(),
            nn,
            near: Vec::new(),
        })
    }

    /// Draws `samples` states and adds the valid ones to the roadmap, each connected to its
    /// neighbors. Returns the number of nodes added.
    pub fn grow<R: Rng + ?Sized>(&mut self, samples: usize, rng: &mut R) -> usize {
        let before = self.states.len();

        for _ in 0..samples {
            let state = self.sampler.sample(rng);

            if self.checker.check_state(&state).is_valid() {
                self.add_node(state);
            }
        }

        self.states.len() - before
    }

    /// Adds a valid state as a node and connects it to the nodes within the current radius.
    fn add_node(&mut self, state: Sp::State) {
        let id = self.states.len();
        let mut near = core::mem::take(&mut self.near);
        self.nn
            .within_radius(&self.space, &state, self.radius(id + 1), &mut near);

        self.states.push(state.clone());
        self.incident.push(Vec::new());

        for &other in &near {
            self.try_connect(other, id);
        }

        self.nn.insert(id, state);
        self.near = near;
    }

    /// Adds the edge between the older node `a` and the newer node `b`, checked unless lazy.
    fn try_connect(&mut self, a: usize, b: usize) {
        let status = if self.config.lazy {
            EdgeStatus::Unchecked
        } else if self.is_edge_valid(a, b) {
            EdgeStatus::Valid
        } else {
            return;
        };

        let edge = self.edges.len();
        self.edges.push(Edge {
            a,
            b,
            cost: self.space.distance(&self.states[a], &self.states[b]),
            status,
        });
        self.incident[a].push(edge);
        self.incident[b].push(edge);
    }

    /// Whether the roadmap edge between nodes `a` and `b` checks as valid, in both directions
    /// if configured.
    fn is_edge_valid(&self, a: usize, b: usize) -> bool {
        let (sa, sb) = (&self.states[a], &self.states[b]);
        let forward = self.checker.check_motion(&self.space, sa, sb).is_valid();

        forward
            && (!self.config.check_both_directions
                || self.checker.check_motion(&self.space, sb, sa).is_valid())
    }

    /// The connection radius for a roadmap of `n` nodes.
    fn radius(&self, n: usize) -> f64 {
        match self.config.connection {
            Connection::Radius(radius) => radius,
            Connection::Star { gamma } => {
                shrinking_radius(gamma, n, dimension_f64(self.space.dimension()))
            }
        }
    }

    /// The cheapest path from `start` to `goal` through the roadmap, or `None` if the roadmap
    /// does not connect them yet.
    ///
    /// The start connects to every node within the current connection radius by a valid
    /// motion. The path ends at a roadmap node inside `goal` or at `goal`'s center, which
    /// connects like the start when it is valid. A lazy roadmap checks the edges of each
    /// candidate path, drops invalid ones and searches again, so its result equals the eager
    /// one for the same samples.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidStart`] if `start` does not check as valid.
    pub fn query(
        &mut self,
        start: &Sp::State,
        goal: &GoalRegion<Sp::State>,
    ) -> Result<Option<Path<Sp::State>>, Error> {
        let validity = self.checker.check_state(start);
        if !validity.is_valid() {
            return Err(Error::InvalidStart(validity));
        }

        let ends = self.connect_endpoints(start, goal);

        loop {
            let Some(route) = self.search(&ends, goal) else {
                return Ok(None);
            };

            if self.confirm_lazy_edges(&route) {
                return Ok(Some(self.to_path(&route, start, goal)));
            }
        }
    }

    /// The query-local arcs from the start into the roadmap and from the roadmap to the goal
    /// center, each checked in travel direction.
    fn connect_endpoints(&self, start: &Sp::State, goal: &GoalRegion<Sp::State>) -> Endpoints {
        let radius = self.radius(self.states.len().max(1));
        let center = goal.center();
        let center_valid = self.checker.check_state(center).is_valid();
        let mut near = Vec::new();

        self.nn.within_radius(&self.space, start, radius, &mut near);
        let from_start = near
            .iter()
            .filter(|&&node| self.is_motion_valid(start, &self.states[node]))
            .map(|&node| (node, self.space.distance(start, &self.states[node])))
            .collect();

        let mut to_center = alloc::vec![f64::INFINITY; self.states.len()];
        if center_valid {
            self.nn
                .within_radius(&self.space, center, radius, &mut near);

            for &node in &near {
                if self.is_motion_valid(&self.states[node], center) {
                    to_center[node] = self.space.distance(&self.states[node], center);
                }
            }
        }

        let direct = self.space.distance(start, center);
        let start_to_center =
            (center_valid && direct <= radius && self.is_motion_valid(start, center))
                .then_some(direct);

        Endpoints {
            from_start,
            to_center,
            start_to_center,
            start_in_goal: goal.contains(&self.space, start),
        }
    }

    /// Whether the motion from `from` to `to` checks as valid.
    fn is_motion_valid(&self, from: &Sp::State, to: &Sp::State) -> bool {
        self.checker.check_motion(&self.space, from, to).is_valid()
    }

    /// Dijkstra from the start over the roadmap, skipping edges known to be invalid.
    ///
    /// Node ids: roadmap nodes `0..n`, the start `n`, the goal center `n + 1`.
    fn search(&self, ends: &Endpoints, goal: &GoalRegion<Sp::State>) -> Option<Route> {
        let n = self.states.len();
        let (start, center) = (n, n + 1);

        let is_goal = |node: usize| {
            node == center
                || (node == start && ends.start_in_goal)
                || (node < n && goal.contains(&self.space, &self.states[node]))
        };

        let arcs = |node: usize, out: &mut Vec<Arc>| {
            out.clear();

            if node == start {
                out.extend(ends.from_start.iter().map(|&(to, cost)| Arc {
                    to,
                    cost,
                    edge: None,
                }));
                out.extend(ends.start_to_center.map(|cost| Arc {
                    to: center,
                    cost,
                    edge: None,
                }));
            } else if node < n {
                self.roadmap_arcs(node, out);

                if ends.to_center[node].is_finite() {
                    out.push(Arc {
                        to: center,
                        cost: ends.to_center[node],
                        edge: None,
                    });
                }
            }
        };

        shortest_path(n + 2, start, is_goal, arcs)
    }

    /// Appends the arcs along the not-invalid roadmap edges of `node`.
    fn roadmap_arcs(&self, node: usize, out: &mut Vec<Arc>) {
        for &id in &self.incident[node] {
            let edge = &self.edges[id];

            if edge.status != EdgeStatus::Invalid {
                let to = if edge.a == node { edge.b } else { edge.a };
                out.push(Arc {
                    to,
                    cost: edge.cost,
                    edge: Some(id),
                });
            }
        }
    }

    /// Checks the unchecked roadmap edges along `route` in travel order. Returns whether all are
    /// valid; marks the first invalid one and stops there otherwise.
    fn confirm_lazy_edges(&mut self, route: &Route) -> bool {
        for (step, edge) in route.edges.iter().enumerate() {
            let Some(id) = *edge else { continue };

            if self.edges[id].status != EdgeStatus::Unchecked {
                continue;
            }

            let (from, to) = (route.nodes[step], route.nodes[step + 1]);
            let valid = self.is_motion_valid(&self.states[from], &self.states[to])
                && (!self.config.check_both_directions
                    || self.is_motion_valid(&self.states[to], &self.states[from]));

            self.edges[id].status = if valid {
                EdgeStatus::Valid
            } else {
                EdgeStatus::Invalid
            };
            if !valid {
                return false;
            }
        }

        true
    }

    /// Converts a route over query node ids into a path of states.
    fn to_path(
        &self,
        route: &Route,
        start: &Sp::State,
        goal: &GoalRegion<Sp::State>,
    ) -> Path<Sp::State> {
        let n = self.states.len();
        let states = route
            .nodes
            .iter()
            .map(|&node| match node {
                node if node < n => self.states[node].clone(),
                node if node == n => start.clone(),
                _ => goal.center().clone(),
            })
            .collect();

        Path {
            states,
            cost: route.cost,
        }
    }

    /// The number of roadmap nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.states.len()
    }

    /// The number of roadmap edges that are valid or not checked yet.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.edges
            .iter()
            .filter(|edge| edge.status != EdgeStatus::Invalid)
            .count()
    }
}

/// Query-local connections of a start state and a goal region to the roadmap.
struct Endpoints {
    /// Roadmap nodes reachable from the start by a valid motion, with the motion cost.
    from_start: Vec<(usize, f64)>,
    /// Per roadmap node, the cost of a valid motion to the goal center, or infinity.
    to_center: Vec<f64>,
    /// The cost of a valid direct motion from the start to the goal center, if any.
    start_to_center: Option<f64>,
    /// Whether the start itself lies in the goal region.
    start_in_goal: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Euclidean, UniformBox, Validity, discretized_motion};
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

    /// A PRM* over the wall world grown by 400 samples with a fixed seed.
    fn roadmap(lazy: bool) -> Prm<Euclidean<2>, UniformBox<2>, Wall> {
        let gamma = 1.5 * PrmConfig::gamma_lower_bound(2, 100.0);
        let config = if lazy {
            PrmConfig::star(gamma).lazy()
        } else {
            PrmConfig::star(gamma)
        };
        let mut prm = Prm::new(
            Euclidean::<2>,
            UniformBox::new([0.0; 2], [10.0; 2]),
            Wall,
            config,
        )
        .expect("valid config");
        prm.grow(400, &mut rand_pcg::Pcg64Mcg::seed_from_u64(3));

        prm
    }

    #[test]
    fn lazy_and_eager_roadmaps_agree() {
        let (mut eager, mut lazy) = (roadmap(false), roadmap(true));
        let goal = GoalRegion::new([9.0, 1.0], 0.3);

        let a = eager
            .query(&[1.0, 1.0], &goal)
            .expect("valid")
            .expect("connected");
        let b = lazy
            .query(&[1.0, 1.0], &goal)
            .expect("valid")
            .expect("connected");

        assert!((a.cost - b.cost).abs() < 1e-9, "{} vs {}", a.cost, b.cost);
        assert!(
            lazy.edge_count() > eager.edge_count(),
            "lazy keeps unchecked edges"
        );
        for pair in b.states.windows(2) {
            assert_eq!(
                Wall.check_motion(&Euclidean::<2>, &pair[0], &pair[1]),
                Validity::Valid
            );
        }
    }

    #[test]
    fn rejects_bad_radius_and_invalid_start() {
        let make = |config| {
            Prm::new(
                Euclidean::<2>,
                UniformBox::new([0.0; 2], [1.0; 2]),
                Wall,
                config,
            )
        };

        assert_eq!(
            make(PrmConfig::fixed(0.0)).err(),
            Some(Error::InvalidRadius)
        );
        assert_eq!(
            make(PrmConfig::star(f64::NAN)).err(),
            Some(Error::InvalidRadius)
        );
        assert_eq!(
            roadmap(false).query(&[5.0, 1.0], &GoalRegion::new([9.0, 1.0], 0.3)),
            Err(Error::InvalidStart(Validity::Invalid))
        );
    }
}
