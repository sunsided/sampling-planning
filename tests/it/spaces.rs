//! Integration tests for the planners over SE(2), Dubins and Reeds-Shepp spaces.

use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    Dubins, GoalRegion, Path, Problem, ReedsShepp, Se2, Se2State, StateSpace, StraightLine,
    UniformSe2, Validity, ValidityChecker, discretized_motion,
    trees::{Rrt, RrtConfig, RrtStar, RrtStarConfig},
};

/// A 10 x 10 world with one wall from `(4, 0)` to `(6, 6)`; positions outside are invalid.
#[derive(Debug, Clone, Copy)]
struct Yard;

impl Yard {
    /// The validity of a pose by its position.
    fn check(state: &Se2State) -> Validity {
        let inside = (0.0..=10.0).contains(&state.x) && (0.0..=10.0).contains(&state.y);
        let wall = (4.0..=6.0).contains(&state.x) && state.y <= 6.0;

        if inside && !wall {
            Validity::Valid
        } else {
            Validity::Invalid
        }
    }
}

impl<Sp: StateSpace<State = Se2State>> ValidityChecker<Sp> for Yard {
    fn check_state(&self, state: &Se2State) -> Validity {
        Self::check(state)
    }

    fn check_motion(&self, space: &Sp, from: &Se2State, to: &Se2State) -> Validity {
        discretized_motion(space, from, to, 0.05, Self::check)
    }
}

/// The yard problem in `space`: from `(1, 1)` heading north to `(9, 1)` heading south.
fn yard<Sp: StateSpace<State = Se2State>>(
    space: Sp,
) -> Problem<Sp, UniformSe2, Yard, StraightLine> {
    Problem {
        space,
        sampler: UniformSe2::new([0.0, 0.0], [10.0, 10.0]),
        checker: Yard,
        steering: StraightLine::new(1.0),
        start: Se2State::new(1.0, 1.0, core::f64::consts::FRAC_PI_2),
        goal: GoalRegion::new(Se2State::new(9.0, 1.0, -core::f64::consts::FRAC_PI_2), 0.3),
    }
}

/// Asserts that `path` solves the yard problem in `space`: it starts at the start, ends in
/// the goal, every edge checks as valid in travel direction, and its cost is its length.
fn assert_solves<Sp: StateSpace<State = Se2State> + Clone>(space: &Sp, path: &Path<Se2State>) {
    let problem = yard(space.clone());
    assert_eq!(path.states.first(), Some(&problem.start));
    assert!(
        problem
            .goal
            .contains(space, path.states.last().expect("non-empty"))
    );

    let mut length = 0.0;
    for pair in path.states.windows(2) {
        let validity = Yard.check_motion(space, &pair[0], &pair[1]);
        assert_eq!(validity, Validity::Valid, "{pair:?}");
        length += space.distance(&pair[0], &pair[1]);
    }

    assert!(
        (length - path.cost).abs() < 1e-9,
        "{length} vs {}",
        path.cost
    );
}

/// An RRT* config for the yard with a generous radius constant.
fn star_config() -> RrtStarConfig {
    RrtStarConfig::new(30.0, 1.0)
}

#[test]
fn rrt_star_plans_over_se2() {
    let space = Se2::new(0.5);
    let mut planner = RrtStar::new(yard(space), star_config()).expect("valid problem");
    planner.run(3000, &mut Pcg64Mcg::seed_from_u64(1));

    let path = planner.best_path().expect("RRT* solves the SE(2) yard");
    assert_solves(&space, &path);
}

#[test]
fn rrt_plans_forward_only_over_dubins() {
    let space = Dubins::new(0.8);
    let mut planner = Rrt::new(yard(space), RrtConfig::default()).expect("valid problem");
    let path = planner
        .solve(20_000, &mut Pcg64Mcg::seed_from_u64(2))
        .expect("RRT solves the Dubins yard");

    // Every edge is a forward Dubins curve that the checker accepted in travel direction.
    assert_solves(&space, &path);
}

#[test]
fn rrt_star_plans_over_reeds_shepp_and_improves() {
    let space = ReedsShepp::new(0.8);
    let mut planner = RrtStar::new(yard(space), star_config()).expect("valid problem");
    let mut rng = Pcg64Mcg::seed_from_u64(3);
    let mut costs = Vec::new();

    // Reeds-Shepp distances cost a few microseconds each and the linear scan computes one per
    // node per query, so this run stays short.
    for _ in 0..4 {
        planner.run(200, &mut rng);
        costs.extend(planner.best_path().map(|path| path.cost));
    }

    let path = planner
        .best_path()
        .expect("RRT* solves the Reeds-Shepp yard");
    assert_solves(&space, &path);
    assert!(costs.windows(2).all(|pair| pair[1] <= pair[0]), "{costs:?}");
}
