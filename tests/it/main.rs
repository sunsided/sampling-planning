//! Integration tests for the tree planners on a 2D world with obstacles.

mod batch;
mod informed;
mod kinodynamic;
mod objective;
mod properties;
mod retirement;
mod riemannian;
mod roadmaps;
mod spaces;
mod world;

use rand_core::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    Euclidean, KdTree, Problem, Sampler, StraightLine, UniformBox, Validity, shortcut,
    shortcut_random,
    trees::{Error, Rrt, RrtConfig, RrtConnect, RrtSharp, RrtStar, RrtStarConfig},
};
use world::{
    Boxes, POLE_OPTIMUM, World2d, assert_solves, assert_solves_with, pole, problem, slalom,
};

/// The optimal slalom cost, rounded down: the taut string from (1, 1) over the corners (4, 7)
/// and (5, 7), under (6, 3) and (7, 3), to the goal ball of radius 0.5 around (9, 9), which is
/// `19.1559 - 0.5`. No solution can be cheaper.
pub const SLALOM_OPTIMUM: f64 = 18.655;

/// An RRT* config whose radius constant is safe for the 10 x 10 world.
fn star_config() -> RrtStarConfig {
    RrtStarConfig::new(1.5 * RrtStarConfig::gamma_lower_bound(2, 100.0), 1.0)
}

/// Runs RRT* on `world` for `iterations` iterations with the given seed.
fn rrt_star(
    world: World2d,
    seed: u64,
    iterations: usize,
) -> RrtStar<Euclidean<2>, UniformBox<2>, Boxes, StraightLine> {
    let mut planner = RrtStar::new(world, star_config()).expect("valid problem");
    planner.run(iterations, &mut Pcg64Mcg::seed_from_u64(seed));

    planner
}

#[test]
fn rrt_finds_valid_path_in_bounds() {
    for seed in 0..5 {
        let mut planner = Rrt::new(slalom(), RrtConfig::default()).expect("valid problem");
        let path = planner
            .solve(20_000, &mut Pcg64Mcg::seed_from_u64(seed))
            .expect("RRT solves the slalom");

        assert_solves(&slalom(), &path);
    }
}

#[test]
fn rrt_star_finds_valid_path_in_bounds() {
    for seed in 0..3 {
        let path = rrt_star(slalom(), seed, 3000)
            .best_path()
            .expect("RRT* solves the slalom");

        assert_solves(&slalom(), &path);
    }
}

#[test]
fn rrt_star_cost_never_increases() {
    let mut planner = RrtStar::new(slalom(), star_config()).expect("valid problem");
    let mut rng = Pcg64Mcg::seed_from_u64(11);
    let mut costs = Vec::new();

    for _ in 0..20 {
        planner.run(200, &mut rng);
        costs.extend(planner.best_path().map(|path| path.cost));
    }

    assert!(costs.len() > 5, "found a solution early enough");
    assert!(costs.windows(2).all(|pair| pair[1] <= pair[0]), "{costs:?}");
    assert!(
        costs.last() < costs.first(),
        "RRT* improved the path: {costs:?}"
    );
    assert!(
        costs.iter().all(|&cost| cost >= SLALOM_OPTIMUM),
        "{costs:?}"
    );
}

#[test]
fn rrt_star_is_reproducible_for_a_seed() {
    let a = rrt_star(slalom(), 5, 1000).best_path();
    let b = rrt_star(slalom(), 5, 1000).best_path();

    assert_eq!(a, b);
}

#[test]
fn kd_tree_gives_the_same_rrt_star_run_as_linear_scan() {
    let mut planner = RrtStar::with_nearest_neighbors(slalom(), star_config(), KdTree::default())
        .expect("valid problem");
    planner.run(1500, &mut Pcg64Mcg::seed_from_u64(5));

    assert_eq!(planner.best_path(), rrt_star(slalom(), 5, 1500).best_path());
}

/// An RRT*-k config with a constant safely above the bound and the steering step as cap.
fn star_k_config() -> RrtStarConfig {
    RrtStarConfig::k_nearest(1.1 * RrtStarConfig::k_constant_lower_bound(2), 1.0)
}

#[test]
fn rrt_star_k_solves_improves_and_matches_across_indices() {
    let mut planner = RrtStar::new(slalom(), star_k_config()).expect("valid problem");
    let mut kd = RrtStar::with_nearest_neighbors(slalom(), star_k_config(), KdTree::default())
        .expect("valid problem");
    let (mut rng, mut kd_rng) = (Pcg64Mcg::seed_from_u64(4), Pcg64Mcg::seed_from_u64(4));
    let mut costs = Vec::new();

    for _ in 0..10 {
        planner.run(400, &mut rng);
        kd.run(400, &mut kd_rng);
        assert_eq!(planner.best_path(), kd.best_path());
        costs.extend(planner.best_path().map(|path| path.cost));
    }

    let path = planner.best_path().expect("RRT*-k solves the slalom");
    assert_solves(&slalom(), &path);
    assert!(costs.windows(2).all(|pair| pair[1] <= pair[0]), "{costs:?}");
    assert!(costs.last() < costs.first(), "{costs:?}");
    assert!(
        costs.iter().all(|&cost| cost >= SLALOM_OPTIMUM),
        "{costs:?}"
    );
    assert_eq!(
        RrtStar::new(slalom(), RrtStarConfig::k_nearest(f64::NAN, 1.0)).err(),
        Some(Error::InvalidGamma)
    );
}

#[test]
fn unknown_regions_are_treated_as_invalid() {
    let fogged = || {
        problem(Boxes {
            obstacles: Vec::new(),
            fog: vec![([2.0, 2.0], [8.0, 8.0])],
        })
    };

    let path = rrt_star(fogged(), 1, 3000)
        .best_path()
        .expect("RRT* goes around the fog");
    assert_solves(&fogged(), &path);

    let mut rrt = Rrt::new(fogged(), RrtConfig::default()).expect("valid problem");
    let path = rrt
        .solve(20_000, &mut Pcg64Mcg::seed_from_u64(1))
        .expect("RRT goes around the fog");
    assert_solves(&fogged(), &path);
}

#[test]
fn invalid_or_unknown_start_is_rejected() {
    let mut blocked = slalom();
    blocked.start = [4.5, 1.0];
    let mut fogged = problem(Boxes {
        obstacles: Vec::new(),
        fog: vec![([0.0, 0.0], [2.0, 2.0])],
    });
    fogged.start = [1.0, 1.0];

    assert_eq!(
        Rrt::new(blocked, RrtConfig::default()).err(),
        Some(Error::InvalidStart(Validity::Invalid))
    );
    assert_eq!(
        RrtStar::new(fogged, star_config()).err(),
        Some(Error::InvalidStart(Validity::Unknown))
    );
}

#[test]
fn bad_config_is_rejected() {
    assert_eq!(
        Rrt::new(slalom(), RrtConfig { goal_bias: 1.5 }).err(),
        Some(Error::GoalBiasOutOfRange)
    );
    assert_eq!(
        RrtStar::new(slalom(), RrtStarConfig::new(0.0, 1.0)).err(),
        Some(Error::InvalidGamma)
    );
    assert_eq!(
        RrtStar::new(slalom(), RrtStarConfig::new(1.0, f64::NAN)).err(),
        Some(Error::InvalidMaxRadius)
    );
}

#[test]
fn shortcutting_an_rrt_path_keeps_it_valid_and_shortens_it() {
    let world = slalom();
    let mut rng = Pcg64Mcg::seed_from_u64(3);
    let mut planner = Rrt::new(slalom(), RrtConfig::default()).expect("valid problem");
    let path = planner
        .solve(20_000, &mut rng)
        .expect("RRT solves the slalom");

    let greedy = shortcut(&world.space, &world.checker, &path, f64::INFINITY);
    assert_solves_with(&world, &greedy, f64::INFINITY);
    assert!(greedy.cost < path.cost);

    let smoothed = shortcut_random(&world.space, &world.checker, &greedy, 2.0, 300, &mut rng);
    assert_solves_with(&world, &smoothed, f64::INFINITY);
    assert!(smoothed.cost <= greedy.cost);
    assert!(smoothed.cost >= SLALOM_OPTIMUM, "{}", smoothed.cost);
}

#[test]
fn rrt_connect_finds_valid_paths_reproducibly() {
    for seed in 0..5 {
        let solve = || {
            RrtConnect::new(slalom())
                .expect("valid problem")
                .solve(20_000, &mut Pcg64Mcg::seed_from_u64(seed))
                .expect("RRT-Connect solves the slalom")
        };
        let path = solve();

        assert_solves(&slalom(), &path);
        assert!(path.cost >= SLALOM_OPTIMUM);
        assert_eq!(path, solve());
    }
}

#[test]
fn rrt_connect_avoids_fog_and_rejects_bad_goals() {
    let fogged = || {
        problem(Boxes {
            obstacles: Vec::new(),
            fog: vec![([2.0, 2.0], [8.0, 8.0])],
        })
    };
    let path = RrtConnect::new(fogged())
        .expect("valid problem")
        .solve(20_000, &mut Pcg64Mcg::seed_from_u64(2))
        .expect("RRT-Connect goes around the fog");
    assert_solves(&fogged(), &path);

    let mut blocked = slalom();
    blocked.goal = sampling_planning::GoalRegion::new([4.5, 1.0], 0.5);
    assert_eq!(
        RrtConnect::new(blocked).err(),
        Some(Error::InvalidGoal(Validity::Invalid))
    );
}

#[test]
fn informed_rrt_star_converges_faster_on_an_open_world() {
    for seed in 0..4 {
        let plain = rrt_star(pole(), seed, 3000).best_path().expect("solved");
        let mut informed = RrtStar::new(pole(), star_config().informed()).expect("valid problem");
        informed.run(3000, &mut Pcg64Mcg::seed_from_u64(seed));
        let informed = informed.best_path().expect("solved");

        assert_solves(&pole(), &informed);
        assert!(informed.cost >= POLE_OPTIMUM);
        assert!(
            informed.cost < plain.cost,
            "seed {seed}: informed {} vs plain {}",
            informed.cost,
            plain.cost
        );
    }
}

/// A [`UniformBox`] without direct informed sampling, so informed RRT* falls back to rejection.
#[derive(Debug, Clone)]
struct RejectionOnly(UniformBox<2>);

impl Sampler<[f64; 2]> for RejectionOnly {
    fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> [f64; 2] {
        self.0.sample(rng)
    }
}

#[test]
fn direct_informed_sampling_beats_rejection_on_a_thin_informed_set() {
    let (mut direct_total, mut rejection_total) = (0.0, 0.0);

    for seed in 0..4 {
        let mut direct = RrtStar::new(pole(), star_config().informed()).expect("valid problem");
        direct.run(2000, &mut Pcg64Mcg::seed_from_u64(seed));
        let direct = direct.best_path().expect("solved");
        assert_solves(&pole(), &direct);
        assert!(direct.cost >= POLE_OPTIMUM);

        let world = pole();
        let rejection_world = Problem {
            space: world.space,
            sampler: RejectionOnly(world.sampler),
            checker: world.checker,
            steering: world.steering,
            start: world.start,
            goal: world.goal,
        };
        let mut rejection = RrtStar::new(rejection_world, star_config().informed()).expect("valid");
        rejection.run(2000, &mut Pcg64Mcg::seed_from_u64(seed));

        direct_total += direct.cost;
        rejection_total += rejection.best_path().expect("solved").cost;
    }

    assert!(
        direct_total < rejection_total,
        "direct {direct_total} vs rejection {rejection_total}"
    );
}

#[test]
fn informed_sampling_changes_nothing_when_the_informed_set_covers_the_box() {
    // Every slalom solution costs more than the largest start-to-goal detour inside the box,
    // so every sample is accepted at the first attempt and the RNG stream is unchanged.
    let mut informed = RrtStar::new(slalom(), star_config().informed()).expect("valid problem");
    informed.run(2000, &mut Pcg64Mcg::seed_from_u64(6));

    assert_eq!(
        informed.best_path(),
        rrt_star(slalom(), 6, 2000).best_path()
    );
}

#[test]
fn rrt_sharp_solves_improves_and_beats_rrt_star() {
    let (mut sharp_total, mut star_total) = (0.0, 0.0);

    for seed in 0..3 {
        let mut sharp = RrtSharp::new(slalom(), star_config()).expect("valid problem");
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let mut costs = Vec::new();

        for _ in 0..10 {
            sharp.run(200, &mut rng);
            costs.extend(sharp.best_path().map(|path| path.cost));
        }

        let path = sharp.best_path().expect("RRT# solves the slalom");
        assert_solves(&slalom(), &path);
        assert!(costs.windows(2).all(|pair| pair[1] <= pair[0]), "{costs:?}");
        assert!(
            costs.iter().all(|&cost| cost >= SLALOM_OPTIMUM),
            "{costs:?}"
        );

        sharp_total += path.cost;
        star_total += rrt_star(slalom(), seed, 2000)
            .best_path()
            .expect("solved")
            .cost;
    }

    assert!(
        sharp_total < star_total,
        "RRT# {sharp_total} vs RRT* {star_total}"
    );
}

#[test]
fn rrt_sharp_variants_match_across_indices_and_stay_valid() {
    for config in [star_config(), star_k_config(), star_config().informed()] {
        let mut linear = RrtSharp::new(pole(), config).expect("valid problem");
        let mut kd =
            RrtSharp::with_nearest_neighbors(pole(), config, KdTree::default()).expect("valid");
        linear.run(1500, &mut Pcg64Mcg::seed_from_u64(9));
        kd.run(1500, &mut Pcg64Mcg::seed_from_u64(9));

        let path = linear.best_path().expect("RRT# solves the pole world");
        assert_solves(&pole(), &path);
        assert!(path.cost >= POLE_OPTIMUM);
        assert_eq!(Some(path), kd.best_path());
    }
}

#[test]
fn reaching_the_goal_center_again_adds_no_duplicate_nodes() {
    let free = || problem(Boxes::default());
    let mut rng = Pcg64Mcg::seed_from_u64(1);
    // Every target is the goal center, about 11.3 steps away: once reached, steps stand still.
    let mut rrt = Rrt::new(free(), RrtConfig { goal_bias: 1.0 }).expect("valid problem");
    rrt.run(1000, &mut rng);
    assert!(rrt.node_count() < 20, "RRT: {} nodes", rrt.node_count());

    let config = RrtStarConfig {
        goal_bias: 1.0,
        ..star_config()
    };
    let mut star = RrtStar::new(free(), config).expect("valid problem");
    star.run(1000, &mut rng);
    assert!(star.node_count() < 20, "RRT*: {} nodes", star.node_count());

    let mut sharp = RrtSharp::new(free(), config).expect("valid problem");
    sharp.run(1000, &mut rng);
    assert!(
        sharp.node_count() < 20,
        "RRT#: {} nodes",
        sharp.node_count()
    );
}

/// The plane with a broken metric: every distance is NaN.
#[derive(Debug, Clone, Copy)]
struct NanPlane;

impl sampling_planning::StateSpace for NanPlane {
    type State = [f64; 2];

    fn dimension(&self) -> usize {
        2
    }

    fn distance(&self, _: &[f64; 2], _: &[f64; 2]) -> f64 {
        f64::NAN
    }

    fn interpolate(&self, from: &[f64; 2], to: &[f64; 2], t: f64) -> [f64; 2] {
        Euclidean::<2>.interpolate(from, to, t)
    }
}

/// Free space for [`NanPlane`].
struct FreePlane;

impl sampling_planning::ValidityChecker<NanPlane> for FreePlane {
    fn check_state(&self, _: &[f64; 2]) -> Validity {
        Validity::Valid
    }

    fn check_motion(&self, _: &NanPlane, _: &[f64; 2], _: &[f64; 2]) -> Validity {
        Validity::Valid
    }
}

#[test]
#[should_panic(expected = "StateSpace::distance returned NaN for every node")]
fn a_nan_metric_panics_with_a_message_that_names_it() {
    let world = problem(Boxes::default());
    let problem = Problem {
        space: NanPlane,
        sampler: world.sampler,
        checker: FreePlane,
        steering: world.steering,
        start: world.start,
        goal: world.goal,
    };
    let mut rrt = Rrt::new(problem, RrtConfig::default()).expect("valid problem");

    rrt.step(&mut Pcg64Mcg::seed_from_u64(1));
}
