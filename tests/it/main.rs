//! Integration tests for the tree planners on a 2D world with obstacles.

mod world;

use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    Euclidean, KdTree, StraightLine, UniformBox, Validity,
    trees::{Error, Rrt, RrtConfig, RrtStar, RrtStarConfig},
};
use world::{Boxes, World2d, assert_solves, problem, slalom};

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
