//! Integration tests for the roadmap planners.

use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    Euclidean, GoalRegion, KdTree, Path, UniformBox, Validity, ValidityChecker,
    roadmaps::{FmtStar, FmtStarConfig, Prm, PrmConfig},
};

use crate::SLALOM_OPTIMUM;
use crate::world::{Boxes, assert_solves_with, slalom};

/// A PRM* config whose radius constant is safe for the 10 x 10 world.
fn star() -> PrmConfig {
    PrmConfig::star(1.5 * PrmConfig::gamma_lower_bound(2, 100.0))
}

/// A slalom roadmap with the given config, grown by `samples` samples with a fixed seed.
fn slalom_roadmap(config: PrmConfig, samples: usize) -> Prm<Euclidean<2>, UniformBox<2>, Boxes> {
    let world = slalom();
    let mut prm = Prm::new(world.space, world.sampler, world.checker, config).expect("valid");
    prm.grow(samples, &mut Pcg64Mcg::seed_from_u64(8));

    prm
}

/// Solves the slalom query on `prm`.
fn slalom_query(prm: &mut Prm<Euclidean<2>, UniformBox<2>, Boxes>) -> Option<Path<[f64; 2]>> {
    let world = slalom();

    prm.query(&world.start, &world.goal).expect("valid start")
}

#[test]
fn prm_star_answers_several_queries_on_one_roadmap() {
    let mut prm = slalom_roadmap(star(), 1500);
    let nodes = prm.node_count();

    let first = slalom_query(&mut prm).expect("connected");
    assert_solves_with(&slalom(), &first, f64::INFINITY);
    assert!(first.cost >= SLALOM_OPTIMUM);

    let other_goal = GoalRegion::new([9.0, 1.0], 0.5);
    let second = prm
        .query(&[1.0, 9.0], &other_goal)
        .expect("valid start")
        .expect("connected");
    assert_eq!(second.states.first(), Some(&[1.0, 9.0]));
    assert!(other_goal.contains(&Euclidean::<2>, second.states.last().expect("non-empty")));
    for pair in second.states.windows(2) {
        let validity = slalom()
            .checker
            .check_motion(&Euclidean::<2>, &pair[0], &pair[1]);
        assert_eq!(validity, Validity::Valid);
    }

    assert_eq!(
        slalom_query(&mut prm),
        Some(first),
        "queries are repeatable"
    );
    assert_eq!(prm.node_count(), nodes, "queries do not grow the roadmap");
}

#[test]
fn prm_star_improves_as_the_roadmap_grows() {
    let small = slalom_query(&mut slalom_roadmap(star(), 300)).expect("connected");
    let large = slalom_query(&mut slalom_roadmap(star(), 4000)).expect("connected");

    assert!(large.cost < small.cost, "{} vs {}", large.cost, small.cost);
    assert!(large.cost >= SLALOM_OPTIMUM && large.cost < SLALOM_OPTIMUM * 1.05);
}

#[test]
fn lazy_prm_matches_eager_prm() {
    let eager = slalom_query(&mut slalom_roadmap(star(), 1000)).expect("connected");
    let lazy = slalom_query(&mut slalom_roadmap(star().lazy(), 1000)).expect("connected");

    assert!((eager.cost - lazy.cost).abs() < 1e-9);
    assert_solves_with(&slalom(), &lazy, f64::INFINITY);
}

#[test]
fn fixed_radius_prm_with_kd_tree_matches_linear_scan() {
    let world = slalom();
    let mut kd = Prm::with_nearest_neighbors(
        world.space,
        world.sampler,
        world.checker,
        PrmConfig::fixed(1.0),
        KdTree::default(),
    )
    .expect("valid");
    kd.grow(1000, &mut Pcg64Mcg::seed_from_u64(8));

    let path = kd
        .query(&slalom().start, &slalom().goal)
        .expect("valid start");
    let reference = slalom_query(&mut slalom_roadmap(PrmConfig::fixed(1.0), 1000));

    assert_eq!(path, reference);
    assert_solves_with(&slalom(), &path.expect("connected"), 1.0);
}

/// FMT* on the slalom with `samples` samples and a fixed seed.
fn fmt_slalom(samples: usize, seed: u64) -> Option<Path<[f64; 2]>> {
    let world = slalom();
    let config = FmtStarConfig {
        gamma: 1.5 * FmtStarConfig::gamma_lower_bound(2, 100.0),
    };
    let mut fmt = FmtStar::with_nearest_neighbors(
        world.space,
        world.sampler,
        world.checker,
        config,
        KdTree::default(),
    )
    .expect("valid");

    fmt.plan(
        &world.start,
        &world.goal,
        samples,
        &mut Pcg64Mcg::seed_from_u64(seed),
    )
    .expect("valid start")
}

#[test]
fn fmt_star_finds_valid_paths_reproducibly() {
    for seed in 0..3 {
        let path = fmt_slalom(1500, seed).expect("FMT* solves the slalom");

        assert_solves_with(&slalom(), &path, f64::INFINITY);
        assert!(path.cost >= SLALOM_OPTIMUM);
        assert_eq!(Some(path), fmt_slalom(1500, seed));
    }
}

#[test]
fn fmt_star_improves_with_larger_batches() {
    for seed in 0..3 {
        let small = fmt_slalom(300, seed).expect("solved").cost;
        let large = fmt_slalom(5000, seed).expect("solved").cost;

        assert!(large < small, "seed {seed}: {large} vs {small}");
        assert!(large < SLALOM_OPTIMUM * 1.05, "seed {seed}: {large}");
    }
}

#[test]
fn fmt_star_reports_disconnected_batches() {
    let mut world = slalom();
    world.checker.obstacles.push(([0.0, 5.0], [10.0, 5.5]));
    let mut fmt = FmtStar::new(
        world.space,
        world.sampler,
        world.checker,
        FmtStarConfig { gamma: 5.0 },
    )
    .expect("valid");

    let path = fmt.plan(
        &world.start,
        &world.goal,
        500,
        &mut Pcg64Mcg::seed_from_u64(1),
    );
    assert_eq!(path, Ok(None));
}
