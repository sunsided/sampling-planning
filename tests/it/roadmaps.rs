//! Integration tests for the roadmap planners.

use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    Euclidean, GoalRegion, KdTree, Path, UniformBox, Validity, ValidityChecker,
    roadmaps::{
        BfmtStar, BfmtStarConfig, Connection, FmtStar, FmtStarConfig, Prm, PrmConfig, Termination,
    },
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
    fmt_slalom_to(&slalom().goal, samples, seed)
}

/// FMT* on the slalom toward `goal`.
fn fmt_slalom_to(goal: &GoalRegion<[f64; 2]>, samples: usize, seed: u64) -> Option<Path<[f64; 2]>> {
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
        goal,
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

/// A k-nearest PRM* config with a constant safely above the bound.
fn star_k() -> PrmConfig {
    PrmConfig::star_k(2.0 * PrmConfig::k_constant_lower_bound(2))
}

#[test]
fn k_nearest_prm_star_solves_and_improves() {
    let small = slalom_query(&mut slalom_roadmap(star_k(), 300)).expect("connected");
    let large = slalom_query(&mut slalom_roadmap(star_k(), 4000)).expect("connected");

    assert_solves_with(&slalom(), &small, f64::INFINITY);
    assert_solves_with(&slalom(), &large, f64::INFINITY);
    assert!(large.cost < small.cost, "{} vs {}", large.cost, small.cost);
    assert!(large.cost >= SLALOM_OPTIMUM && large.cost < SLALOM_OPTIMUM * 1.05);
}

#[test]
fn k_nearest_prm_with_kd_tree_matches_linear_scan() {
    for config in [
        star_k(),
        PrmConfig {
            connection: Connection::KNearest(8),
            ..star()
        },
    ] {
        let world = slalom();
        let mut kd = Prm::with_nearest_neighbors(
            world.space,
            world.sampler,
            world.checker,
            config,
            KdTree::default(),
        )
        .expect("valid");
        kd.grow(1000, &mut Pcg64Mcg::seed_from_u64(8));

        let path = kd.query(&world.start, &world.goal).expect("valid start");
        let mut reference = slalom_roadmap(config, 1000);

        assert_eq!(kd.edge_count(), reference.edge_count());
        assert_eq!(path, slalom_query(&mut reference));
        assert_solves_with(&slalom(), &path.expect("connected"), f64::INFINITY);
    }
}

#[test]
fn zero_neighbors_are_rejected() {
    let world = slalom();
    let config = PrmConfig {
        connection: Connection::KNearest(0),
        ..star()
    };

    assert_eq!(
        Prm::new(world.space, world.sampler, world.checker, config).err(),
        Some(sampling_planning::roadmaps::Error::InvalidRadius)
    );
}

/// The optimal slalom cost to the goal center itself, rounded down: [`SLALOM_OPTIMUM`] plus the
/// goal tolerance. BFMT*'s goal tree ends every path through it at the center.
const SLALOM_CENTER_OPTIMUM: f64 = 19.155;

/// BFMT* on the slalom with `samples` samples, a fixed seed and the given termination.
fn bfmt_slalom(samples: usize, seed: u64, termination: Termination) -> Option<Path<[f64; 2]>> {
    bfmt_slalom_to(&slalom().goal, samples, seed, termination)
}

/// BFMT* on the slalom toward `goal`.
fn bfmt_slalom_to(
    goal: &GoalRegion<[f64; 2]>,
    samples: usize,
    seed: u64,
    termination: Termination,
) -> Option<Path<[f64; 2]>> {
    let world = slalom();
    let config = BfmtStarConfig {
        gamma: BfmtStarConfig::gamma_lower_bound(2, 100.0),
        termination,
    };
    let mut bfmt = BfmtStar::with_nearest_neighbors(
        world.space,
        world.sampler,
        world.checker,
        config,
        KdTree::default(),
    )
    .expect("valid");

    bfmt.plan(
        &world.start,
        goal,
        samples,
        &mut Pcg64Mcg::seed_from_u64(seed),
    )
    .expect("valid endpoints")
}

#[test]
fn bfmt_star_finds_valid_paths_and_improves_with_larger_batches() {
    for seed in 0..3 {
        let small = bfmt_slalom(300, seed, Termination::Optimal).expect("solved");
        let large = bfmt_slalom(5000, seed, Termination::Optimal).expect("solved");

        for path in [&small, &large] {
            assert_solves_with(&slalom(), path, f64::INFINITY);
            assert!(path.cost >= SLALOM_OPTIMUM);
        }

        assert!(
            large.cost < small.cost,
            "seed {seed}: {} vs {}",
            large.cost,
            small.cost
        );
        assert!(
            large.cost < SLALOM_CENTER_OPTIMUM * 1.05,
            "seed {seed}: {}",
            large.cost
        );
        assert_eq!(Some(large), bfmt_slalom(5000, seed, Termination::Optimal));
    }
}

#[test]
fn bfmt_star_matches_fmt_star_and_first_meetings_cost_more() {
    for seed in 0..3 {
        // A point goal, so that both planners have to end at the goal center.
        let center = GoalRegion::new(*slalom().goal.center(), 0.0);
        let optimal = bfmt_slalom_to(&center, 2000, seed, Termination::Optimal);
        let first = bfmt_slalom_to(&center, 2000, seed, Termination::FirstMeeting);
        let fmt = fmt_slalom_to(&center, 2000, seed).expect("solved");
        let (optimal, first) = (optimal.expect("solved"), first.expect("solved"));

        assert_solves_with(&slalom(), &first, f64::INFINITY);
        assert!(first.cost >= optimal.cost - 1e-9, "seed {seed}");
        assert!(
            (optimal.cost - fmt.cost).abs() < 0.02 * fmt.cost,
            "seed {seed}: BFMT* {} vs FMT* {}",
            optimal.cost,
            fmt.cost
        );
    }
}

/// A one-way gate at `x = 5` on the line: motions may cross it only while `x` increases.
struct OneWay;

impl ValidityChecker<Euclidean<1>> for OneWay {
    fn check_state(&self, _: &[f64; 1]) -> Validity {
        Validity::Valid
    }

    fn check_motion(&self, _: &Euclidean<1>, from: &[f64; 1], to: &[f64; 1]) -> Validity {
        if (from[0] < 5.0) != (to[0] < 5.0) && to[0] < from[0] {
            Validity::Invalid
        } else {
            Validity::Valid
        }
    }
}

#[test]
fn bfmt_star_checks_goal_tree_edges_toward_the_goal() {
    let plan = |start: f64, goal: f64| {
        let config = BfmtStarConfig::new(BfmtStarConfig::gamma_lower_bound(1, 10.0));
        let mut bfmt = BfmtStar::new(
            Euclidean::<1>,
            UniformBox::new([0.0], [10.0]),
            OneWay,
            config,
        )
        .expect("valid");

        bfmt.plan(
            &[start],
            &GoalRegion::new([goal], 0.05),
            500,
            &mut Pcg64Mcg::seed_from_u64(3),
        )
        .expect("valid endpoints")
    };

    let rightward = plan(1.0, 9.0).expect("crossing rightward is allowed");
    for pair in rightward.states.windows(2) {
        assert_eq!(
            OneWay.check_motion(&Euclidean::<1>, &pair[0], &pair[1]),
            Validity::Valid
        );
    }

    assert_eq!(plan(9.0, 1.0), None, "crossing leftward is not");
}

#[test]
fn bfmt_star_rejects_an_invalid_goal_center() {
    let world = slalom();
    let mut bfmt = BfmtStar::new(
        world.space,
        world.sampler,
        world.checker,
        BfmtStarConfig::new(5.0),
    )
    .expect("valid");

    assert_eq!(
        bfmt.plan(
            &world.start,
            &GoalRegion::new([4.5, 1.0], 0.5),
            100,
            &mut Pcg64Mcg::seed_from_u64(1)
        ),
        Err(sampling_planning::roadmaps::Error::InvalidGoal(
            Validity::Invalid
        ))
    );
}
