//! Integration tests for the batch-informed planners.

use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    Euclidean, KdTree, LinearScan, NearestNeighbors, UniformBox,
    informed::{AitStar, AitStarConfig, BitStar, BitStarConfig, Error},
};

use crate::SLALOM_OPTIMUM;
use crate::world::{Boxes, POLE_OPTIMUM, World2d, assert_solves_with, pole, slalom};

/// A BIT* config whose radius constant is safe for the 10 x 10 world.
fn config() -> BitStarConfig {
    BitStarConfig::new(1.5 * BitStarConfig::gamma_lower_bound(2, 100.0))
}

/// BIT* on `world` with the given config and index.
fn bit_star<N>(
    world: World2d,
    config: BitStarConfig,
    nn: N,
) -> BitStar<Euclidean<2>, UniformBox<2>, Boxes, N>
where
    N: NearestNeighbors<Euclidean<2>> + Clone,
{
    BitStar::with_nearest_neighbors(
        world.space,
        world.sampler,
        world.checker,
        world.start,
        world.goal,
        config,
        nn,
    )
    .expect("valid problem")
}

#[test]
fn bit_star_solves_improves_and_stays_above_the_optimum() {
    for seed in 0..3 {
        let mut planner = bit_star(slalom(), config(), KdTree::default());
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let mut costs = Vec::new();

        for _ in 0..20 {
            planner.run(2000, &mut rng);
            costs.extend(planner.best_path().map(|path| path.cost));
        }

        let path = planner.best_path().expect("BIT* solves the slalom");
        assert_solves_with(&slalom(), &path, f64::INFINITY);
        assert!(
            costs.len() > 15,
            "seed {seed}: solved early enough: {costs:?}"
        );
        assert!(costs.windows(2).all(|pair| pair[1] <= pair[0]), "{costs:?}");
        assert!(costs.last() < costs.first(), "{costs:?}");
        assert!(
            costs.iter().all(|&cost| cost >= SLALOM_OPTIMUM),
            "{costs:?}"
        );
        assert!(
            path.cost < SLALOM_OPTIMUM * 1.05,
            "seed {seed}: {}",
            path.cost
        );
        assert!(planner.batch_count() > 1);
    }
}

#[test]
fn bit_star_matches_across_indices_and_supports_k_nearest() {
    let k_config = BitStarConfig {
        k_nearest: Some(1.5 * BitStarConfig::k_constant_lower_bound(2)),
        ..config()
    };

    for config in [config(), k_config] {
        let mut kd = bit_star(pole(), config, KdTree::default());
        let mut linear = bit_star(pole(), config, LinearScan::default());
        kd.run(3000, &mut Pcg64Mcg::seed_from_u64(5));
        linear.run(3000, &mut Pcg64Mcg::seed_from_u64(5));

        let path = kd.best_path().expect("BIT* solves the pole world");
        assert_solves_with(&pole(), &path, f64::INFINITY);
        assert!(
            path.cost >= POLE_OPTIMUM && path.cost < POLE_OPTIMUM * 1.05,
            "{}",
            path.cost
        );
        assert_eq!(Some(path), linear.best_path());
    }
}

#[test]
fn bit_star_respects_the_edge_cap_and_validates_its_config() {
    let capped = BitStarConfig {
        max_radius: 1.0,
        ..config()
    };
    let mut planner = bit_star(slalom(), capped, KdTree::default());
    planner.run(5000, &mut Pcg64Mcg::seed_from_u64(2));
    assert_solves_with(&slalom(), &planner.best_path().expect("solved"), 1.0);

    let world = slalom();
    let make = |config| {
        BitStar::new(
            world.space,
            world.sampler,
            world.checker.clone(),
            world.start,
            world.goal.clone(),
            config,
        )
        .err()
    };
    assert_eq!(
        make(BitStarConfig {
            batch_size: 0,
            ..config()
        }),
        Some(Error::EmptyBatch)
    );
    assert_eq!(
        make(BitStarConfig::new(f64::NAN)),
        Some(Error::InvalidRadius)
    );
    assert_eq!(
        make(BitStarConfig {
            max_radius: 0.0,
            ..config()
        }),
        Some(Error::InvalidRadius)
    );
}

/// AIT* on `world` with the given config and index.
fn ait_star<N>(
    world: World2d,
    config: AitStarConfig,
    nn: N,
) -> AitStar<Euclidean<2>, UniformBox<2>, Boxes, N>
where
    N: NearestNeighbors<Euclidean<2>> + Clone,
{
    AitStar::with_nearest_neighbors(
        world.space,
        world.sampler,
        world.checker,
        world.start,
        world.goal,
        config,
        nn,
    )
    .expect("valid problem")
}

#[test]
fn ait_star_solves_improves_and_stays_above_the_optimum() {
    for seed in 0..3 {
        let mut planner = ait_star(slalom(), config(), KdTree::default());
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let mut costs = Vec::new();

        for _ in 0..20 {
            planner.run(200, &mut rng);
            costs.extend(planner.best_path().map(|path| path.cost));
        }

        let path = planner.best_path().expect("AIT* solves the slalom");
        assert_solves_with(&slalom(), &path, f64::INFINITY);
        assert!(
            costs.len() > 5,
            "seed {seed}: solved early enough: {costs:?}"
        );
        assert!(costs.windows(2).all(|pair| pair[1] <= pair[0]), "{costs:?}");
        assert!(costs.last() < costs.first(), "{costs:?}");
        assert!(
            costs.iter().all(|&cost| cost >= SLALOM_OPTIMUM),
            "{costs:?}"
        );
        assert!(
            path.cost < SLALOM_OPTIMUM * 1.05,
            "seed {seed}: {}",
            path.cost
        );
        assert!(planner.batch_count() > 1);
    }
}

#[test]
fn ait_star_matches_across_indices_and_supports_k_nearest() {
    let k_config = AitStarConfig {
        k_nearest: Some(1.5 * AitStarConfig::k_constant_lower_bound(2)),
        ..config()
    };

    for (config, iterations) in [(config(), 3000), (k_config, 1500)] {
        let mut kd = ait_star(pole(), config, KdTree::default());
        let mut linear = ait_star(pole(), config, LinearScan::default());
        kd.run(iterations, &mut Pcg64Mcg::seed_from_u64(5));
        linear.run(iterations, &mut Pcg64Mcg::seed_from_u64(5));

        let path = kd.best_path().expect("AIT* solves the pole world");
        assert_solves_with(&pole(), &path, f64::INFINITY);
        assert!(
            path.cost >= POLE_OPTIMUM && path.cost < POLE_OPTIMUM * 1.05,
            "{}",
            path.cost
        );
        assert_eq!(Some(path), linear.best_path());
        assert_eq!(kd.vertex_count(), linear.vertex_count());
    }
}

#[test]
fn ait_star_respects_the_edge_cap_and_validates_its_config() {
    let capped = AitStarConfig {
        max_radius: 1.0,
        ..config()
    };
    let mut planner = ait_star(slalom(), capped, KdTree::default());
    planner.run(1000, &mut Pcg64Mcg::seed_from_u64(2));
    assert_solves_with(&slalom(), &planner.best_path().expect("solved"), 1.0);

    let world = slalom();
    let make = |config| {
        AitStar::new(
            world.space,
            world.sampler,
            world.checker.clone(),
            world.start,
            world.goal.clone(),
            config,
        )
        .err()
    };
    assert_eq!(
        make(AitStarConfig {
            batch_size: 0,
            ..config()
        }),
        Some(Error::EmptyBatch)
    );
    assert_eq!(
        make(AitStarConfig::new(f64::NAN)),
        Some(Error::InvalidRadius)
    );
    assert_eq!(
        make(AitStarConfig {
            k_nearest: Some(0.0),
            ..config()
        }),
        Some(Error::InvalidRadius)
    );
    assert_eq!(
        AitStar::new(
            world.space,
            world.sampler,
            world.checker.clone(),
            [4.5, 1.0],
            world.goal.clone(),
            config(),
        )
        .err(),
        Some(Error::InvalidStart(sampling_planning::Validity::Invalid))
    );
}
