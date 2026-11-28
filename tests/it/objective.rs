//! Integration tests for RRT* under a cost objective other than path length.

use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    CostObjective, Euclidean, GaussianObstacle, KdTree, Path, RiskDensity, Validity,
    ValidityChecker,
    trees::{RrtStar, RrtStarConfig},
};

use crate::world::{Boxes, World2d, pole, problem};

/// Asserts that `path` leads from the start of `world` into its goal along valid edges; its
/// cost is not a length here, so it is not compared with one.
fn assert_feasible(world: &World2d, path: &Path<[f64; 2]>) {
    assert_eq!(path.states.first(), Some(&world.start));
    let last = path.states.last().expect("non-empty");
    assert!(world.goal.contains(&world.space, last));

    for pair in path.states.windows(2) {
        let validity = world.checker.check_motion(&world.space, &pair[0], &pair[1]);
        assert_eq!(validity, Validity::Valid, "{pair:?}");
    }
}

/// An uncertain obstacle on the straight line of the open 10 x 10 world, `(1, 1)` to `(9, 9)`.
fn risk() -> RiskDensity<2> {
    let obstacle = GaussianObstacle::new([5.0, 5.0], [[0.6, 0.0], [0.0, 0.6]], 2.0)
        .expect("positive definite");

    RiskDensity::new(vec![obstacle], 0.01)
}

/// The risk along a path's edges.
fn path_risk(objective: &RiskDensity<2>, states: &[[f64; 2]]) -> f64 {
    states
        .windows(2)
        .map(|pair| objective.risk(&pair[0], &pair[1]))
        .sum()
}

#[test]
fn risk_aware_rrt_star_avoids_the_uncertain_obstacle() {
    let world = problem(Boxes::default());
    let config = RrtStarConfig::new(1.5 * RrtStarConfig::gamma_lower_bound(2, 100.0), 1.0);
    let objective = risk();

    for seed in 0..3 {
        let mut aware =
            RrtStar::with_objective(world.clone(), config, KdTree::default(), objective.clone())
                .expect("valid problem");
        let mut plain = RrtStar::new(world.clone(), config).expect("valid problem");
        aware.run(3000, &mut Pcg64Mcg::seed_from_u64(seed));
        plain.run(3000, &mut Pcg64Mcg::seed_from_u64(seed));

        let safe = aware
            .best_path()
            .expect("risk-aware RRT* solves the open world");
        let short = plain.best_path().expect("RRT* solves the open world");
        assert_feasible(&world, &safe);

        // The tree cost is the objective's cost of the returned path.
        let cost: f64 = safe
            .states
            .windows(2)
            .map(|pair| objective.motion_cost(&Euclidean::<2>, &pair[0], &pair[1]))
            .sum();
        assert!((cost - safe.cost).abs() < 1e-9);

        let (safe_risk, short_risk) = (
            path_risk(&objective, &safe.states),
            path_risk(&objective, &short.states),
        );
        assert!(safe_risk < 0.25 * short_risk, "{safe_risk} vs {short_risk}");
    }
}

#[test]
fn informed_risk_aware_rrt_star_improves_monotonically() {
    let config =
        RrtStarConfig::new(1.5 * RrtStarConfig::gamma_lower_bound(2, 100.0), 1.0).informed();
    let mut planner =
        RrtStar::with_objective(pole(), config, KdTree::default(), risk()).expect("valid problem");
    let mut rng = Pcg64Mcg::seed_from_u64(4);
    let mut costs = Vec::new();

    for _ in 0..6 {
        planner.run(1000, &mut rng);
        costs.extend(planner.best_path().map(|path| path.cost));
    }

    assert_feasible(
        &pole(),
        &planner.best_path().expect("solves the pole world"),
    );
    assert!(costs.windows(2).all(|pair| pair[1] <= pair[0]), "{costs:?}");
}
