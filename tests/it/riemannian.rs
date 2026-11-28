//! Integration tests for direct informed sampling under a configuration-dependent metric.

use core::num::NonZeroUsize;

use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    GoalRegion, LoewnerBound, Problem, Riemannian, Sampler, StateSpace, StraightLine, UniformBox,
    Validity, ValidityChecker, discretized_motion,
    trees::{RrtStar, RrtStarConfig},
};

/// A kinetic-energy-like metric over `[0, 10]^2` that is smallest at `(5, 5)` and grows
/// quadratically away from it, with coupled axes: `A + |q - c|^2 B`.
fn metric(q: &[f64; 2]) -> [[f64; 2]; 2] {
    let r = (q[0] - 5.0).powi(2) + (q[1] - 5.0).powi(2);

    [
        [4.0 + 0.05 * r, 1.0 + 0.01 * r],
        [1.0 + 0.01 * r, 1.0 + 0.03 * r],
    ]
}

/// A metric tensor field over the plane.
type MetricField = fn(&[f64; 2]) -> [[f64; 2]; 2];

/// The Riemannian space of [`metric`], four midpoint pieces per distance.
fn space() -> Riemannian<2, MetricField> {
    Riemannian::new(metric, NonZeroUsize::new(4).expect("non-zero"))
}

/// A wall at `4.5 <= x <= 5.5`, `y <= 7`, in the 10 x 10 box.
#[derive(Debug, Clone, Copy)]
struct Wall;

impl Wall {
    /// The validity of a point.
    fn check(q: &[f64; 2]) -> Validity {
        let inside = q.iter().all(|c| (0.0..=10.0).contains(c));
        let wall = (4.5..=5.5).contains(&q[0]) && q[1] <= 7.0;

        if inside && !wall {
            Validity::Valid
        } else {
            Validity::Invalid
        }
    }
}

impl<Sp: StateSpace<State = [f64; 2]>> ValidityChecker<Sp> for Wall {
    fn check_state(&self, q: &[f64; 2]) -> Validity {
        Self::check(q)
    }

    fn check_motion(&self, space: &Sp, from: &[f64; 2], to: &[f64; 2]) -> Validity {
        discretized_motion(space, from, to, 0.05, Self::check)
    }
}

/// The metric bound estimated from a grid of metric evaluations that includes the minimum.
fn grid_bound() -> LoewnerBound<2> {
    let grid = (0..=10).flat_map(|i| (0..=10).map(move |j| [f64::from(i), f64::from(j)]));

    LoewnerBound::from_metrics(grid.map(|q| metric(&q))).expect("positive definite metrics")
}

#[test]
fn the_grid_estimate_lies_below_the_metric_and_bounds_the_distance() {
    let bound = grid_bound();
    let mut sampler = UniformBox::new([0.0; 2], [10.0; 2]);
    let mut rng = Pcg64Mcg::seed_from_u64(1);

    for _ in 0..2000 {
        let (a, b) = (sampler.sample(&mut rng), sampler.sample(&mut rng));
        assert!(bound.is_below(&metric(&a), 1e-9), "{a:?}");
        assert!(bound.distance(&a, &b) <= space().distance(&a, &b) + 1e-9);
    }

    // Along the stiff axis it is much tighter than the scalar bound sqrt(lambda_min) |b - a|.
    let [[a, b], [_, d]] = bound.matrix();
    let lambda_min = 0.5 * (a + d - libm::sqrt((a - d) * (a - d) + 4.0 * b * b));
    assert!(bound.distance(&[0.0, 0.0], &[1.0, 0.0]) > 2.0 * libm::sqrt(lambda_min));
}

#[test]
fn bounded_informed_samples_are_in_the_true_informed_set() {
    let space = space();
    let mut sampler = UniformBox::new([0.0; 2], [10.0; 2]).with_metric_bound(grid_bound());
    let mut rng = Pcg64Mcg::seed_from_u64(2);
    // Near the metric's minimum the bound is tight, so the hyperspheroid is small.
    let (start, goal) = ([3.0, 5.0], [7.0, 5.5]);
    let max_cost = 1.15 * space.distance(&start, &goal);
    let mut drawn = 0;

    for _ in 0..500 {
        if let Some(q) = sampler.sample_informed(&space, &mut rng, &start, &goal, max_cost) {
            assert!(q.iter().all(|c| (0.0..=10.0).contains(c)));
            assert!(space.distance(&start, &q) + space.distance(&q, &goal) <= max_cost);
            drawn += 1;
        }
    }

    assert!(drawn > 400, "direct sampling rarely gives up: {drawn}");

    // Far from the minimum the bound is loose; once its set outgrows the box, it declines.
    let (far_start, far_goal) = ([0.5, 0.5], [9.5, 9.5]);
    let far_cost = 1.15 * space.distance(&far_start, &far_goal);
    assert!(
        sampler
            .sample_informed(&space, &mut rng, &far_start, &far_goal, far_cost)
            .is_none()
    );

    // Without a bound the non-Euclidean space falls back to rejection.
    let mut plain = UniformBox::new([0.0; 2], [10.0; 2]);
    assert!(
        plain
            .sample_informed(&space, &mut rng, &start, &goal, max_cost)
            .is_none()
    );
}

#[test]
fn informed_rrt_star_with_a_metric_bound_solves_and_improves() {
    let problem = Problem {
        space: space(),
        sampler: UniformBox::new([0.0; 2], [10.0; 2]).with_metric_bound(grid_bound()),
        checker: Wall,
        steering: StraightLine::new(1.0),
        start: [1.0, 2.0],
        goal: GoalRegion::new([9.0, 2.0], 0.5),
    };
    let config = RrtStarConfig::new(60.0, 3.0).informed();
    let mut planner = RrtStar::new(problem.clone(), config).expect("valid problem");
    let mut rng = Pcg64Mcg::seed_from_u64(3);
    let mut costs = Vec::new();

    for _ in 0..6 {
        planner.run(500, &mut rng);
        costs.extend(planner.best_path().map(|path| path.cost));
    }

    let path = planner.best_path().expect("informed RRT* solves the wall");
    let length: f64 = path
        .states
        .windows(2)
        .map(|pair| {
            assert_eq!(
                Wall.check_motion(&problem.space, &pair[0], &pair[1]),
                Validity::Valid
            );
            problem.space.distance(&pair[0], &pair[1])
        })
        .sum();

    assert!((length - path.cost).abs() < 1e-9);
    assert!(costs.windows(2).all(|pair| pair[1] <= pair[0]), "{costs:?}");
    assert!(costs.last() < costs.first(), "{costs:?}");
}
