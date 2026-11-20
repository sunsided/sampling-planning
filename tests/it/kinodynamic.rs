//! Integration tests for SST and SST* with a velocity-controlled point robot.

use rand_core::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    Euclidean, KdTree, UniformBox, Validity, ValidityChecker,
    kinodynamic::{
        Error, KinodynamicProblem, Propagator, Sst, SstConfig, SstStarSchedule, Trajectory,
    },
    unit_f64,
};

use crate::world::{Boxes, POLE_OPTIMUM, pole};

/// Integration step of every test planner.
const DT: f64 = 0.1;

/// A point robot whose control is its velocity, at most 1 per axis.
#[derive(Debug, Clone, Copy)]
struct Velocity;

impl Propagator<Euclidean<2>> for Velocity {
    type Control = [f64; 2];

    fn control_dimension(&self) -> usize {
        2
    }

    fn sample_control<R: Rng + ?Sized>(&mut self, rng: &mut R) -> [f64; 2] {
        [2.0 * unit_f64(rng) - 1.0, 2.0 * unit_f64(rng) - 1.0]
    }

    fn propagate(&self, _: &Euclidean<2>, from: &[f64; 2], u: &[f64; 2], dt: f64) -> [f64; 2] {
        [from[0] + u[0] * dt, from[1] + u[1] * dt]
    }
}

/// The problem type of these tests.
type World = KinodynamicProblem<Euclidean<2>, UniformBox<2>, Boxes, Velocity>;

/// The [`pole`] world with the velocity robot.
fn world() -> World {
    let world = pole();

    KinodynamicProblem {
        space: world.space,
        sampler: world.sampler,
        checker: world.checker,
        propagator: Velocity,
        start: world.start,
        goal: world.goal,
    }
}

/// An SST config tuned for the pole world.
fn config() -> SstConfig {
    SstConfig::new(0.8, 0.25, DT, 10)
}

/// Asserts that `trajectory` solves `world`: replaying its controls step by step from the start
/// reproduces its states, every step checks as valid, it ends in the goal, and its cost is its
/// total duration.
fn assert_solves(world: &World, trajectory: &Trajectory<[f64; 2], [f64; 2]>) {
    let mut state = world.start;
    assert_eq!(trajectory.states.first(), Some(&state));
    assert_eq!(trajectory.controls.len() + 1, trajectory.states.len());

    for ((control, duration), expected) in trajectory.controls.iter().zip(&trajectory.states[1..]) {
        let steps: Vec<u32> = (1..=1000)
            .take_while(|&k| f64::from(k) * DT <= duration + 1e-9)
            .collect();
        let last = steps.last().copied().map_or(0.0, f64::from);
        assert!(
            (last * DT - duration).abs() < 1e-9,
            "duration {duration} is whole steps"
        );

        for _ in steps {
            let next = Velocity.propagate(&world.space, &state, control, DT);
            let validity = world.checker.check_motion(&world.space, &state, &next);
            assert_eq!(validity, Validity::Valid, "{state:?} -> {next:?}");
            state = next;
        }

        assert_eq!(&state, expected);
    }

    assert!(world.goal.contains(&world.space, &state));
    let total: f64 = trajectory.controls.iter().map(|(_, d)| d).sum();
    assert!((total - trajectory.cost).abs() < 1e-9);
}

/// The shortest possible duration: the pole optimum at the top diagonal speed `sqrt(2)`.
fn time_lower_bound() -> f64 {
    POLE_OPTIMUM / core::f64::consts::SQRT_2
}

#[test]
fn sst_solves_and_stays_sparse() {
    let mut sst = Sst::new(world(), config()).expect("valid");
    sst.run(20_000, &mut Pcg64Mcg::seed_from_u64(1));

    let trajectory = sst.best_trajectory().expect("SST solves the pole world");
    assert_solves(&world(), &trajectory);
    assert!(trajectory.cost >= time_lower_bound());
    assert!(
        sst.active_count() * 2 < sst.node_count(),
        "{} active of {}",
        sst.active_count(),
        sst.node_count()
    );
}

#[test]
fn sst_cost_never_increases_and_improves() {
    let mut sst = Sst::with_nearest_neighbors(world(), config(), KdTree::default()).expect("valid");
    let mut rng = Pcg64Mcg::seed_from_u64(2);
    let mut costs = Vec::new();

    for _ in 0..20 {
        sst.run(2000, &mut rng);
        costs.extend(sst.best_trajectory().map(|t| t.cost));
    }

    assert!(costs.len() > 10, "found a solution early enough: {costs:?}");
    assert!(costs.windows(2).all(|pair| pair[1] <= pair[0]), "{costs:?}");
    assert!(costs.last() < costs.first(), "{costs:?}");
}

#[test]
fn sst_star_shrinks_its_radii_and_gets_close_to_optimal() {
    let schedule = SstStarSchedule {
        shrink: 0.8,
        first_epoch: 2000,
    };
    let mut star = Sst::with_nearest_neighbors(world(), config().star(schedule), KdTree::default())
        .expect("valid");
    star.run(40_000, &mut Pcg64Mcg::seed_from_u64(3));

    let (selection, pruning) = star.radii();
    assert!(selection < config().selection_radius && pruning < config().pruning_radius);

    let trajectory = star.best_trajectory().expect("SST* solves the pole world");
    assert_solves(&world(), &trajectory);
    assert!(trajectory.cost >= time_lower_bound());
    assert!(
        trajectory.cost < 1.5 * time_lower_bound(),
        "{}",
        trajectory.cost
    );
}

#[test]
fn sst_with_kd_tree_matches_linear_scan() {
    let mut kd = Sst::with_nearest_neighbors(world(), config(), KdTree::default()).expect("valid");
    let mut linear = Sst::new(world(), config()).expect("valid");
    kd.run(5000, &mut Pcg64Mcg::seed_from_u64(6));
    linear.run(5000, &mut Pcg64Mcg::seed_from_u64(6));

    assert_eq!(kd.active_count(), linear.active_count());
    assert_eq!(kd.best_trajectory(), linear.best_trajectory());
}

#[test]
fn sst_is_reproducible_and_validates_its_inputs() {
    let run = || {
        let mut sst = Sst::new(world(), config()).expect("valid");
        sst.run(3000, &mut Pcg64Mcg::seed_from_u64(4));
        sst.best_trajectory()
    };
    assert_eq!(run(), run());

    let mut blocked = world();
    blocked.start = [5.0, 5.0];
    assert_eq!(
        Sst::new(blocked, config()).err(),
        Some(Error::InvalidStart(Validity::Invalid))
    );
    assert_eq!(
        Sst::new(world(), SstConfig::new(0.0, 0.2, DT, 10)).err(),
        Some(Error::InvalidRadius)
    );
    assert_eq!(
        Sst::new(world(), SstConfig::new(1.0, 0.2, DT, 0)).err(),
        Some(Error::InvalidPropagation)
    );
    let bad_schedule = SstStarSchedule {
        shrink: 1.0,
        first_epoch: 10,
    };
    assert_eq!(
        Sst::new(world(), config().star(bad_schedule)).err(),
        Some(Error::InvalidSchedule)
    );
}
