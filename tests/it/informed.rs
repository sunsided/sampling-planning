//! Integration tests for the batch-informed planners.

use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    Euclidean, KdTree, LinearScan, NearestNeighbors, UniformBox,
    informed::{
        AbitStar, AbitStarConfig, AitStar, AitStarConfig, BitStar, BitStarConfig, EitStar,
        EitStarConfig, Error,
    },
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

        for _ in 0..12 {
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

    for (config, iterations) in [(config(), 1200), (k_config, 1500)] {
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

/// ABIT* on `world` with OMPL's default factors over a k-d tree.
fn abit_star(world: World2d) -> AbitStar<Euclidean<2>, UniformBox<2>, Boxes, KdTree<2>> {
    AbitStar::with_nearest_neighbors(
        world.space,
        world.sampler,
        world.checker,
        world.start,
        world.goal,
        AbitStarConfig::from_bit_star(config()),
        KdTree::default(),
    )
    .expect("valid problem")
}

/// The number of steps until `step` first reports a solution, at most `budget`.
fn steps_to_solution(mut step: impl FnMut() -> bool, budget: usize) -> usize {
    (1..=budget).find(|_| step()).unwrap_or(budget)
}

#[test]
fn abit_star_solves_improves_and_stays_above_the_optimum() {
    for seed in 0..3 {
        let mut planner = abit_star(slalom());
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let mut costs = Vec::new();

        // Cascading rewirings pop many cheap rewiring edges, so ABIT* needs more steps than
        // BIT* for the same number of motion checks.
        for _ in 0..20 {
            planner.run(4000, &mut rng);
            costs.extend(planner.best_path().map(|path| path.cost));
        }

        let path = planner.best_path().expect("ABIT* solves the slalom");
        assert_solves_with(&slalom(), &path, f64::INFINITY);
        assert!(costs.len() > 15, "seed {seed}: {costs:?}");
        assert!(costs.windows(2).all(|pair| pair[1] <= pair[0]), "{costs:?}");
        assert!(
            costs.iter().all(|&cost| cost >= SLALOM_OPTIMUM),
            "{costs:?}"
        );
        assert!(
            path.cost < SLALOM_OPTIMUM * 1.05,
            "seed {seed}: {}",
            path.cost
        );
        assert!(planner.inflation_factor() < 2.0 && planner.truncation_factor() < 2.0);
    }
}

#[test]
fn abit_star_finds_a_first_solution_in_fewer_steps_than_bit_star() {
    let (mut abit_steps, mut bit_steps) = (0, 0);

    for seed in 0..5 {
        let mut abit = abit_star(slalom());
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        abit_steps += steps_to_solution(
            || {
                abit.step(&mut rng);
                abit.best_path().is_some()
            },
            50_000,
        );

        let mut bit = bit_star(slalom(), config(), KdTree::default());
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        bit_steps += steps_to_solution(
            || {
                bit.step(&mut rng);
                bit.best_path().is_some()
            },
            50_000,
        );
    }

    assert!(
        abit_steps < bit_steps,
        "ABIT* {abit_steps} vs BIT* {bit_steps}"
    );
}

#[test]
#[should_panic(expected = "initial inflation")]
fn abit_star_rejects_deflating_factors() {
    let world = slalom();
    let config = AbitStarConfig {
        initial_inflation: 0.5,
        ..AbitStarConfig::new(1.0)
    };
    let _ = AbitStar::new(
        world.space,
        world.sampler,
        world.checker,
        world.start,
        world.goal,
        config,
    );
}

/// Wraps [`Boxes`] and counts every state it evaluates: single state checks and the states
/// of discretized motion checks, at the world's resolution.
#[derive(Debug, Clone)]
struct Counting {
    /// The world.
    world: Boxes,
    /// The number of states evaluated.
    states: std::rc::Rc<std::cell::Cell<usize>>,
}

impl sampling_planning::ValidityChecker<Euclidean<2>> for Counting {
    fn check_state(&self, state: &[f64; 2]) -> sampling_planning::Validity {
        self.states.set(self.states.get() + 1);
        self.world.check_state(state)
    }

    fn check_motion(
        &self,
        space: &Euclidean<2>,
        from: &[f64; 2],
        to: &[f64; 2],
    ) -> sampling_planning::Validity {
        sampling_planning::discretized_motion(space, from, to, 0.02, |state| {
            self.check_state(state)
        })
    }
}

/// The EIT* config for the 10 x 10 world, at the world's check resolution.
fn eit_config() -> EitStarConfig {
    EitStarConfig::from_bit_star(config(), 0.02)
}

/// EIT* on `world` over a k-d tree.
fn eit_star(world: World2d) -> EitStar<Euclidean<2>, UniformBox<2>, Boxes, KdTree<2>> {
    EitStar::with_nearest_neighbors(
        world.space,
        world.sampler,
        world.checker,
        world.start,
        world.goal,
        eit_config(),
        KdTree::default(),
    )
    .expect("valid problem")
}

#[test]
fn eit_star_solves_improves_and_stays_above_the_optimum() {
    for seed in 0..3 {
        let mut planner = eit_star(slalom());
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let mut costs = Vec::new();

        // Every reverse-search edge is a step of its own, so budget EIT* by batches.
        for round in 1..=12 {
            while planner.batch_count() < 3 * round {
                planner.step(&mut rng);
            }

            costs.extend(planner.best_path().map(|path| path.cost));
        }

        let path = planner.best_path().expect("EIT* solves the slalom");
        assert_solves_with(&slalom(), &path, f64::INFINITY);
        assert_eq!(costs.len(), 12, "seed {seed}: solved in the first batches");
        assert!(costs.windows(2).all(|pair| pair[1] <= pair[0]), "{costs:?}");
        assert!(
            costs.iter().all(|&cost| cost >= SLALOM_OPTIMUM),
            "{costs:?}"
        );
        assert!(
            path.cost < SLALOM_OPTIMUM * 1.05,
            "seed {seed}: {}",
            path.cost
        );
    }
}

/// The states evaluated until `step` first reports a solution.
fn checks_to_solution(states: &std::cell::Cell<usize>, mut step: impl FnMut() -> bool) -> usize {
    for _ in 0..200_000 {
        if step() {
            return states.get();
        }
    }

    panic!("no solution within the budget");
}

#[test]
fn eit_star_evaluates_fewer_states_to_a_first_solution_than_ait_star() {
    let (mut eit_checks, mut ait_checks) = (0, 0);

    for seed in 0..5 {
        let world = slalom();
        let counter = std::rc::Rc::new(std::cell::Cell::new(0));
        let checker = Counting {
            world: world.checker.clone(),
            states: counter.clone(),
        };
        let mut eit = EitStar::new(
            world.space,
            world.sampler,
            checker.clone(),
            world.start,
            world.goal.clone(),
            eit_config(),
        )
        .expect("valid problem");
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        eit_checks += checks_to_solution(&counter, || {
            eit.step(&mut rng);
            eit.best_path().is_some()
        });

        counter.set(0);
        let mut ait = AitStar::new(
            world.space,
            world.sampler,
            checker,
            world.start,
            world.goal,
            config(),
        )
        .expect("valid problem");
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        ait_checks += checks_to_solution(&counter, || {
            ait.step(&mut rng);
            ait.best_path().is_some()
        });
    }

    eprintln!("first solution: EIT* {eit_checks} vs AIT* {ait_checks} state evaluations");
    assert!(
        eit_checks < ait_checks,
        "EIT* {eit_checks} vs AIT* {ait_checks}"
    );
}

#[test]
#[should_panic(expected = "suboptimality")]
fn eit_star_rejects_a_suboptimality_below_one() {
    let world = slalom();
    let config = EitStarConfig {
        suboptimality: 0.5,
        ..eit_config()
    };
    let _ = EitStar::new(
        world.space,
        world.sampler,
        world.checker,
        world.start,
        world.goal,
        config,
    );
}

/// The plane with [`Euclidean`]'s metric that does not claim to be symmetric, so AIT* and
/// EIT* build their batch graph eagerly and symmetrized instead of lazily.
#[derive(Debug, Clone, Copy)]
struct UnclaimedPlane;

impl sampling_planning::StateSpace for UnclaimedPlane {
    type State = [f64; 2];

    fn dimension(&self) -> usize {
        2
    }

    fn distance(&self, a: &[f64; 2], b: &[f64; 2]) -> f64 {
        Euclidean::<2>.distance(a, b)
    }

    fn interpolate(&self, a: &[f64; 2], b: &[f64; 2], t: f64) -> [f64; 2] {
        Euclidean::<2>.interpolate(a, b, t)
    }

    fn is_euclidean(&self) -> bool {
        true
    }
}

/// [`Boxes`] over [`UnclaimedPlane`].
#[derive(Debug, Clone)]
struct UnclaimedBoxes(Boxes);

impl sampling_planning::ValidityChecker<UnclaimedPlane> for UnclaimedBoxes {
    fn check_state(&self, state: &[f64; 2]) -> sampling_planning::Validity {
        self.0.check_state(state)
    }

    fn check_motion(
        &self,
        _: &UnclaimedPlane,
        from: &[f64; 2],
        to: &[f64; 2],
    ) -> sampling_planning::Validity {
        self.0.check_motion(&Euclidean::<2>, from, to)
    }
}

#[test]
fn eager_batch_graphs_of_unsymmetric_spaces_equal_the_lazy_ones() {
    use sampling_planning::StateSpace;

    let world = slalom();
    assert!(Euclidean::<2>.is_symmetric() && !UnclaimedPlane.is_symmetric());
    let unclaimed = UnclaimedBoxes(world.checker.clone());

    for seed in 0..3 {
        let rng = || Pcg64Mcg::seed_from_u64(seed);
        let (start, goal) = (world.start, world.goal.clone());

        let mut lazy = AitStar::new(
            world.space,
            world.sampler,
            world.checker.clone(),
            start,
            goal.clone(),
            config(),
        )
        .expect("valid problem");
        let mut eager = AitStar::new(
            UnclaimedPlane,
            world.sampler,
            unclaimed.clone(),
            start,
            goal.clone(),
            config(),
        )
        .expect("valid problem");
        lazy.run(1500, &mut rng());
        eager.run(1500, &mut rng());
        assert!(lazy.best_path().is_some());
        assert_eq!(lazy.best_path(), eager.best_path(), "AIT*, seed {seed}");

        let mut lazy = EitStar::new(
            world.space,
            world.sampler,
            world.checker.clone(),
            start,
            goal.clone(),
            eit_config(),
        )
        .expect("valid problem");
        let mut eager = EitStar::new(
            UnclaimedPlane,
            world.sampler,
            unclaimed.clone(),
            start,
            goal,
            eit_config(),
        )
        .expect("valid problem");
        lazy.run(20_000, &mut rng());
        eager.run(20_000, &mut rng());
        assert!(lazy.best_path().is_some());
        assert_eq!(lazy.best_path(), eager.best_path(), "EIT*, seed {seed}");
    }
}
