//! Integration tests for retiring exhausted vertices from the tree planners' indices.

use core::num::NonZeroUsize;

use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    Euclidean, GoalRegion, KdTree, Problem, StateSpace, Steering, UniformBox,
    trees::{Rrt, RrtConfig, RrtConnect, RrtStar},
};

use crate::star_config;
use crate::world::{Boxes, assert_solves, slalom};

/// The retirement cap used by the tests: three consecutive failures.
fn cap() -> NonZeroUsize {
    NonZeroUsize::new(3).expect("non-zero")
}

#[test]
fn rrt_with_retirement_still_solves_and_retires_wall_vertices() {
    let mut retired = 0;

    for seed in 0..5 {
        let planner =
            Rrt::with_nearest_neighbors(slalom(), RrtConfig::default(), KdTree::default())
                .expect("valid problem");
        let mut planner = planner.with_retirement(cap());
        let path = planner
            .solve(20_000, &mut Pcg64Mcg::seed_from_u64(seed))
            .expect("RRT with retirement solves the slalom");

        assert_solves(&slalom(), &path);
        retired += planner.retired_count();
    }

    assert!(retired > 0, "vertices against the walls get retired");
}

#[test]
fn retirement_is_off_by_default_and_leaves_runs_unchanged() {
    let solve = |retire: bool| {
        let mut planner = Rrt::new(slalom(), RrtConfig::default()).expect("valid problem");
        if retire {
            // A cap no vertex reaches in this run behaves exactly like no retirement.
            planner = planner.with_retirement(NonZeroUsize::MAX);
        }

        let path = planner.solve(20_000, &mut Pcg64Mcg::seed_from_u64(9));
        (path, planner.retired_count())
    };

    let (plain, none) = solve(false);
    let (capped, also_none) = solve(true);

    assert_eq!((none, also_none), (0, 0));
    assert_eq!(plain, capped);
}

#[test]
fn rrt_star_and_rrt_connect_retire_and_keep_valid_paths() {
    let mut star = RrtStar::new(slalom(), star_config())
        .expect("valid problem")
        .with_retirement(cap());
    star.run(3000, &mut Pcg64Mcg::seed_from_u64(1));
    assert_solves(
        &slalom(),
        &star.best_path().expect("RRT* solves the slalom"),
    );
    assert!(star.retired_count() > 0);

    let mut connect = RrtConnect::new(slalom())
        .expect("valid problem")
        .with_retirement(cap());
    let path = connect
        .solve(20_000, &mut Pcg64Mcg::seed_from_u64(2))
        .expect("RRT-Connect solves the slalom");
    assert_solves(&slalom(), &path);
}

/// Unit moves on the integer lattice of `[0, 10]^2`: one step along the axis that brings the
/// state closest to the target; finitely many successors, the in-bounds lattice neighbors.
#[derive(Debug, Clone, Copy)]
struct Lattice;

impl Lattice {
    /// The in-bounds lattice neighbors of `from`.
    fn moves(from: &[f64; 2]) -> Vec<[f64; 2]> {
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]]
            .iter()
            .map(|d| [from[0] + d[0], from[1] + d[1]])
            .filter(|q| q.iter().all(|c| (0.0..=10.0).contains(c)))
            .collect()
    }
}

impl Steering<Euclidean<2>> for Lattice {
    fn steer(&self, space: &Euclidean<2>, from: &[f64; 2], toward: &[f64; 2]) -> [f64; 2] {
        Self::moves(from)
            .into_iter()
            .min_by(|a, b| {
                space
                    .distance(a, toward)
                    .total_cmp(&space.distance(b, toward))
            })
            .unwrap_or(*from)
    }

    fn successors(&self, _: &Euclidean<2>, from: &[f64; 2]) -> Option<Vec<[f64; 2]>> {
        Some(Self::moves(from))
    }
}

#[test]
fn finite_steering_retires_vertices_whose_successors_are_all_in_the_tree() {
    let problem = Problem {
        space: Euclidean::<2>,
        sampler: UniformBox::new([0.0; 2], [10.0; 2]),
        checker: Boxes {
            obstacles: vec![([4.5, -1.0], [5.5, 7.5])],
            fog: Vec::new(),
        },
        steering: Lattice,
        start: [1.0, 1.0],
        goal: GoalRegion::new([9.0, 1.0], 0.1),
    };
    let planner =
        Rrt::with_nearest_neighbors(problem.clone(), RrtConfig::default(), KdTree::default())
            .expect("valid problem");
    // No vertex fails this often: only exhaustion retires.
    let mut planner = planner.with_retirement(NonZeroUsize::MAX);
    let path = planner
        .solve(20_000, &mut Pcg64Mcg::seed_from_u64(3))
        .expect("RRT over the lattice gets around the wall");

    assert_eq!(path.states.first(), Some(&problem.start));
    assert!(
        problem
            .goal
            .contains(&problem.space, path.states.last().expect("non-empty"))
    );
    assert!(
        planner.retired_count() > 10,
        "{} retired",
        planner.retired_count()
    );
}

#[test]
fn connecting_stops_when_steering_bounces_around_the_target() {
    // The goal tree's lattice is offset by 0.5, so the start tree hops 8.5, 9.5, 8.5, ...
    // around goal-tree states and never reaches one; connecting must give up, not loop.
    let problem = Problem {
        space: Euclidean::<2>,
        sampler: UniformBox::new([0.0; 2], [10.0; 2]),
        checker: crate::world::Boxes::default(),
        steering: Lattice,
        start: [1.5, 1.0],
        goal: GoalRegion::new([9.0, 1.0], 0.1),
    };
    let mut planner = RrtConnect::new(problem).expect("valid problem");
    let mut rng = Pcg64Mcg::seed_from_u64(1);

    for _ in 0..200 {
        planner.step(&mut rng);
    }

    assert!(
        planner.node_count() < 10_000,
        "{} nodes",
        planner.node_count()
    );
}
