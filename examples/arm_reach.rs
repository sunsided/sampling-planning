//! A two-link robot arm reaching under a shelf: BIT*, ABIT*, AIT* and EIT* in joint space,
//! compared by how many collision checks they spend.
//!
//! The arm stands on a table at the origin, with an upper arm of length 1 and a forearm of
//! 0.8. Its configuration is the pair of joint angles, so the planners work in a 2D joint
//! space, but collisions happen in the workspace: a configuration is valid when both links,
//! run through forward kinematics, stay above the table and clear of a shelf board and a
//! pillar. The goal configuration comes from inverse kinematics for a grasp point under the
//! board.
//!
//! For real arms the collision check (meshes, many links) dominates planning time, so every
//! planner gets the same budget of collision checks. All four search batches of informed
//! samples. ABIT* inflates its heuristic to find a first solution with few checks. AIT* and
//! EIT* add a reverse search that guides the forward search toward edges that are likely
//! valid, so they spend the budget on edges that improve the path: within 10 000 checks
//! they cut the joint-space path well below the first solution, while BIT* and ABIT* are
//! still at theirs. EIT* also checks edges sparsely in that search and estimates the
//! checking effort, which buys it a first solution with the fewest checks. The reverse
//! searches cost CPU time of their own, as the `ms` column shows: they pay off when one
//! collision check costs more than this example's microseconds.
//!
//! Run with `cargo run --release --example arm_reach [seed]`.

use std::cell::Cell;
use std::f64::consts::PI;
use std::rc::Rc;
use std::time::Instant;

use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    Euclidean, GoalRegion, KdTree, Path, UniformBox, Validity, ValidityChecker, discretized_motion,
    informed::{AbitStar, AbitStarConfig, AitStar, BitStar, BitStarConfig, EitStar, EitStarConfig},
};

/// An axis-aligned box given by its lower and upper corners.
type Rect = ([f64; 2], [f64; 2]);

/// Upper arm and forearm lengths.
const LINKS: [f64; 2] = [1.0, 0.8];

/// A shelf board over the grasp point and a pillar left of the arm.
const OBSTACLES: [Rect; 2] = [([0.7, 0.75], [1.6, 0.85]), ([-0.9, 0.0], [-0.7, 1.2])];

/// The point the gripper has to reach, under the shelf board.
const GRASP: [f64; 2] = [1.1, 0.35];

/// The joint-space distance between two collision checks along a motion, in radians.
const RESOLUTION: f64 = 0.02;

/// The collision checks each planner may spend in total.
const CHECK_BUDGET: usize = 10_000;

/// Collision checks of the arm against the table and [`OBSTACLES`], counting every state it
/// evaluates.
#[derive(Debug, Clone, Default)]
struct Arm {
    /// The number of configurations checked so far, shared by all clones.
    checks: Rc<Cell<usize>>,
}

impl Arm {
    /// Whether both links at the joint angles `q` stay above the table and off the
    /// obstacles, tested at points every 0.05 along each link.
    fn check(&self, q: &[f64; 2]) -> Validity {
        self.checks.set(self.checks.get() + 1);

        let [elbow, gripper] = forward_kinematics(q);
        let points = (0..=20).flat_map(|i| {
            let t = f64::from(i) / 20.0;
            [lerp([0.0, 0.0], elbow, t), lerp(elbow, gripper, t)]
        });

        if points.into_iter().all(|point| is_free(&point)) {
            Validity::Valid
        } else {
            Validity::Invalid
        }
    }
}

impl ValidityChecker<Euclidean<2>> for Arm {
    fn check_state(&self, q: &[f64; 2]) -> Validity {
        self.check(q)
    }

    fn check_motion(&self, space: &Euclidean<2>, from: &[f64; 2], to: &[f64; 2]) -> Validity {
        discretized_motion(space, from, to, RESOLUTION, |q| self.check(q))
    }
}

/// The elbow and gripper positions for the joint angles `q`: the shoulder angle from the
/// table's x axis, the elbow angle relative to the upper arm.
fn forward_kinematics(q: &[f64; 2]) -> [[f64; 2]; 2] {
    let elbow = [LINKS[0] * q[0].cos(), LINKS[0] * q[0].sin()];
    let gripper = [
        elbow[0] + LINKS[1] * (q[0] + q[1]).cos(),
        elbow[1] + LINKS[1] * (q[0] + q[1]).sin(),
    ];

    [elbow, gripper]
}

/// The elbow-up joint angles that put the gripper at `target`.
fn inverse_kinematics(target: [f64; 2]) -> [f64; 2] {
    let [l1, l2] = LINKS;
    let reach_squared = target[0] * target[0] + target[1] * target[1];
    let elbow = -((reach_squared - l1 * l1 - l2 * l2) / (2.0 * l1 * l2)).acos();
    let shoulder = target[1].atan2(target[0]) - (l2 * elbow.sin()).atan2(l1 + l2 * elbow.cos());

    [shoulder, elbow]
}

/// The point a fraction `t` of the way from `a` to `b`.
fn lerp(a: [f64; 2], b: [f64; 2], t: f64) -> [f64; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

/// Whether a workspace point is above the table and outside every obstacle.
fn is_free(point: &[f64; 2]) -> bool {
    let inside = |rect: &Rect| (0..2).all(|i| (rect.0[i]..=rect.1[i]).contains(&point[i]));

    point[1] >= 0.0 && !OBSTACLES.iter().any(inside)
}

/// One planner's run: checks to the first solution, the cost then, and the cost once the
/// [`CHECK_BUDGET`] is spent.
struct Outcome {
    /// The planner's name.
    name: &'static str,
    /// Configurations checked until the first solution.
    first_checks: usize,
    /// The first solution's cost in joint-space radians.
    first_cost: f64,
    /// The final solution.
    path: Path<[f64; 2]>,
    /// Wall-clock time of the whole run in milliseconds.
    millis: f64,
}

/// The common interface of the four planners this example drives.
trait Planner {
    /// Runs one step.
    fn step(&mut self, rng: &mut Pcg64Mcg);
    /// The best path so far.
    fn best_path(&self) -> Option<Path<[f64; 2]>>;
}

/// Implements [`Planner`] for each planner type by forwarding to its inherent methods.
macro_rules! planner {
    ($($ty:ident),*) => {$(
        impl Planner for $ty<Euclidean<2>, UniformBox<2>, Arm, KdTree<2>> {
            fn step(&mut self, rng: &mut Pcg64Mcg) {
                $ty::step(self, rng);
            }

            fn best_path(&self) -> Option<Path<[f64; 2]>> {
                $ty::best_path(self)
            }
        }
    )*};
}

planner!(BitStar, AbitStar, AitStar, EitStar);

/// Steps `planner` to its first solution and then on until it has spent [`CHECK_BUDGET`]
/// collision checks, counting the checks of `arm`.
///
/// # Panics
///
/// Panics if the budget runs out before a first solution.
fn run(name: &'static str, arm: &Arm, mut planner: impl Planner, seed: u64) -> Outcome {
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let started = Instant::now();
    arm.checks.set(0);

    let first = loop {
        planner.step(&mut rng);
        if let Some(path) = planner.best_path() {
            break path;
        }

        assert!(arm.checks.get() < CHECK_BUDGET, "{name} found no solution");
    };

    let first_checks = arm.checks.get();
    while arm.checks.get() < CHECK_BUDGET {
        planner.step(&mut rng);
    }

    Outcome {
        name,
        first_checks,
        first_cost: first.cost,
        path: planner.best_path().expect("solutions are kept"),
        millis: started.elapsed().as_secs_f64() * 1e3,
    }
}

fn main() {
    let seed = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(3);
    let goal = inverse_kinematics(GRASP);
    let outcomes = plan_all([1.9, -0.6], goal, seed);
    print_results(goal, &outcomes);
}

/// Runs the four planners from the joint angles `start` to `goal`.
fn plan_all(start: [f64; 2], goal: [f64; 2], seed: u64) -> [Outcome; 4] {
    let arm = Arm::default();
    assert!(arm.check(&start).is_valid() && arm.check(&goal).is_valid());

    let (space, sampler) = (Euclidean::<2>, UniformBox::new([0.0, -PI], [PI, PI]));
    let target = GoalRegion::new(goal, 0.05);
    let bit = BitStarConfig::new(1.5 * BitStarConfig::gamma_lower_bound(2, 2.0 * PI * PI));
    let eit = EitStarConfig::from_bit_star(bit, RESOLUTION);
    let abit = AbitStarConfig::from_bit_star(bit);

    // Each planner over a k-d tree, which indexes the 2D joint space.
    macro_rules! make {
        ($ty:ident, $config:expr) => {
            $ty::with_nearest_neighbors(
                space,
                sampler,
                arm.clone(),
                start,
                target.clone(),
                $config,
                KdTree::default(),
            )
            .expect("valid problem")
        };
    }

    [
        run("BIT*", &arm, make!(BitStar, bit), seed),
        run("ABIT*", &arm, make!(AbitStar, abit), seed),
        run("AIT*", &arm, make!(AitStar, bit), seed),
        run("EIT*", &arm, make!(EitStar, eit), seed),
    ]
}

/// Prints the comparison table and the gripper positions along EIT*'s path.
fn print_results(goal: [f64; 2], outcomes: &[Outcome; 4]) {
    println!(
        "grasp at ({:.2}, {:.2}): shoulder {:.2} rad, elbow {:.2} rad",
        GRASP[0], GRASP[1], goal[0], goal[1]
    );
    println!(
        "{:<6} {:>14} {:>10} {:>16} {:>9}",
        "", "checks to 1st", "1st cost", "cost at budget", "ms"
    );

    for outcome in outcomes {
        println!(
            "{:<6} {:>14} {:>10.3} {:>16.3} {:>9.1}",
            outcome.name,
            outcome.first_checks,
            outcome.first_cost,
            outcome.path.cost,
            outcome.millis
        );
    }

    println!("(costs in joint-space radians; budget {CHECK_BUDGET} checks each)");

    let best = &outcomes[3].path;
    println!("\nEIT* gripper waypoints:");
    for q in &best.states {
        let [_, gripper] = forward_kinematics(q);
        println!("  ({:>5.2}, {:>5.2})", gripper[0], gripper[1]);
    }
}
