//! A differential-drive robot through an office floor: kinodynamic planning with SST and
//! single- and multi-level KPIECE.
//!
//! A differential-drive robot cannot turn on the spot here: it drives forward at 0.2 to 1 m/s
//! and turns at most 1.5 rad/s, so its state is a pose and its motions are arcs (a
//! unicycle). Geometric planners would connect poses with straight lines it cannot follow.
//! Kinodynamic planners instead sample controls and simulate them with a [`Propagator`], so
//! every trajectory they return can be driven, control by control.
//!
//! - SST keeps a sparse tree of the best trajectories and keeps improving the arrival time.
//! - KPIECE grows its tree where a grid over a projection of the state (here the position)
//!   is least explored, and stops at the first solution. Several grid levels let it pick a
//!   coarse region first and a fine cell inside it, which helps in large or cluttered spaces.
//!
//! Run with `cargo run --release --example unicycle_maze [seed]`.

use std::num::NonZeroUsize;
use std::time::Instant;

use rand_core::{Rng, SeedableRng};
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    GoalRegion, Se2, Se2State, UniformSe2, Validity, ValidityChecker, discretized_motion,
    kinodynamic::{
        KinodynamicProblem, Kpiece, KpieceConfig, Propagator, Sst, SstConfig, Trajectory,
    },
    unit_f64,
};

/// An axis-aligned box given by its lower and upper corners.
type Rect = ([f64; 2], [f64; 2]);

/// The floor: 12 x 8 m.
const FLOOR: [f64; 2] = [12.0, 8.0];

/// Office walls, each 0.2 m thick, with doors the robot has to find.
const WALLS: [Rect; 4] = [
    ([3.0, 0.0], [3.2, 5.5]),
    ([6.0, 2.5], [6.2, 8.0]),
    ([9.0, 0.0], [9.2, 5.5]),
    ([9.2, 5.3], [10.5, 5.5]),
];

/// The robot's radius: walls are grown by it.
const ROBOT_RADIUS: f64 = 0.25;

/// The integration step in seconds.
const DT: f64 = 0.1;

/// The kinodynamic problem of this example.
type Floor = KinodynamicProblem<Se2, UniformSe2, Office, Unicycle>;

/// A unicycle: the control is forward speed in m/s and turn rate in rad/s.
#[derive(Debug, Clone, Copy)]
struct Unicycle;

impl Propagator<Se2> for Unicycle {
    type Control = [f64; 2];

    fn control_dimension(&self) -> usize {
        2
    }

    fn sample_control<R: Rng + ?Sized>(&mut self, rng: &mut R) -> [f64; 2] {
        [0.2 + 0.8 * unit_f64(rng), 3.0 * unit_f64(rng) - 1.5]
    }

    /// Drives the exact arc for a constant speed and turn rate.
    fn propagate(&self, _: &Se2, from: &Se2State, control: &[f64; 2], duration: f64) -> Se2State {
        let [speed, turn] = *control;
        let theta = from.theta + turn * duration;

        if turn.abs() < 1e-9 {
            let (sin, cos) = from.theta.sin_cos();
            return Se2State::new(
                from.x + speed * duration * cos,
                from.y + speed * duration * sin,
                theta,
            );
        }

        let radius = speed / turn;
        Se2State::new(
            from.x + radius * (theta.sin() - from.theta.sin()),
            from.y - radius * (theta.cos() - from.theta.cos()),
            sampling_planning::wrap_angle(theta),
        )
    }
}

/// The office floor: walls grown by the robot radius, and the floor's outline.
#[derive(Debug, Clone, Copy)]
struct Office;

impl Office {
    /// Whether the robot at `pose` stays on the floor and clear of the walls.
    fn check(pose: &Se2State) -> Validity {
        let point = [pose.x, pose.y];
        let on_floor = (0..2).all(|i| (ROBOT_RADIUS..=FLOOR[i] - ROBOT_RADIUS).contains(&point[i]));
        let hits_wall = WALLS.iter().any(|wall| {
            (0..2)
                .all(|i| (wall.0[i] - ROBOT_RADIUS..=wall.1[i] + ROBOT_RADIUS).contains(&point[i]))
        });

        if on_floor && !hits_wall {
            Validity::Valid
        } else {
            Validity::Invalid
        }
    }
}

impl ValidityChecker<Se2> for Office {
    fn check_state(&self, pose: &Se2State) -> Validity {
        Self::check(pose)
    }

    fn check_motion(&self, space: &Se2, from: &Se2State, to: &Se2State) -> Validity {
        discretized_motion(space, from, to, 0.05, Self::check)
    }
}

/// From the left office, facing up, to the right office; the heading at the goal barely
/// counts.
fn problem() -> Floor {
    KinodynamicProblem {
        space: Se2::new(0.1),
        sampler: UniformSe2::new([0.0, 0.0], FLOOR),
        checker: Office,
        propagator: Unicycle,
        start: Se2State::new(1.0, 1.0, std::f64::consts::FRAC_PI_2),
        goal: GoalRegion::new(Se2State::new(10.8, 1.0, 0.0), 0.5),
    }
}

/// The position of a pose, which KPIECE grids over.
fn position(pose: &Se2State) -> [f64; 2] {
    [pose.x, pose.y]
}

fn main() {
    let seed = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(1);

    println!(
        "{:<20} {:>11} {:>11} {:>9}",
        "planner", "iterations", "arrival", "ms"
    );

    let config = SstConfig::new(0.5, 0.2, DT, 20);
    let mut sst = Sst::new(problem(), config).expect("start is free");
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let started = Instant::now();
    let mut first = None;
    for iteration in 1..=60_000 {
        sst.step(&mut rng);
        if first.is_none() && sst.best_trajectory().is_some() {
            first = Some(iteration);
            report(
                "SST, first solution",
                iteration,
                sst.best_trajectory(),
                started,
            );
        }
    }

    report("SST, 60000 iters", 60_000, sst.best_trajectory(), started);

    let config = KpieceConfig::new([0.4, 0.4], DT, 20);
    for levels in [1, 3] {
        let levels = NonZeroUsize::new(levels).expect("non-zero");
        let mut kpiece =
            Kpiece::with_levels(problem(), position, config, levels, 3).expect("valid config");
        let mut rng = Pcg64Mcg::seed_from_u64(seed);
        let started = Instant::now();
        let mut iterations = 0;
        while kpiece.best_trajectory().is_none() && iterations < 200_000 {
            kpiece.step(&mut rng);
            iterations += 1;
        }

        let name = format!("KPIECE, {levels} level(s)");
        report(&name, iterations, kpiece.best_trajectory(), started);
    }
}

/// Prints one result line: the arrival time of `trajectory` in seconds, or "no solution" when
/// the run used up its `iterations` without reaching the goal.
fn report(
    name: &str,
    iterations: usize,
    trajectory: Option<Trajectory<Se2State, [f64; 2]>>,
    started: Instant,
) {
    let arrival = trajectory.map_or("no solution".to_owned(), |t| format!("{:.1} s", t.cost));
    println!(
        "{name:<20} {iterations:>11} {arrival:>11} {:>9.1}",
        started.elapsed().as_secs_f64() * 1e3
    );
}
