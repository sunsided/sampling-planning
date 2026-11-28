//! Parallel parking with a car that can reverse: RRT-Connect over the Reeds-Shepp space,
//! followed by path shortcutting.
//!
//! A 2.4 x 1.2 car drives down a street and has to park in a gap between two parked cars at
//! the curb. A car cannot move sideways, so straight lines in the plane are useless here;
//! the Reeds-Shepp space connects two poses with the shortest drive of forward and reverse
//! arcs at the turning radius, and the planner searches over those drives. The validity
//! checker tests the car's whole footprint, not just its center.
//!
//! The parking pose is a narrow target that a tree grown only from the start rarely hits.
//! RRT-Connect also grows a tree backwards out of the gap and connects the two, which finds a
//! maneuver quickly; [`shortcut`] then replaces detours with direct Reeds-Shepp drives.
//!
//! Run with `cargo run --release --example car_parking [seed]`.

use std::fmt::Write;

use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    GoalRegion, Path, Problem, ReedsShepp, Se2State, StateSpace, StraightLine, UniformSe2,
    Validity, ValidityChecker, discretized_motion, shortcut, trees::RrtConnect,
};

/// An axis-aligned box given by its lower and upper corners.
type Rect = ([f64; 2], [f64; 2]);

/// The street: 20 long, 8 wide, curb at `y = 0`.
const STREET: Rect = ([0.0, 0.0], [20.0, 8.0]);

/// The two parked cars; the gap between them is 4 long, 1.7 car lengths.
const PARKED: [Rect; 2] = [([1.0, 0.3], [5.2, 1.5]), ([9.2, 0.3], [13.4, 1.5])];

/// Car length and width.
const CAR: [f64; 2] = [2.4, 1.2];

/// The minimum turning radius of the car.
const TURNING_RADIUS: f64 = 1.5;

/// Map cells per world unit.
const CELLS_PER_UNIT: f64 = 2.0;

/// The parking problem: the street, its footprint check and the two poses.
type Parking = Problem<ReedsShepp, UniformSe2, Footprint, StraightLine>;

/// Checks the car's footprint against the street bounds and the parked cars.
#[derive(Debug, Clone, Copy)]
struct Footprint;

impl Footprint {
    /// The validity of the car at `pose`: every outline point must be on the street and off
    /// the parked cars.
    fn check(pose: &Se2State) -> Validity {
        let free = outline(pose)
            .iter()
            .all(|point| inside(&STREET, point) && !PARKED.iter().any(|car| inside(car, point)));

        if free {
            Validity::Valid
        } else {
            Validity::Invalid
        }
    }
}

impl ValidityChecker<ReedsShepp> for Footprint {
    fn check_state(&self, pose: &Se2State) -> Validity {
        Self::check(pose)
    }

    fn check_motion(&self, space: &ReedsShepp, from: &Se2State, to: &Se2State) -> Validity {
        discretized_motion(space, from, to, 0.05, Self::check)
    }
}

/// Twelve points on the outline of the car at `pose`: corners and points along the sides.
fn outline(pose: &Se2State) -> Vec<[f64; 2]> {
    let (sin, cos) = pose.theta.sin_cos();
    let [half_length, half_width] = [CAR[0] / 2.0, CAR[1] / 2.0];
    let along = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0];
    let across = [-1.0, 0.0, 1.0];

    along
        .iter()
        .flat_map(|&a| {
            across
                .iter()
                .map(move |&b| (a * half_length, b * half_width))
        })
        .map(|(a, b)| [pose.x + a * cos - b * sin, pose.y + a * sin + b * cos])
        .collect()
}

/// Whether `point` lies in the closed box `rect`.
fn inside(rect: &Rect, point: &[f64; 2]) -> bool {
    (0..2).all(|i| (rect.0[i]..=rect.1[i]).contains(&point[i]))
}

/// From the left end of the street, driving right, into the gap, also facing right.
fn problem() -> Parking {
    Problem {
        space: ReedsShepp::new(TURNING_RADIUS),
        sampler: UniformSe2::new(STREET.0, STREET.1),
        checker: Footprint,
        steering: StraightLine::new(3.0),
        start: Se2State::new(2.0, 4.0, 0.0),
        goal: GoalRegion::new(Se2State::new(7.2, 0.9, 0.0), 0.25),
    }
}

fn main() {
    let seed = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(1);
    let world = problem();
    let space = world.space;

    // Reeds-Shepp distances are not Euclidean, so the default linear scan index is the right
    // one; a k-d tree would prune neighbors wrongly.
    let mut planner = RrtConnect::new(world).expect("start and goal are free");
    let Some(raw) = planner.solve(20_000, &mut Pcg64Mcg::seed_from_u64(seed)) else {
        println!("no parking maneuver found; try another seed");
        return;
    };

    let path = shortcut(&space, &Footprint, &raw, f64::INFINITY);
    println!("tree nodes:  {}", planner.node_count());
    for (label, path) in [("planned:  ", &raw), ("shortcut: ", &path)] {
        println!(
            "{label} drive length {:.2}, {} waypoints, {} gear changes",
            path.cost,
            path.states.len(),
            gear_changes(space, path)
        );
    }

    print!("{}", draw(space, &path));
}

/// How often the car switches between forward and reverse along `path`.
fn gear_changes(space: ReedsShepp, path: &Path<Se2State>) -> usize {
    let directions: Vec<bool> = path
        .states
        .windows(2)
        .flat_map(|pair| space.path(&pair[0], &pair[1]).segments().to_vec())
        .filter(|segment| segment.length.abs() > 1e-9)
        .map(|segment| segment.length > 0.0)
        .collect();

    directions
        .windows(2)
        .filter(|pair| pair[0] != pair[1])
        .count()
}

/// The street with the parked cars (`#`), the driven path (`*`), start (`S`) and goal
/// (`G`), north up.
fn draw(space: ReedsShepp, path: &Path<Se2State>) -> String {
    let width = cell(STREET.1[0]) + 1;
    let height = cell(STREET.1[1]) + 1;
    let mut grid = vec![vec![' '; width]; height];

    for car in &PARKED {
        for line in &mut grid[cell(car.0[1])..=cell(car.1[1])] {
            line[cell(car.0[0])..=cell(car.1[0])].fill('#');
        }
    }

    for pair in path.states.windows(2) {
        for step in 0..=40 {
            let pose = space.interpolate(&pair[0], &pair[1], f64::from(step) / 40.0);
            grid[cell(pose.y)][cell(pose.x)] = '*';
        }
    }

    let (start, goal) = (path.states[0], path.states[path.states.len() - 1]);
    grid[cell(start.y)][cell(start.x)] = 'S';
    grid[cell(goal.y)][cell(goal.x)] = 'G';

    let mut out = format!("+{}+\n", "-".repeat(width));
    for line in grid.iter().rev() {
        out.push('|');
        out.extend(line);
        out.push_str("|\n");
    }

    writeln!(out, "+{}+ curb", "=".repeat(width)).expect("writing to a String");
    writeln!(out, "final heading {:.1} degrees", goal.theta.to_degrees())
        .expect("writing to a String");

    out
}

/// The map cell index of a world coordinate (non-negative by construction).
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "flooring a non-negative, bounded world coordinate is the intent"
)]
fn cell(value: f64) -> usize {
    (value.max(0.0) * CELLS_PER_UNIT) as usize
}
