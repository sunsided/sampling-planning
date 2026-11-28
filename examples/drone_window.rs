//! A quadrotor flying through a window and landing facing a given direction: RRT* over SE(3).
//!
//! The drone's state is a full rigid-body pose, a position and a unit quaternion. The
//! [`Se3`] metric adds the rotation angle between two orientations, weighted, to the distance
//! between positions, and interpolation turns along the shortest rotation (slerp), so RRT*
//! plans where to fly and how to turn on the way. The only way into the second room is a
//! window in the wall between them.
//!
//! A k-d tree cannot index SE(3) poses, so the planner keeps its nodes in a [`VpTree`], which
//! only needs the metric and returns exactly what a linear scan would.
//!
//! With `--features parallel` the example also grows one RRT* tree per thread with
//! [`grow_forest`](sampling_planning::trees::grow_forest) and merges the trees. The trees grow
//! in parallel, but merging them costs one RRT* insertion per node: with 20 threads the forest
//! took about 6 s for 39k nodes, against 15 s for one tree grown to the same size.
//!
//! Run with `cargo run --release --example drone_window [seed]`, or with
//! `cargo run --release --features parallel --example drone_window [seed]`.

use std::f64::consts::FRAC_PI_2;
use std::time::Instant;

use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    GoalRegion, Path, Problem, Se3, Se3State, StraightLine, UniformSe3, Validity, ValidityChecker,
    VpTree, discretized_motion,
    trees::{RrtStar, RrtStarConfig},
};

/// The flight volume: two 5 x 8 x 4 m rooms side by side.
const VOLUME: [f64; 3] = [10.0, 8.0, 4.0];

/// The wall between the rooms at `x` from 4.9 to 5.1, with a window over `y` 5 to 6.5 and
/// `z` 2 to 3.2.
const WALL_X: [f64; 2] = [4.9, 5.1];

/// The window's extent in `y` and `z`.
const WINDOW: [[f64; 2]; 2] = [[5.0, 6.5], [2.0, 3.2]];

/// The drone's radius: it keeps this clearance from walls and the window frame.
const DRONE_RADIUS: f64 = 0.3;

/// RRT* iterations per tree.
const ITERATIONS: usize = 3000;

/// The planning problem of this example.
type Flight = Problem<Se3, UniformSe3, Rooms, StraightLine>;

/// The two rooms and the wall with its window. Orientation never collides here: the drone is
/// modeled as a sphere.
#[derive(Debug, Clone, Copy)]
struct Rooms;

impl Rooms {
    /// Whether the drone at `pose` keeps its clearance from walls, floor and ceiling.
    fn check(pose: &Se3State) -> Validity {
        let p = pose.position;
        let inside = (0..3).all(|i| (DRONE_RADIUS..=VOLUME[i] - DRONE_RADIUS).contains(&p[i]));
        let near_wall = (WALL_X[0] - DRONE_RADIUS..=WALL_X[1] + DRONE_RADIUS).contains(&p[0]);
        let in_window = (WINDOW[0][0] + DRONE_RADIUS..=WINDOW[0][1] - DRONE_RADIUS).contains(&p[1])
            && (WINDOW[1][0] + DRONE_RADIUS..=WINDOW[1][1] - DRONE_RADIUS).contains(&p[2]);

        if inside && (!near_wall || in_window) {
            Validity::Valid
        } else {
            Validity::Invalid
        }
    }
}

impl ValidityChecker<Se3> for Rooms {
    fn check_state(&self, pose: &Se3State) -> Validity {
        Self::check(pose)
    }

    fn check_motion(&self, space: &Se3, from: &Se3State, to: &Se3State) -> Validity {
        discretized_motion(space, from, to, 0.05, Self::check)
    }
}

/// Take off in the first room facing along `x`, land on a pad in the second room facing
/// along `y`.
fn problem() -> Flight {
    Problem {
        space: Se3::new(0.5),
        sampler: UniformSe3::new([0.0; 3], VOLUME),
        checker: Rooms,
        steering: StraightLine::new(1.5),
        start: Se3State::at([1.0, 1.0, 0.5]),
        goal: GoalRegion::new(
            Se3State::from_axis_angle([8.5, 2.0, 0.5], [0.0, 0.0, 1.0], FRAC_PI_2),
            0.4,
        ),
    }
}

/// The RRT* settings: a generous radius constant, edges up to one steering step.
fn config() -> RrtStarConfig {
    RrtStarConfig::new(60.0, 1.5)
}

fn main() {
    let seed = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(1);

    let started = Instant::now();
    let index = VpTree::new(problem().space);
    let mut planner =
        RrtStar::with_nearest_neighbors(problem(), config(), index).expect("take-off pose is free");
    planner.run(ITERATIONS, &mut Pcg64Mcg::seed_from_u64(seed));
    let millis = started.elapsed().as_secs_f64() * 1e3;

    let Some(path) = planner.best_path() else {
        println!("no flight found; try another seed");
        return;
    };

    println!(
        "RRT*: {} nodes, cost {:.2}, {millis:.0} ms",
        planner.node_count(),
        path.cost
    );
    print_flight(&path);

    #[cfg(feature = "parallel")]
    forest(seed, path.cost);
}

/// Prints each waypoint's position and heading (yaw) in degrees.
fn print_flight(path: &Path<Se3State>) {
    for pose in &path.states {
        let [w, x, y, z] = pose.rotation;
        let yaw = (2.0 * (w * z + x * y)).atan2(1.0 - 2.0 * (y * y + z * z));
        let [px, py, pz] = pose.position;
        println!(
            "  ({px:>4.1}, {py:>4.1}, {pz:>4.1})  yaw {:>6.1} deg",
            yaw.to_degrees()
        );
    }
}

/// Grows one tree per available thread with [`ITERATIONS`] iterations each, merges them and
/// compares the result with the single tree's `single_cost` (feature `parallel`).
#[cfg(feature = "parallel")]
fn forest(seed: u64, single_cost: f64) {
    use sampling_planning::{PathLength, trees::grow_forest};

    let threads = std::thread::available_parallelism().map_or(1, std::num::NonZeroUsize::get);
    let mut rngs: Vec<Pcg64Mcg> = (0..threads)
        .map(|i| Pcg64Mcg::seed_from_u64(seed + u64::try_from(i).expect("few threads")))
        .collect();

    let started = Instant::now();
    let merged = grow_forest(
        &problem(),
        config(),
        &VpTree::new(problem().space),
        &PathLength,
        &mut rngs,
        ITERATIONS,
    )
    .expect("take-off pose is free");
    let millis = started.elapsed().as_secs_f64() * 1e3;
    let cost = merged.best_path().map_or(f64::NAN, |path| path.cost);

    println!(
        "forest of {threads} trees: {} nodes, cost {cost:.2} (single tree {single_cost:.2}), {millis:.0} ms",
        merged.node_count()
    );
}
