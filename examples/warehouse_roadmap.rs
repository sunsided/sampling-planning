//! A warehouse robot answering many pick requests on one roadmap: build a PRM* once, then
//! query it again and again.
//!
//! The robot is a disc of radius 0.4 in a 30 x 20 hall with three double rows of shelves.
//! Inflating every shelf by the robot radius turns it into a point in the configuration
//! space, which is what the planner sees. Building the roadmap is the expensive part; each
//! query after that only connects its endpoints and runs Dijkstra. A [`QueryWorkspace`] keeps
//! the search buffers between queries, so a query allocates almost nothing.
//!
//! The example builds the roadmap twice. The eager one checks every edge while it grows; the
//! lazy one builds in a fraction of the time and checks edges only when a query's candidate
//! path uses them. Lazy verdicts stay in the roadmap, so later queries reuse them, also through
//! the shared `&self` queries used here. Both answer every request with the same cost.
//!
//! With `--features parallel` the roadmap is also built with a [`ParallelChecker`] and all
//! requests are answered at once with [`Prm::query_batch`]; both give the same results as the
//! single-threaded runs.
//!
//! Run with `cargo run --release --example warehouse_roadmap`, or with
//! `cargo run --release --features parallel --example warehouse_roadmap`.

use std::time::Instant;

use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    Euclidean, GoalRegion, KdTree, Path, UniformBox, Validity, ValidityChecker, discretized_motion,
    roadmaps::{Prm, PrmConfig, QueryWorkspace},
};

/// An axis-aligned box given by its lower and upper corners.
type Rect = ([f64; 2], [f64; 2]);

/// Hall width and depth.
const HALL: [f64; 2] = [30.0, 20.0];

/// The robot's radius.
const ROBOT_RADIUS: f64 = 0.4;

/// Shelf rows: each 2 deep and 14 long, with aisles between and around them.
const SHELVES: [Rect; 3] = [
    ([4.0, 3.0], [18.0, 5.0]),
    ([4.0, 9.0], [18.0, 11.0]),
    ([4.0, 15.0], [18.0, 17.0]),
];

/// A packing station wall that leaves a door at `y` 8 to 12.
const STATION_WALLS: [Rect; 2] = [([22.0, 0.0], [23.0, 8.0]), ([22.0, 12.0], [23.0, 20.0])];

/// The loading dock where every request starts.
const DOCK: [f64; 2] = [27.0, 2.0];

/// The roadmap planner of this example.
type Roadmap<C> = Prm<Euclidean<2>, UniformBox<2>, C, KdTree<2>>;

/// The hall's obstacles, each grown by [`ROBOT_RADIUS`].
#[derive(Debug, Clone, Copy)]
struct Hall;

impl Hall {
    /// Whether the robot's center at `point` keeps the robot clear of walls and shelves.
    fn check(point: &[f64; 2]) -> Validity {
        let in_hall = (0..2).all(|i| (ROBOT_RADIUS..=HALL[i] - ROBOT_RADIUS).contains(&point[i]));
        let blocked = SHELVES
            .iter()
            .chain(&STATION_WALLS)
            .any(|rect| distance_to_rect(rect, point) < ROBOT_RADIUS);

        if in_hall && !blocked {
            Validity::Valid
        } else {
            Validity::Invalid
        }
    }
}

impl ValidityChecker<Euclidean<2>> for Hall {
    fn check_state(&self, point: &[f64; 2]) -> Validity {
        Self::check(point)
    }

    fn check_motion(&self, space: &Euclidean<2>, from: &[f64; 2], to: &[f64; 2]) -> Validity {
        discretized_motion(space, from, to, 0.05, Self::check)
    }
}

/// The distance from `point` to the closed box `rect`; zero inside.
fn distance_to_rect(rect: &Rect, point: &[f64; 2]) -> f64 {
    let gap = |i: usize| (rect.0[i] - point[i]).max(point[i] - rect.1[i]).max(0.0);

    gap(0).hypot(gap(1))
}

/// Pick requests: from the dock to a slot in front of every shelf face, every 3 units.
fn requests() -> Vec<([f64; 2], GoalRegion<[f64; 2]>)> {
    let faces = SHELVES
        .iter()
        .flat_map(|shelf| [shelf.0[1] - 0.8, shelf.1[1] + 0.8]);

    faces
        .flat_map(|y| (0..5).map(move |slot| [5.0 + 3.0 * f64::from(slot), y]))
        .map(|slot| (DOCK, GoalRegion::new(slot, 0.3)))
        .collect()
}

/// Builds a PRM* roadmap with `samples` samples over the hall with `checker`, lazy or eager.
fn build<C: ValidityChecker<Euclidean<2>>>(checker: C, samples: usize, lazy: bool) -> Roadmap<C> {
    let free_volume = HALL[0] * HALL[1] - 3.0 * 14.0 * 2.0 - 16.0;
    let mut config = PrmConfig::star(1.5 * PrmConfig::gamma_lower_bound(2, free_volume));
    config.lazy = lazy;

    let mut prm = Prm::with_nearest_neighbors(
        Euclidean,
        UniformBox::new([0.0, 0.0], HALL),
        checker,
        config,
        KdTree::default(),
    )
    .expect("valid config");
    prm.grow(samples, &mut Pcg64Mcg::seed_from_u64(7));

    prm
}

/// Milliseconds since `start`.
fn millis(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1e3
}

fn main() {
    let requests = requests();
    let eager = plan(false, &requests);
    let lazy = plan(true, &requests);

    for (a, b) in eager.iter().zip(&lazy) {
        let cost = |path: &Option<Path<[f64; 2]>>| path.as_ref().map(|path| path.cost);
        assert_eq!(cost(a), cost(b), "lazy and eager roadmaps agree");
    }

    for ((_, goal), path) in requests.iter().zip(&eager).take(4) {
        let cost = path.as_ref().map_or(f64::NAN, |path| path.cost);
        println!(
            "  dock -> ({:>4.1}, {:>4.1}): {cost:.2}",
            goal.center()[0],
            goal.center()[1]
        );
    }

    #[cfg(feature = "parallel")]
    parallel::compare(&requests, &eager);
}

/// Builds a lazy or eager roadmap, answers every request through one [`QueryWorkspace`] and
/// prints the timings; returns the paths.
fn plan(lazy: bool, requests: &[([f64; 2], GoalRegion<[f64; 2]>)]) -> Vec<Option<Path<[f64; 2]>>> {
    let kind = if lazy { "lazy" } else { "eager" };
    let started = Instant::now();
    let prm = build(Hall, 3000, lazy);
    println!(
        "{kind} roadmap: {} nodes, built in {:.1} ms",
        prm.node_count(),
        millis(started)
    );

    let mut workspace = QueryWorkspace::default();
    let started = Instant::now();
    let paths: Vec<Option<Path<[f64; 2]>>> = requests
        .iter()
        .map(|(start, goal)| {
            prm.query_with(&mut workspace, start, goal)
                .expect("the dock is free")
        })
        .collect();
    let elapsed = millis(started);

    let solved = paths.iter().flatten().count();
    let total: f64 = paths.iter().flatten().map(|path| path.cost).sum();
    println!(
        "  {} requests in {elapsed:.2} ms ({:.3} ms each): {solved} solved, {total:.1} total drive",
        requests.len(),
        elapsed / f64::from(u32::try_from(requests.len()).expect("few requests")),
    );

    paths
}

/// The multi-threaded variants (feature `parallel`).
#[cfg(feature = "parallel")]
mod parallel {
    use std::num::NonZeroUsize;
    use std::time::Instant;

    use sampling_planning::{GoalRegion, ParallelChecker, Path};

    use super::{Hall, build, millis};

    /// Builds the roadmap with batched parallel checks and answers all requests at once, and
    /// checks that both match the single-threaded results in `paths`.
    pub(super) fn compare(
        requests: &[([f64; 2], GoalRegion<[f64; 2]>)],
        paths: &[Option<Path<[f64; 2]>>],
    ) {
        let threads = std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);

        let started = Instant::now();
        let prm = build(ParallelChecker::new(Hall, threads, 64), 3000, false);
        println!(
            "parallel roadmap on {threads} threads: built in {:.1} ms",
            millis(started)
        );

        let started = Instant::now();
        let batch: Vec<_> = prm
            .query_batch(requests, threads)
            .into_iter()
            .map(|result| result.expect("the dock is free"))
            .collect();
        println!(
            "parallel queries: {} requests in {:.2} ms",
            requests.len(),
            millis(started)
        );

        assert_eq!(batch, paths, "parallel results equal the sequential ones");
    }
}
