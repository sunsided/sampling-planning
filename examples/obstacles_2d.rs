//! Plans through a 2D world with two walls with every geometric planner of the crate, prints a
//! comparison table and draws each path.
//!
//! Every slalom solution is longer than the largest start-to-goal detour inside the box, so the
//! informed set covers the whole box here and Informed RRT* runs exactly like RRT*; on a world
//! with a thin informed set (see the integration tests' pole world) it converges faster.
//!
//! Run with `cargo run --release --example obstacles_2d [seed]`.

use std::fmt::Write;
use std::time::Instant;

use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    Euclidean, GoalRegion, KdTree, Path, Problem, StraightLine, UniformBox, Validity,
    ValidityChecker, discretized_motion,
    informed::{BitStar, BitStarConfig},
    roadmaps::{
        BfmtStar, BfmtStarConfig, FmtStar, FmtStarConfig, Prm, PrmConfig, Spars2, Spars2Config,
    },
    trees::{Rrt, RrtConfig, RrtConnect, RrtSharp, RrtStar, RrtStarConfig},
};

/// An axis-aligned box given by its lower and upper corners.
type Rect = ([f64; 2], [f64; 2]);

/// The planning problem every tree planner solves.
type World = Problem<Euclidean<2>, UniformBox<2>, Walls, StraightLine>;

/// Two walls in a 10 x 10 world, forcing a slalom from the bottom left to the top right.
const WALLS: [Rect; 2] = [([4.0, 0.0], [5.0, 7.0]), ([6.0, 3.0], [7.0, 10.0])];

/// The free volume: the box minus both walls.
const FREE_VOLUME: f64 = 100.0 - 7.0 - 7.0;

/// Map width and height in characters.
const MAP_SIZE: usize = 24;

/// Map cells per world unit: [`MAP_SIZE`] cells across the 10 unit wide world.
const CELLS_PER_UNIT: f64 = 2.4;

/// How many maps are drawn side by side.
const MAPS_PER_ROW: usize = 4;

/// A validity checker for [`WALLS`].
#[derive(Debug, Clone, Copy)]
struct Walls;

impl ValidityChecker<Euclidean<2>> for Walls {
    fn check_state(&self, state: &[f64; 2]) -> Validity {
        if WALLS.iter().any(|wall| inside(wall, state)) {
            Validity::Invalid
        } else {
            Validity::Valid
        }
    }

    fn check_motion(&self, space: &Euclidean<2>, from: &[f64; 2], to: &[f64; 2]) -> Validity {
        discretized_motion(space, from, to, 0.02, |state| self.check_state(state))
    }
}

/// Whether `state` lies in the closed box `rect`.
fn inside(rect: &Rect, state: &[f64; 2]) -> bool {
    (0..2).all(|i| (rect.0[i]..=rect.1[i]).contains(&state[i]))
}

/// The slalom problem from (1, 1) to within 0.5 of (9, 9).
fn problem() -> World {
    Problem {
        space: Euclidean::<2>,
        sampler: UniformBox::new([0.0, 0.0], [10.0, 10.0]),
        checker: Walls,
        steering: StraightLine::new(1.0),
        start: [1.0, 1.0],
        goal: GoalRegion::new([9.0, 9.0], 0.5),
    }
}

/// One planner's result.
struct Outcome {
    /// The planner's name.
    name: &'static str,
    /// The path found, if any.
    path: Option<Path<[f64; 2]>>,
    /// The planner's own size measure: nodes, guards or vertices.
    nodes: usize,
    /// Wall-clock time in milliseconds.
    millis: f64,
}

/// Runs `plan`, timing it, and wraps the result.
fn timed(name: &'static str, plan: impl FnOnce() -> (Option<Path<[f64; 2]>>, usize)) -> Outcome {
    let start = Instant::now();
    let (path, nodes) = plan();

    Outcome {
        name,
        path,
        nodes,
        millis: start.elapsed().as_secs_f64() * 1e3,
    }
}

fn main() {
    let seed = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(42);
    let mut outcomes = Vec::from(tree_planners(seed));
    outcomes.extend(roadmap_planners(seed));
    outcomes.extend(batch_planners(seed));

    println!("{:<16} {:>8} {:>7} {:>9}", "planner", "cost", "nodes", "ms");
    for outcome in &outcomes {
        let cost = outcome
            .path
            .as_ref()
            .map_or("-".to_owned(), |p| format!("{:.3}", p.cost));
        println!(
            "{:<16} {cost:>8} {:>7} {:>9.1}",
            outcome.name, outcome.nodes, outcome.millis
        );
    }

    println!();
    for row in outcomes.chunks(MAPS_PER_ROW) {
        print!("{}", draw_row(row));
    }
}

/// RRT, RRT-Connect, RRT*, Informed RRT* and RRT# on the slalom.
fn tree_planners(seed: u64) -> [Outcome; 5] {
    let rng = || Pcg64Mcg::seed_from_u64(seed);
    let star = RrtStarConfig::new(1.5 * RrtStarConfig::gamma_lower_bound(2, FREE_VOLUME), 1.0);

    [
        timed("RRT", || {
            let mut rrt = Rrt::new(problem(), RrtConfig::default()).expect("start is free");
            (rrt.solve(20_000, &mut rng()), rrt.node_count())
        }),
        timed("RRT-Connect", || {
            let mut connect = RrtConnect::new(problem()).expect("start and goal are free");
            (connect.solve(20_000, &mut rng()), connect.node_count())
        }),
        timed("RRT*", || {
            let mut planner = RrtStar::with_nearest_neighbors(problem(), star, KdTree::default())
                .expect("start is free");
            planner.run(5000, &mut rng());
            (planner.best_path(), planner.node_count())
        }),
        timed("Informed RRT*", || {
            let mut planner =
                RrtStar::with_nearest_neighbors(problem(), star.informed(), KdTree::default())
                    .expect("start is free");
            planner.run(5000, &mut rng());
            (planner.best_path(), planner.node_count())
        }),
        timed("RRT#", || {
            let mut planner = RrtSharp::with_nearest_neighbors(problem(), star, KdTree::default())
                .expect("start is free");
            planner.run(5000, &mut rng());
            (planner.best_path(), planner.node_count())
        }),
    ]
}

/// The multi-query roadmaps PRM* and SPARS2 on the slalom.
fn roadmap_planners(seed: u64) -> [Outcome; 2] {
    let rng = || Pcg64Mcg::seed_from_u64(seed);
    let world = problem();
    let (start, goal) = (world.start, world.goal);
    let sampler = world.sampler;

    [
        timed("PRM*", || {
            let config = PrmConfig::star(1.5 * PrmConfig::gamma_lower_bound(2, FREE_VOLUME));
            let mut prm =
                Prm::with_nearest_neighbors(Euclidean, sampler, Walls, config, KdTree::default())
                    .expect("valid config");
            prm.grow(3000, &mut rng());
            (
                prm.query(&start, &goal).expect("start is free"),
                prm.node_count(),
            )
        }),
        timed("SPARS2", || {
            let mut config = Spars2Config::new(2.0, 0.05);
            config.max_failures = 1000;
            let mut spars = Spars2::with_nearest_neighbors(
                Euclidean,
                sampler,
                Walls,
                config,
                KdTree::default(),
            )
            .expect("valid config");
            spars.construct(200_000, &mut rng());
            (
                spars.query(&start, &goal).expect("start is free"),
                spars.node_count(),
            )
        }),
    ]
}

/// The batch planners FMT*, BFMT* and BIT* on the slalom.
fn batch_planners(seed: u64) -> [Outcome; 3] {
    let rng = || Pcg64Mcg::seed_from_u64(seed);
    let world = problem();
    let (start, goal) = (world.start, world.goal);
    let sampler = world.sampler;

    [
        timed("FMT*", || {
            let config = FmtStarConfig {
                gamma: 1.5 * FmtStarConfig::gamma_lower_bound(2, FREE_VOLUME),
            };
            let mut fmt = FmtStar::with_nearest_neighbors(
                Euclidean,
                sampler,
                Walls,
                config,
                KdTree::default(),
            )
            .expect("valid config");
            (
                fmt.plan(&start, &goal, 3000, &mut rng())
                    .expect("start is free"),
                3000,
            )
        }),
        timed("BFMT*", || {
            let config = BfmtStarConfig::new(BfmtStarConfig::gamma_lower_bound(2, FREE_VOLUME));
            let mut bfmt = BfmtStar::with_nearest_neighbors(
                Euclidean,
                sampler,
                Walls,
                config,
                KdTree::default(),
            )
            .expect("valid config");
            (
                bfmt.plan(&start, &goal, 3000, &mut rng())
                    .expect("valid endpoints"),
                3000,
            )
        }),
        timed("BIT*", || {
            let config = BitStarConfig::new(1.5 * BitStarConfig::gamma_lower_bound(2, FREE_VOLUME));
            let mut bit = BitStar::with_nearest_neighbors(
                Euclidean,
                sampler,
                Walls,
                start,
                goal.clone(),
                config,
                KdTree::default(),
            )
            .expect("start is free");
            bit.run(20_000, &mut rng());
            (bit.best_path(), bit.vertex_count())
        }),
    ]
}

/// Renders the maps of `outcomes` side by side: walls (`#`), the path (`*`), start (`S`) and
/// goal (`G`), north up, with the planner name above each map.
fn draw_row(outcomes: &[Outcome]) -> String {
    let maps: Vec<Vec<String>> = outcomes
        .iter()
        .map(|outcome| draw(outcome.path.as_ref()))
        .collect();
    let mut out = String::new();

    for outcome in outcomes {
        write!(out, "{:<width$}  ", outcome.name, width = MAP_SIZE + 2)
            .expect("writing to a String");
    }

    out.push('\n');

    for line in 0..MAP_SIZE + 2 {
        for map in &maps {
            out.push_str(&map[line]);
            out.push_str("  ");
        }

        out.push('\n');
    }

    out.push('\n');
    out
}

/// The lines of one bordered map with the walls and `path`.
fn draw(path: Option<&Path<[f64; 2]>>) -> Vec<String> {
    let mut grid = [[' '; MAP_SIZE]; MAP_SIZE];

    for (row, line) in grid.iter_mut().enumerate() {
        for (col, cell) in line.iter_mut().enumerate() {
            if Walls.check_state(&cell_center(row, col)) == Validity::Invalid {
                *cell = '#';
            }
        }
    }

    for pair in path.iter().flat_map(|path| path.states.windows(2)) {
        plot_segment(&mut grid, pair[0], pair[1], '*');
    }

    for (point, mark) in [([1.0, 1.0], 'S'), ([9.0, 9.0], 'G')] {
        let (row, col) = cell_of(point);
        grid[row][col] = mark;
    }

    let border = format!("+{}+", "-".repeat(MAP_SIZE));
    let mut lines = vec![border.clone()];
    lines.extend(
        grid.iter()
            .map(|line| format!("|{}|", line.iter().collect::<String>())),
    );
    lines.push(border);

    lines
}

/// Marks the cells along the segment from `a` to `b`.
fn plot_segment(grid: &mut [[char; MAP_SIZE]; MAP_SIZE], a: [f64; 2], b: [f64; 2], mark: char) {
    for step in 0..=50 {
        let t = f64::from(step) / 50.0;
        let point = [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
        let (row, col) = cell_of(point);
        grid[row][col] = mark;
    }
}

/// The world coordinates of the center of a map cell; row 0 is the top.
fn cell_center(row: usize, col: usize) -> [f64; 2] {
    let world = |index: usize| {
        let index = f64::from(u8::try_from(index).expect("map indices fit in u8"));
        (index + 0.5) / CELLS_PER_UNIT
    };

    [world(col), 10.0 - world(row)]
}

/// The map cell containing a world point, clamped to the map.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "flooring a clamped, non-negative cell coordinate is the intent"
)]
fn cell_of(point: [f64; 2]) -> (usize, usize) {
    let to_index = |v: f64| ((v * CELLS_PER_UNIT).max(0.0) as usize).min(MAP_SIZE - 1);

    (MAP_SIZE - 1 - to_index(point[1]), to_index(point[0]))
}
