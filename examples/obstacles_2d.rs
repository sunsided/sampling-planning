//! Plans through a 2D world with two walls using RRT and RRT*, then draws both paths.
//!
//! Run with `cargo run --release --example obstacles_2d [seed]`.

use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    Euclidean, GoalRegion, Path, Problem, StraightLine, UniformBox, Validity, ValidityChecker,
    discretized_motion,
    trees::{Rrt, RrtConfig, RrtStar, RrtStarConfig},
};

/// An axis-aligned box given by its lower and upper corners.
type Rect = ([f64; 2], [f64; 2]);

/// Two walls in a 10 x 10 world, forcing a slalom from the bottom left to the top right.
const WALLS: [Rect; 2] = [([4.0, 0.0], [5.0, 7.0]), ([6.0, 3.0], [7.0, 10.0])];

/// Map width and height in characters.
const MAP_SIZE: usize = 40;

/// Map cells per world unit: [`MAP_SIZE`] cells across the 10 unit wide world.
const CELLS_PER_UNIT: f64 = 4.0;

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
fn problem() -> Problem<Euclidean<2>, UniformBox<2>, Walls, StraightLine> {
    Problem {
        space: Euclidean::<2>,
        sampler: UniformBox::new([0.0, 0.0], [10.0, 10.0]),
        checker: Walls,
        steering: StraightLine::new(1.0),
        start: [1.0, 1.0],
        goal: GoalRegion::new([9.0, 9.0], 0.5),
    }
}

fn main() {
    let seed = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(42);

    let mut rrt = Rrt::new(problem(), RrtConfig::default()).expect("start is free");
    let rrt_path = rrt.solve(20_000, &mut Pcg64Mcg::seed_from_u64(seed));

    let gamma = 1.5 * RrtStarConfig::gamma_lower_bound(2, 100.0);
    let mut star = RrtStar::new(problem(), RrtStarConfig::new(gamma)).expect("start is free");
    star.run(5000, &mut Pcg64Mcg::seed_from_u64(seed));
    let star_path = star.best_path();

    report("RRT", rrt.node_count(), rrt_path.as_ref());
    report("RRT*", star.node_count(), star_path.as_ref());
    print!("{}", draw(rrt_path.as_ref(), star_path.as_ref()));
}

/// Prints the cost and waypoints of a planner's result.
fn report(name: &str, nodes: usize, path: Option<&Path<[f64; 2]>>) {
    match path {
        Some(path) => {
            println!(
                "{name}: cost {:.3} with {} waypoints ({nodes} nodes)",
                path.cost,
                path.states.len()
            );
            for [x, y] in &path.states {
                println!("  ({x:.2}, {y:.2})");
            }
        }
        None => println!("{name}: no path ({nodes} nodes)"),
    }
}

/// Renders walls (`#`), the RRT path (`o`) and the RRT* path (`*`) as ASCII art, north up.
fn draw(rrt: Option<&Path<[f64; 2]>>, star: Option<&Path<[f64; 2]>>) -> String {
    let mut grid = [[' '; MAP_SIZE]; MAP_SIZE];

    for (row, line) in grid.iter_mut().enumerate() {
        for (col, cell) in line.iter_mut().enumerate() {
            if Walls.check_state(&cell_center(row, col)) == Validity::Invalid {
                *cell = '#';
            }
        }
    }

    for (path, mark) in [(rrt, 'o'), (star, '*')] {
        for pair in path.iter().flat_map(|path| path.states.windows(2)) {
            plot_segment(&mut grid, pair[0], pair[1], mark);
        }
    }

    let border = format!("+{}+\n", "-".repeat(MAP_SIZE));
    let mut out = border.clone();

    for line in &grid {
        out.push('|');
        out.extend(line);
        out.push_str("|\n");
    }

    out.push_str(&border);
    out
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
