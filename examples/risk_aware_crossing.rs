//! A delivery robot crossing a plaza with people whose positions it only knows roughly:
//! RRT* minimizing collision risk instead of path length.
//!
//! The robot's perception reports three people as Gaussian estimates: a mean position and a
//! covariance that combines the sensor's and the robot's own position uncertainty. Treating
//! them as hard obstacles would need a safety margin picked by hand; treating them as free
//! space drives right through them. The [`RiskDensity`] objective of Paiola et al. instead
//! prices each motion by the probability mass of the obstacles along it (in closed form),
//! plus a small length term, and RRT* minimizes that sum. A fountain stays a hard obstacle
//! for the validity checker.
//!
//! The same RRT* with the default [`PathLength`] objective plans the shortest path for
//! comparison.
//!
//! Run with `cargo run --release --example risk_aware_crossing [seed]`.

use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    Euclidean, GaussianObstacle, GoalRegion, KdTree, Path, PathLength, Problem, RiskDensity,
    StraightLine, UniformBox, Validity, ValidityChecker, discretized_motion,
    trees::{RrtStar, RrtStarConfig},
};

/// Plaza width and depth.
const PLAZA: [f64; 2] = [20.0, 10.0];

/// The fountain, a hard obstacle: lower and upper corner.
const FOUNTAIN: ([f64; 2], [f64; 2]) = ([12.0, 6.0], [15.0, 9.0]);

/// Map cells per world unit.
const CELLS_PER_UNIT: f64 = 2.0;

/// The weight of path length against risk; small, so the robot detours to cut risk.
const LENGTH_WEIGHT: f64 = 0.01;

/// The plaza with its fountain.
#[derive(Debug, Clone, Copy)]
struct Plaza;

impl Plaza {
    /// Whether `point` is on the plaza and off the fountain.
    fn check(point: &[f64; 2]) -> Validity {
        let on_plaza = (0..2).all(|i| (0.0..=PLAZA[i]).contains(&point[i]));
        let in_fountain = (0..2).all(|i| (FOUNTAIN.0[i]..=FOUNTAIN.1[i]).contains(&point[i]));

        if on_plaza && !in_fountain {
            Validity::Valid
        } else {
            Validity::Invalid
        }
    }
}

impl ValidityChecker<Euclidean<2>> for Plaza {
    fn check_state(&self, point: &[f64; 2]) -> Validity {
        Self::check(point)
    }

    fn check_motion(&self, space: &Euclidean<2>, from: &[f64; 2], to: &[f64; 2]) -> Validity {
        discretized_motion(space, from, to, 0.05, Self::check)
    }
}

/// The three people as the robot's perception reports them: mean, covariance, scale.
fn people() -> Vec<GaussianObstacle<2>> {
    let estimates = [
        // Standing still, well localized.
        ([6.0, 5.0], [[0.4, 0.0], [0.0, 0.4]]),
        // Walking along y: its estimate is stretched along its heading.
        ([10.0, 3.5], [[0.3, 0.0], [0.0, 1.5]]),
        // Far from the sensor, so poorly localized in every direction.
        ([16.0, 4.0], [[1.0, 0.3], [0.3, 0.8]]),
    ];

    estimates
        .into_iter()
        .map(|(mean, covariance)| {
            GaussianObstacle::new(mean, covariance, 1.0).expect("positive definite covariance")
        })
        .collect()
}

/// From the left edge of the plaza to the right edge.
fn problem() -> Problem<Euclidean<2>, UniformBox<2>, Plaza, StraightLine> {
    Problem {
        space: Euclidean,
        sampler: UniformBox::new([0.0, 0.0], PLAZA),
        checker: Plaza,
        steering: StraightLine::new(1.0),
        start: [1.0, 4.5],
        goal: GoalRegion::new([19.0, 4.5], 0.5),
    }
}

fn main() {
    let seed = std::env::args()
        .nth(1)
        .and_then(|arg| arg.parse().ok())
        .unwrap_or(1);
    let config = RrtStarConfig::new(1.5 * RrtStarConfig::gamma_lower_bound(2, 191.0), 1.0);
    let risk = RiskDensity::new(people(), LENGTH_WEIGHT);

    // The risk objective bounds its cost below by `LENGTH_WEIGHT` times the length, so
    // informed sampling stays admissible; with so small a weight it rarely prunes anything.
    let mut careful = RrtStar::with_objective(
        problem(),
        config.informed(),
        KdTree::default(),
        risk.clone(),
    )
    .expect("start is free");
    let mut direct = RrtStar::with_objective(problem(), config, KdTree::default(), PathLength)
        .expect("start is free");
    careful.run(6000, &mut Pcg64Mcg::seed_from_u64(seed));
    direct.run(6000, &mut Pcg64Mcg::seed_from_u64(seed));

    let (Some(safe), Some(short)) = (careful.best_path(), direct.best_path()) else {
        println!("no path found; try more iterations or another seed");
        return;
    };

    println!("{:<14} {:>8} {:>10}", "objective", "length", "risk");
    for (name, path) in [("path length", &short), ("risk density", &safe)] {
        println!(
            "{name:<14} {:>8.2} {:>10.4}",
            length(path),
            path_risk(&risk, path)
        );
    }

    print!("{}", draw(&short, &safe));
}

/// The Euclidean length of `path`.
fn length(path: &Path<[f64; 2]>) -> f64 {
    path.states
        .windows(2)
        .map(|pair| (pair[1][0] - pair[0][0]).hypot(pair[1][1] - pair[0][1]))
        .sum()
}

/// The collision risk along `path`, without the length term.
fn path_risk(risk: &RiskDensity<2>, path: &Path<[f64; 2]>) -> f64 {
    path.states
        .windows(2)
        .map(|pair| risk.risk(&pair[0], &pair[1]))
        .sum()
}

/// The plaza with the fountain (`#`), the people's means (`@`), the shortest path (`.`) and
/// the risk-aware path (`o`), north up.
fn draw(short: &Path<[f64; 2]>, safe: &Path<[f64; 2]>) -> String {
    let (width, height) = (cell(PLAZA[0]), cell(PLAZA[1]));
    let mut grid = vec![vec![' '; width]; height];

    for line in &mut grid[cell(FOUNTAIN.0[1])..cell(FOUNTAIN.1[1])] {
        line[cell(FOUNTAIN.0[0])..cell(FOUNTAIN.1[0])].fill('#');
    }

    for (path, mark) in [(short, '.'), (safe, 'o')] {
        for pair in path.states.windows(2) {
            for step in 0..=20 {
                let t = f64::from(step) / 20.0;
                let x = pair[0][0] + (pair[1][0] - pair[0][0]) * t;
                let y = pair[0][1] + (pair[1][1] - pair[0][1]) * t;
                grid[cell(y).min(height - 1)][cell(x).min(width - 1)] = mark;
            }
        }
    }

    for mean in [[6.0, 5.0], [10.0, 3.5], [16.0, 4.0]] {
        grid[cell(mean[1])][cell(mean[0])] = '@';
    }

    let border = format!("+{}+\n", "-".repeat(width));
    let mut out = border.clone();
    for line in grid.iter().rev() {
        out.push('|');
        out.extend(line);
        out.push_str("|\n");
    }

    out.push_str(&border);
    out
}

/// The map cell index of a non-negative world coordinate.
#[expect(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "flooring a non-negative, bounded world coordinate is the intent"
)]
fn cell(value: f64) -> usize {
    (value.max(0.0) * CELLS_PER_UNIT) as usize
}
