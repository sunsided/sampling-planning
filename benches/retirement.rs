//! Benchmarks RRT to the first solution on a cluttered map and out of a bug trap, with and
//! without vertex retirement.

use std::hint::black_box;
use std::num::NonZeroUsize;

use criterion::Criterion;
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    Euclidean, GoalRegion, KdTree, LinearScan, Problem, RemovableNeighbors, StraightLine,
    UniformBox, Validity, ValidityChecker, discretized_motion,
    trees::{Rrt, RrtConfig},
};

/// The iteration budget per run; generous enough that every seed solves.
const BUDGET: usize = 200_000;

/// The seeds each measurement runs over.
const SEEDS: u64 = 8;

/// The maps every measurement runs on.
const MAPS: [Map; 2] = [Map::Clutter, Map::BugTrap];

/// The two 20 x 20 maps the benchmarks compare.
#[derive(Debug, Clone, Copy)]
enum Map {
    /// A staggered grid of 1.2 x 1.2 blocks on a 2-unit lattice: cluttered, but without dead
    /// ends.
    Clutter,
    /// A bug trap around the start: a box from `(2, 2)` to `(10, 10)` with walls 0.4 thick,
    /// open only on the left at `4 <= y <= 5`, while the goal lies up and to the right.
    /// Goal-biased extensions keep hitting the inner walls.
    BugTrap,
}

impl Map {
    /// Whether `state` hits an obstacle of this map.
    fn blocked(self, [x, y]: [f64; 2]) -> bool {
        match self {
            Self::Clutter => {
                let shift = if (y / 2.0).floor() % 2.0 == 0.0 {
                    0.0
                } else {
                    1.0
                };
                let (cx, cy) = ((x + shift) % 2.0, y % 2.0);

                (0.4..1.6).contains(&cx) && (0.4..1.6).contains(&cy) && y > 1.0 && y < 19.0
            }
            Self::BugTrap => {
                let span = (2.0..=10.0).contains(&x) && (2.0..=10.0).contains(&y);
                let in_wall = |c: f64| (2.0..=2.4).contains(&c) || (9.6..=10.0).contains(&c);
                let opening = (2.0..=2.4).contains(&x) && (4.0..=5.0).contains(&y);

                span && (in_wall(x) || in_wall(y)) && !opening
            }
        }
    }

    /// The start state: the bottom-left corner, or inside the trap.
    const fn start(self) -> [f64; 2] {
        match self {
            Self::Clutter => [0.2, 0.2],
            Self::BugTrap => [6.0, 6.0],
        }
    }

    /// The name used in benchmark ids.
    const fn name(self) -> &'static str {
        match self {
            Self::Clutter => "clutter",
            Self::BugTrap => "bug_trap",
        }
    }
}

impl ValidityChecker<Euclidean<2>> for Map {
    fn check_state(&self, state: &[f64; 2]) -> Validity {
        let inside = state.iter().all(|c| (0.0..=20.0).contains(c));

        if inside && !self.blocked(*state) {
            Validity::Valid
        } else {
            Validity::Invalid
        }
    }

    fn check_motion(&self, space: &Euclidean<2>, from: &[f64; 2], to: &[f64; 2]) -> Validity {
        discretized_motion(space, from, to, 0.05, |state| self.check_state(state))
    }
}

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    report_iterations();
    bench_first_solution(&mut criterion);
    criterion.final_summary();
}

/// The problem on `map`: from its start to the top-right corner.
fn problem(map: Map) -> Problem<Euclidean<2>, UniformBox<2>, Map, StraightLine> {
    Problem {
        space: Euclidean::<2>,
        sampler: UniformBox::new([0.0; 2], [20.0; 2]),
        checker: map,
        steering: StraightLine::new(0.5),
        start: map.start(),
        goal: GoalRegion::new([19.8, 19.8], 0.3),
    }
}

/// Runs RRT on `map` over the empty index `nn` to the first solution with `seed`; returns
/// the iterations used and the retired vertex count.
fn first_solution<N>(map: Map, nn: N, seed: u64, retire: Option<NonZeroUsize>) -> (usize, usize)
where
    N: RemovableNeighbors<Euclidean<2>> + Clone,
{
    let planner =
        Rrt::with_nearest_neighbors(problem(map), RrtConfig::default(), nn).expect("valid problem");
    let mut planner = match retire {
        Some(cap) => planner.with_retirement(cap),
        None => planner,
    };
    let mut rng = Pcg64Mcg::seed_from_u64(seed);
    let mut iterations = 0;

    while iterations < BUDGET && planner.best_path().is_none() {
        planner.step(&mut rng);
        iterations += 1;
    }

    (iterations, planner.retired_count())
}

/// The retirement settings compared: off, and a few caps.
fn settings() -> [(&'static str, Option<NonZeroUsize>); 3] {
    [
        ("off", None),
        ("cap_3", NonZeroUsize::new(3)),
        ("cap_10", NonZeroUsize::new(10)),
    ]
}

/// Prints the mean iterations to the first solution and the mean retired count per map and
/// setting.
fn report_iterations() {
    for map in MAPS {
        for (name, retire) in settings() {
            report(map, name, retire);
        }
    }
}

/// Prints the means of one map and setting.
fn report(map: Map, name: &str, retire: Option<NonZeroUsize>) {
    {
        let runs: Vec<_> = (0..SEEDS)
            .map(|seed| first_solution(map, LinearScan::default(), seed, retire))
            .collect();
        let count = f64::from(u32::try_from(runs.len()).expect("few seeds"));
        let mean = |pick: fn(&(usize, usize)) -> usize| {
            runs.iter()
                .map(|run| f64::from(u32::try_from(pick(run)).expect("bounded")))
                .sum::<f64>()
                / count
        };

        eprintln!(
            "{} retirement {name}: {:.0} iterations to the first solution, {:.0} vertices retired",
            map.name(),
            mean(|run| run.0),
            mean(|run| run.1),
        );
    }
}

/// Times the first solution over [`SEEDS`] seeds per map, retirement setting and index.
///
/// Both indices return the same ids, so the iteration counts match; the times differ because
/// a k-d tree query near many retired vertices has to search further for a live one.
fn bench_first_solution(criterion: &mut Criterion) {
    for map in MAPS {
        let mut group = criterion.benchmark_group(format!("rrt_{}_first_solution", map.name()));
        group.sample_size(10);

        for (name, retire) in settings() {
            group.bench_function(format!("linear_{name}"), |b| {
                b.iter(|| total(map, LinearScan::default, retire));
            });
            group.bench_function(format!("kd_{name}"), |b| {
                b.iter(|| total(map, KdTree::default, retire));
            });
        }

        group.finish();
    }
}

/// The iterations to the first solution summed over [`SEEDS`] seeds, each over a fresh index.
fn total<N>(map: Map, index: fn() -> N, retire: Option<NonZeroUsize>) -> usize
where
    N: RemovableNeighbors<Euclidean<2>> + Clone,
{
    (0..SEEDS)
        .map(|seed| first_solution(map, index(), black_box(seed), retire).0)
        .sum()
}
