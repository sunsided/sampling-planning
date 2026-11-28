//! Benchmarks PRM construction with a sequential and a multi-threaded validity checker
//! (feature `parallel`).

use std::hint::black_box;
use std::num::NonZeroUsize;

use criterion::Criterion;
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    Euclidean, ParallelChecker, UniformBox, Validity, ValidityChecker, discretized_motion,
    roadmaps::{Prm, PrmConfig},
};

/// A slalom world checked at a fine resolution, so that checks dominate like with a real
/// collision checker.
#[derive(Debug, Clone, Copy)]
struct Slalom;

impl ValidityChecker<Euclidean<2>> for Slalom {
    fn check_state(&self, s: &[f64; 2]) -> Validity {
        let wall = ((4.0..=5.0).contains(&s[0]) && s[1] < 7.0)
            || ((6.0..=7.0).contains(&s[0]) && s[1] > 3.0);

        if wall {
            Validity::Invalid
        } else {
            Validity::Valid
        }
    }

    fn check_motion(&self, space: &Euclidean<2>, from: &[f64; 2], to: &[f64; 2]) -> Validity {
        discretized_motion(space, from, to, 0.0005, |state| self.check_state(state))
    }
}

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    bench_construction(&mut criterion);
    criterion.final_summary();
}

/// Grows a 500-sample PRM* roadmap over `checker` and returns its edge count.
fn grow<C: ValidityChecker<Euclidean<2>>>(checker: C) -> usize {
    let config = PrmConfig::star(1.5 * PrmConfig::gamma_lower_bound(2, 100.0));
    let sampler = UniformBox::new([0.0; 2], [10.0; 2]);
    let mut prm = Prm::new(Euclidean::<2>, sampler, checker, config).expect("valid config");
    prm.grow(500, &mut Pcg64Mcg::seed_from_u64(1));

    prm.edge_count()
}

/// Times construction with one thread and with `available_parallelism` threads.
fn bench_construction(criterion: &mut Criterion) {
    let threads = std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN);
    let mut group = criterion.benchmark_group("prm_star_construction_500");
    group.sample_size(10);

    group.bench_function("sequential", |b| b.iter(|| grow(black_box(Slalom))));
    group.bench_function(format!("parallel_{threads}"), |b| {
        b.iter(|| grow(ParallelChecker::new(black_box(Slalom), threads, 64)));
    });
    group.finish();
}
