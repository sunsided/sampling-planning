//! Benchmarks for the SE(2), SE(3) and car-like distances, which nearest-neighbor scans call
//! once per node.

use std::hint::black_box;

use criterion::Criterion;
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    Dubins, ReedsShepp, Sampler, Se2, Se2State, Se3, StateSpace, UniformSe2, UniformSe3,
};

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    bench_distances(&mut criterion);
    bench_se3(&mut criterion);
    criterion.final_summary();
}

/// Times one distance evaluation per pair over 256 random pose pairs in each SE(2) space.
fn bench_distances(criterion: &mut Criterion) {
    let mut sampler = UniformSe2::new([0.0, 0.0], [10.0, 10.0]);
    let mut rng = Pcg64Mcg::seed_from_u64(1);
    let pairs: Vec<(Se2State, Se2State)> = (0..256)
        .map(|_| (sampler.sample(&mut rng), sampler.sample(&mut rng)))
        .collect();
    let mut group = criterion.benchmark_group("distance_256_pairs");

    group.bench_function("se2", |b| b.iter(|| sum(&Se2::new(0.5), &pairs)));
    group.bench_function("dubins", |b| b.iter(|| sum(&Dubins::new(1.0), &pairs)));
    group.bench_function("reeds_shepp", |b| {
        b.iter(|| sum(&ReedsShepp::new(1.0), &pairs));
    });
    group.finish();
}

/// Times one SE(3) distance and one interpolation per pair over 256 random spatial pose pairs.
fn bench_se3(criterion: &mut Criterion) {
    let mut sampler = UniformSe3::new([0.0; 3], [10.0; 3]);
    let mut rng = Pcg64Mcg::seed_from_u64(2);
    let pairs: Vec<_> = (0..256)
        .map(|_| (sampler.sample(&mut rng), sampler.sample(&mut rng)))
        .collect();
    let space = Se3::new(0.5);
    let mut group = criterion.benchmark_group("se3_256_pairs");

    group.bench_function("distance", |b| {
        b.iter(|| {
            pairs
                .iter()
                .map(|(x, y)| space.distance(black_box(x), black_box(y)))
                .sum::<f64>()
        });
    });
    group.bench_function("interpolate", |b| {
        b.iter(|| {
            pairs
                .iter()
                .map(|(x, y)| space.interpolate(black_box(x), black_box(y), 0.3).position[0])
                .sum::<f64>()
        });
    });
    group.finish();
}

/// The sum of the distances of all `pairs` in `space`.
fn sum<Sp: StateSpace<State = Se2State>>(space: &Sp, pairs: &[(Se2State, Se2State)]) -> f64 {
    pairs
        .iter()
        .map(|(a, b)| space.distance(black_box(a), black_box(b)))
        .sum()
}
