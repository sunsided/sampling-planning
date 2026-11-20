//! Benchmarks for nearest-neighbor queries, the hot path of every tree planner.

use std::hint::black_box;

use criterion::Criterion;
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{Euclidean, KdTree, LinearScan, NearestNeighbors, Sampler, UniformBox};

/// Number of stored states.
const SIZE: usize = 10_000;

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    bench_nearest(&mut criterion);
    criterion.final_summary();
}

/// Compares [`LinearScan`] and [`KdTree`] nearest queries over [`SIZE`] random 3D states.
fn bench_nearest(criterion: &mut Criterion) {
    let mut sampler = UniformBox::new([0.0; 3], [10.0; 3]);
    let mut rng = Pcg64Mcg::seed_from_u64(1);
    let mut linear = LinearScan::default();
    let mut kd = KdTree::default();

    for id in 0..SIZE {
        let state = sampler.sample(&mut rng);
        NearestNeighbors::<Euclidean<3>>::insert(&mut linear, id, state);
        kd.insert(id, state);
    }

    let queries: Vec<[f64; 3]> = (0..256).map(|_| sampler.sample(&mut rng)).collect();
    let mut group = criterion.benchmark_group("nearest_10k_3d");

    group.bench_function("linear_scan", |b| {
        b.iter(|| {
            for query in &queries {
                black_box(linear.nearest(&Euclidean::<3>, black_box(query)));
            }
        });
    });
    group.bench_function("kd_tree", |b| {
        b.iter(|| {
            for query in &queries {
                black_box(kd.nearest(&Euclidean::<3>, black_box(query)));
            }
        });
    });
    group.finish();
}
