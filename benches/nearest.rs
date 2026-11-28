//! Benchmarks for nearest-neighbor queries, the hot path of every tree planner.

use std::hint::black_box;

use criterion::Criterion;
use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    Euclidean, KdTree, LinearScan, NearestNeighbors, Sampler, Se3, Se3State, UniformBox,
    UniformSe3, VpTree,
};

/// Number of stored states.
const SIZE: usize = 10_000;

/// Runs every benchmark in this file with criterion's command-line flags.
fn main() {
    let mut criterion = Criterion::default().configure_from_args();

    bench_nearest(&mut criterion);
    bench_se3(&mut criterion);
    criterion.final_summary();
}

/// Compares [`LinearScan`], [`KdTree`] and [`VpTree`] nearest and 10-nearest queries over
/// [`SIZE`] random 3D states.
fn bench_nearest(criterion: &mut Criterion) {
    let mut sampler = UniformBox::new([0.0; 3], [10.0; 3]);
    let mut rng = Pcg64Mcg::seed_from_u64(1);
    let mut linear = LinearScan::default();
    let mut kd = KdTree::default();
    let mut vp = VpTree::<Euclidean<3>>::default();

    for id in 0..SIZE {
        let state = sampler.sample(&mut rng);
        NearestNeighbors::<Euclidean<3>>::insert(&mut linear, id, state);
        kd.insert(id, state);
        vp.insert(id, state);
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
    group.bench_function("vp_tree", |b| {
        b.iter(|| {
            for query in &queries {
                black_box(vp.nearest(&Euclidean::<3>, black_box(query)));
            }
        });
    });
    group.bench_function("linear_scan_k10", |b| {
        let mut out = Vec::new();
        b.iter(|| {
            for query in &queries {
                linear.k_nearest(&Euclidean::<3>, black_box(query), 10, &mut out);
                black_box(&out);
            }
        });
    });
    group.bench_function("kd_tree_k10", |b| {
        let mut out = Vec::new();
        b.iter(|| {
            for query in &queries {
                kd.k_nearest(&Euclidean::<3>, black_box(query), 10, &mut out);
                black_box(&out);
            }
        });
    });
    group.finish();
}

/// Compares [`LinearScan`] and [`VpTree`] nearest and radius queries over [`SIZE`] random
/// SE(3) poses, a space a k-d tree cannot index.
fn bench_se3(criterion: &mut Criterion) {
    let space = Se3::new(0.5);
    let mut sampler = UniformSe3::new([0.0; 3], [10.0; 3]);
    let mut rng = Pcg64Mcg::seed_from_u64(2);
    let mut linear = LinearScan::default();
    let mut vp = VpTree::new(space);

    for id in 0..SIZE {
        let state = sampler.sample(&mut rng);
        NearestNeighbors::<Se3>::insert(&mut linear, id, state);
        vp.insert(id, state);
    }

    let queries: Vec<Se3State> = (0..256).map(|_| sampler.sample(&mut rng)).collect();
    let mut group = criterion.benchmark_group("nearest_10k_se3");

    group.bench_function("linear_scan", |b| {
        b.iter(|| {
            for query in &queries {
                black_box(linear.nearest(&space, black_box(query)));
            }
        });
    });
    group.bench_function("vp_tree", |b| {
        b.iter(|| {
            for query in &queries {
                black_box(vp.nearest(&space, black_box(query)));
            }
        });
    });
    group.bench_function("linear_scan_radius", |b| {
        let mut out = Vec::new();
        b.iter(|| {
            for query in &queries {
                linear.within_radius(&space, black_box(query), 1.5, &mut out);
                black_box(&out);
            }
        });
    });
    group.bench_function("vp_tree_radius", |b| {
        let mut out = Vec::new();
        b.iter(|| {
            for query in &queries {
                vp.within_radius(&space, black_box(query), 1.5, &mut out);
                black_box(&out);
            }
        });
    });
    group.finish();
}
