# sampling-planning: motion planning for Rust

[![crates.io](https://img.shields.io/crates/v/sampling-planning.svg)](https://crates.io/crates/sampling-planning)
[![docs.rs](https://img.shields.io/docsrs/sampling-planning)](https://docs.rs/sampling-planning)
[![CI](https://github.com/sunsided/sampling-planning/actions/workflows/ci.yml/badge.svg)](https://github.com/sunsided/sampling-planning/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/sunsided/sampling-planning/graph/badge.svg)](https://codecov.io/gh/sunsided/sampling-planning)
[![MSRV](https://img.shields.io/crates/msrv/sampling-planning)](https://crates.io/crates/sampling-planning)
[![License](https://img.shields.io/crates/l/sampling-planning)](https://github.com/sunsided/sampling-planning#license)
[![unsafe forbidden](https://img.shields.io/badge/unsafe-forbidden-success.svg)](https://github.com/rust-secure-code/safety-dance/)

Sampling-based motion planners for continuous configuration spaces: RRT, RRT*, RRT#, RRT-Connect,
PRM, PRM*, SPARS, SPARS2, FMT*, BFMT*, BIT*, ABIT*, AIT*, EIT*, SST, SST* and KPIECE, together with the state
spaces, samplers, nearest-neighbor structures and path shortcutting they build on.

This is a **motion planning** crate. It is named `sampling-planning` because the crate name `motion-planning` was already taken on crates.io.

The planners share one small set of traits (a state space, a sampler and a validity checker, plus a
steering function for the tree planners and a `Propagator` for the kinodynamic ones), so the
collision checker and the space are written once and work with every planner. The crate is `#![no_std]` with `alloc`, forbids `unsafe`, takes its
randomness from a caller-provided `rand_core` RNG and does its float math through `libm`: a seeded
run gives the same path on every platform.

Part of the [planning and search crates](https://sunsided.github.io/planning-and-search/); see also this crate's [overview page](https://sunsided.github.io/planning-and-search/crates/sampling-planning/).

## Who this is for

- Robotics engineers who need a path for an arm, a mobile base or a car-like vehicle through a
  continuous space with obstacles, and who bring their own collision checker.
- Vehicles with a turning radius: the Dubins and Reeds-Shepp spaces plan along the actual
  shortest car paths, not along straight lines that are fixed up afterwards.
- Systems that can only be forward-simulated (no steering function, just "apply this control for
  that long"): SST, SST* and KPIECE work over a `Propagator` instead.
- Anyone comparing planners on the same problem; the bundled example runs eleven of them on one
  world and draws the results.
- Embedded and other `no_std` targets with an allocator, and test setups that need reproducible runs.
- For discrete graphs and grids, [heuristic-search](https://github.com/sunsided/heuristic-search)
  is the better fit; for many robots sharing one map, [mapf-search](https://github.com/sunsided/mapf-search);
  for dynamics with bounded disturbances where the plan has to hold for every reachable state,
  [robust-planning](https://github.com/sunsided/robust-planning).

## Installation

```toml
[dependencies]
sampling-planning = "0.1"
```

## Features

- Tree planners: RRT, RRT* (radius or k-nearest) with optional informed sampling (Informed RRT*, with direct
  prolate-hyperspheroid sampling in Euclidean spaces and, through a Loewner-order lower bound of the
  metric tensor, under configuration-dependent Riemannian metrics), RRT-Connect (bidirectional), RRT# (RRG with LPA*-style replanning). RRT, RRT* and
  RRT-Connect can retire vertices that keep failing to extend from their nearest-neighbor index.
  With feature `parallel`, `grow_forest` grows several RRT* trees on threads and merges them.
  RRT* minimizes any additive cost objective, such as the hybrid risk-density objective over
  Gaussian uncertain obstacles.
- Roadmaps: PRM (fixed radius or fixed k), PRM* (shrinking radius or `k ln n` nearest), lazy
  PRM, SPARS (sparse roadmap spanner over a dense PRM* graph), SPARS2 (without one);
  multi-query, with Dijkstra queries on a reusable roadmap, also through a shared reference with a
  reusable query workspace and, with feature `parallel`, in batches on several threads; with
  feature `parallel` a roadmap is also built on several threads.
- Batch-informed planners: BIT* (radius or k-nearest, informed batches, pruning), ABIT*
  (inflated and truncated BIT* searches), AIT* (BIT*-style forward search guided by a lazy LPA*
  reverse search that is repaired on every invalid edge) and EIT* (effort-guided forward search
  over a sparsely checked reverse search, few collision checks to a first solution).
- Batch planners: FMT* (fast marching tree) and BFMT* (bidirectional, optimal or first-meeting
  termination).
- Kinodynamic planners: SST and SST* over a `Propagator` (forward simulation) trait, with
  inactive leaves freed; single- or multi-level KPIECE over a user projection.
- Path post-processing: greedy and randomized shortcutting (`shortcut`, `shortcut_random`).
- Nearest neighbors: `LinearScan` (any metric), `KdTree` (Euclidean spaces, with removal and
  balanced rebuilds); nearest, radius and k-nearest queries.
- State spaces: `Euclidean<N>`, SO(2) (`So2`), SE(2) (`Se2`), SE(3) (`Se3`, unit quaternions),
  `Riemannian<N, F>` (a configuration-dependent metric tensor), Dubins and Reeds-Shepp cars
  (`Dubins`, `ReedsShepp`, with their shortest paths exposed), with samplers `UniformBox`,
  `UniformAngle`, `UniformSe2` and `UniformSe3`.
- Building blocks: straight-line steering (along the curve in the car spaces), ball-shaped goal
  regions.

Validity checks are three-valued (valid, invalid, unknown); planners treat unknown as invalid.
They can also come in batches, for SIMD or GPU checkers; feature `parallel` spreads batches over
threads (`ParallelChecker`).

Not yet implemented: compound state spaces.

## Usage

See the [crate documentation](https://docs.rs/sampling-planning) for a complete RRT* example and
[docs/design.md](https://github.com/sunsided/sampling-planning/blob/main/docs/design.md) for the
design. The example compares RRT, RRT-Connect, RRT*, Informed RRT*, RRT#, PRM*, SPARS, SPARS2,
FMT*, BFMT* and BIT* on one world and draws their paths:

```sh
cargo run --release --example obstacles_2d
```

## Minimum supported Rust version

Rust 1.85 (edition 2024). The MSRV applies to the library; tests and benchmarks may need a
newer toolchain. It may be raised in a minor release when a dependency or a language
feature calls for it.

## Contributing

See [CONTRIBUTING.md](https://github.com/sunsided/sampling-planning/blob/main/CONTRIBUTING.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](https://github.com/sunsided/sampling-planning/blob/main/LICENSE-APACHE))
- MIT license ([LICENSE-MIT](https://github.com/sunsided/sampling-planning/blob/main/LICENSE-MIT))

at your option.

The example in [`examples/`](https://github.com/sunsided/sampling-planning/tree/main/examples) is public domain
([CC0-1.0](https://github.com/sunsided/sampling-planning/blob/main/LICENSES/CC0-1.0.txt)), so you can copy it into anything.

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion
in this work by you, as defined in the Apache-2.0 license, shall be dual licensed as above,
without any additional terms or conditions.
