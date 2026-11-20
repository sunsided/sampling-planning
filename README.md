# sampling-planning: motion planning for Rust

Motion planning with sampling-based planners: RRT, RRT*, RRT#, RRT-Connect, PRM, PRM*, SPARS2,
FMT*, BFMT*, BIT*, SST, SST* and KPIECE.

This is a **motion planning** crate. It is named `sampling-planning` because the crate name `motion-planning` was already taken on crates.io.

## Planned scope

Not implemented yet:

- Batch-informed: AIT*, ABIT*, EIT*
- Roadmaps: SPARS (with a dense graph)
- Kinodynamic: multi-level KPIECE
- State spaces: SE(3), compound spaces

## Status

Pre-release; the API will change. Implemented:

- Tree planners: RRT, RRT* (radius or k-nearest) with optional informed sampling (Informed RRT*, with direct
  prolate-hyperspheroid sampling in Euclidean spaces), RRT-Connect (bidirectional), RRT# (RRG with LPA*-style replanning).
- Roadmaps: PRM (fixed radius or fixed k), PRM* (shrinking radius or `k ln n` nearest), lazy
  PRM, SPARS2 (sparse roadmap spanner); multi-query, with Dijkstra queries on a reusable
  roadmap.
- Batch-informed planners: BIT* (radius or k-nearest, informed batches, pruning).
- Batch planners: FMT* (fast marching tree) and BFMT* (bidirectional, optimal or first-meeting
  termination).
- Kinodynamic planners: SST and SST* over a `Propagator` (forward simulation) trait, with
  inactive leaves freed; single-level KPIECE over a user projection.
- Path post-processing: greedy and randomized shortcutting (`shortcut`, `shortcut_random`).
- Nearest neighbors: `LinearScan` (any metric), `KdTree` (Euclidean spaces, with removal and
  balanced rebuilds); nearest, radius and k-nearest queries.
- State spaces: `Euclidean<N>`, SO(2) (`So2`), SE(2) (`Se2`), Dubins and Reeds-Shepp cars
  (`Dubins`, `ReedsShepp`, with their shortest paths exposed), with samplers `UniformBox`,
  `UniformAngle` and `UniformSe2`.
- Building blocks: straight-line steering (along the curve in the car spaces), ball-shaped goal
  regions.

Validity checks are three-valued (valid, invalid, unknown); planners treat unknown as invalid.

See [docs/design.md](docs/design.md) for the design and milestones, and run the example:

```sh
cargo run --release --example obstacles_2d
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
