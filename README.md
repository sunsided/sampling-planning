# sampling-planning: motion planning for Rust

Motion planning with sampling-based planners: RRT, RRT*, RRT#, RRT-Connect, PRM, PRM*, FMT*,
SST and SST*.

This is a **motion planning** crate. It is named `sampling-planning` because the crate name `motion-planning` was already taken on crates.io.

## Planned scope

Not implemented yet:

- Batch-informed: BIT*, AIT*, ABIT*, EIT*
- Roadmaps: SPARS, SPARS2
- Fast marching: BFMT*
- Kinodynamic: KPIECE
- State spaces: SO(2), SE(2), SE(3), compound spaces

## Status

Pre-release; the API will change. Implemented:

- Tree planners: RRT, RRT* (radius or k-nearest) with optional informed sampling (Informed RRT*, with direct
  prolate-hyperspheroid sampling in Euclidean spaces), RRT-Connect (bidirectional), RRT# (RRG with LPA*-style replanning).
- Roadmaps: PRM (fixed radius or fixed k), PRM* (shrinking radius or `k ln n` nearest), lazy
  PRM; multi-query, with Dijkstra queries on a reusable roadmap.
- Batch planners: FMT* (fast marching tree).
- Kinodynamic planners: SST and SST* over a `Propagator` (forward simulation) trait; inactive
  leaves are freed.
- Path post-processing: greedy and randomized shortcutting (`shortcut`, `shortcut_random`).
- Nearest neighbors: `LinearScan` (any metric), `KdTree` (Euclidean spaces, with removal and
  balanced rebuilds); nearest, radius and k-nearest queries.
- Building blocks: `Euclidean<N>` spaces, `UniformBox` sampling, straight-line steering,
  ball-shaped goal regions.

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
