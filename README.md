# sampling-planning: motion planning for Rust

Motion planning with sampling-based planners: RRT, RRT*, PRM, FMT* and SST.

This is a **motion planning** crate. It is named `sampling-planning` because the crate name `motion-planning` was already taken on crates.io.

## Planned scope

- Tree expansion: RRT, RRT*, RRT#
- Batch-informed: BIT*, AIT*, ABIT*, EIT*
- Roadmaps: PRM, PRM*, SPARS, SPARS2
- Fast marching: FMT*, BFMT*
- Kinodynamic: SST, SST*, KPIECE

## Status

Pre-release; the API will change. Implemented:

- Tree planners: RRT, RRT* with optional informed sampling (Informed RRT*), RRT-Connect
  (bidirectional).
- Roadmaps: PRM (fixed radius), PRM* (shrinking radius), lazy PRM; multi-query, with
  Dijkstra queries on a reusable roadmap.
- Path post-processing: greedy and randomized shortcutting (`shortcut`, `shortcut_random`).
- Nearest neighbors: `LinearScan` (any metric), `KdTree` (Euclidean spaces).
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
