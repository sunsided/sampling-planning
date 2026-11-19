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

Pre-release. RRT and RRT* work in Euclidean spaces with a uniform box sampler, linear-scan
nearest neighbors and straight-line steering. Validity checks are three-valued (valid,
invalid, unknown); planners treat unknown as invalid. The API will change.

See [docs/design.md](docs/design.md) for the design and milestones, and run the example:

```sh
cargo run --release --example obstacles_2d
```

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.
