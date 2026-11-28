# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0](https://github.com/sunsided/sampling-planning/compare/v0.1.1...v0.2.0) - 2026-11-28

### Added

- Add real-world examples for every planner family
- Add ABIT*, EIT*, multi-level KPIECE, SE(3) and cost objectives

### Changed

- Make the parking example solve every seed, flag failed KPIECE runs
- Report infinite risk where the risk integral is not computable
- Reject KPIECE level ratios whose scale overflows, stop self-neighbors
- Keep every node when merging an RRT* forest

## [0.1.1](https://github.com/sunsided/sampling-planning/compare/v0.1.0...v0.1.1) - 2026-11-22

### Changed

- Link the crate to its page on the planning and search site

## [0.1.0](https://github.com/sunsided/sampling-planning/releases/tag/v0.1.0) - 2026-11-21

### Added

- Add CONTRIBUTING.md
- Add README badges and use absolute links
- Add property tests for angles and the SE(2) and car metrics
- Add fuzz targets for car paths and the kd-tree
- Add REUSE licensing metadata
- Add cargo-deny, Dependabot and typos configuration
- Add the Codecov configuration
- Add the Taskfile and CI workflows
- Add AIT*, adaptively informed trees
- Add SPARS sparse roadmap spanners with a dense graph
- Add SO(2), SE(2), Dubins and Reeds-Shepp state spaces
- Add SPARS2 sparse roadmap spanners
- Add BIT*
- Add single-level KPIECE
- Add BFMT*
- Add RRT#
- Add k-nearest PRM* and RRT*-k
- Add k-nearest queries and real k-d tree removal
- Add SST and SST* with a Propagator trait
- Add FMT*
- Add PRM, PRM* and lazy PRM
- Add informed sampling to RRT*
- Add RRT-Connect
- Add greedy and randomized path shortcutting
- Add a k-d tree nearest-neighbor index for Euclidean spaces
- Add 2D obstacle example and planner integration tests
- Add RRT and RRT* tree planners
- Add state space, sampler, validity, nearest-neighbor and steering building blocks
- Add design document

### Changed

- Describe the released scope and audience in the README
- Stop tracking Cargo.lock
- Extend the lint table, clippy.toml and rustfmt.toml
- Render mermaid diagrams in the rustdoc output
- Track Cargo.lock
- Cross-check AIT* in k-nearest mode and later batches
- Make the SPARS2 tests discriminate and fix two doc claims
- Show every geometric planner in the obstacles_2d example
- Plan over the new spaces, bench their distances, document them
- Use Theorem 39's bound for RRT*-k
- Sample the informed set of Euclidean problems directly
- Free SST's inactive leaves
- Test directional checks, SST's witness rules and its index cross-check
- Keep RRT-Connect's direct goal solution
- Check the stub motions of random shortcuts
- Document the implemented planners and remaining scope
- Cap the RRT* connection radius
- Describe the current status in the README
- Reserve the crate name with a 0.0.0 placeholder

### Fixed

- Fix ambiguous shortcut doc links
