//! # Sampling-based motion planning
//!
//! Motion planning with sampling-based planners: RRT, RRT*, PRM, FMT* and SST.
//!
//! This is a **motion planning** crate. It is named `sampling-planning` because the crate name `motion-planning` was already taken on crates.io.
//!
//! ## Planned scope
//!
//! - Tree expansion: RRT, RRT*, RRT#
//! - Batch-informed: BIT*, AIT*, ABIT*, EIT*
//! - Roadmaps: PRM, PRM*, SPARS, SPARS2
//! - Fast marching: FMT*, BFMT*
//! - Kinodynamic: SST, SST*, KPIECE
//!
//! ## Status
//!
//! Pre-release. Version 0.0.0 reserves the name; the API is not started yet and will change.
//!
//! Repository: <https://github.com/sunsided/sampling-planning>

#![no_std]

/// A continuous configuration space.
pub trait StateSpace {
    /// A configuration.
    type State;

    /// The distance between two configurations.
    fn distance(&self, from: &Self::State, to: &Self::State) -> f64;

    /// The configuration at fraction `t` in `[0, 1]` of the way from `from` to `to`.
    fn interpolate(&self, from: &Self::State, to: &Self::State, t: f64) -> Self::State;
}
