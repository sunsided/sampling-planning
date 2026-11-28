//! # Sampling-based motion planning
//!
//! Sampling-based motion planners for continuous configuration spaces: RRT, RRT*, RRT#,
//! RRT-Connect, PRM, PRM*, SPARS, SPARS2, FMT*, BFMT*, BIT*, ABIT*, AIT*, EIT*, SST, SST* and
//! (multi-level) KPIECE. The crate is `#![no_std]` with `alloc`; feature `parallel` adds
//! multi-threaded batch checks and roadmap queries on `std`.
//!
//! This is a **motion planning** crate. It is named `sampling-planning` because the crate name `motion-planning` was already taken on crates.io.
//!
//! ## Building blocks
//!
//! A [`Problem`] bundles everything a planner needs:
//!
//! - a [`StateSpace`] with a metric and interpolation, for example [`Euclidean`], [`So2`],
//!   [`Se2`], [`Se3`], or the car-like [`Dubins`] and [`ReedsShepp`];
//! - a [`Sampler`] that draws states, for example [`UniformBox`], which samples informed sets
//!   directly in Euclidean spaces and, with a [`LoewnerBound`], under a configuration-dependent
//!   metric such as [`Riemannian`]'s;
//! - a [`ValidityChecker`] for states and motions, which answers with a three-valued
//!   [`Validity`]. Planners accept only [`Validity::Valid`]; [`Validity::Unknown`] is rejected
//!   like [`Validity::Invalid`]. Checks can come in batches, which `ParallelChecker` (feature
//!   `parallel`) spreads over threads;
//! - a [`Steering`] function, for example [`StraightLine`];
//! - a start state and a [`GoalRegion`].
//!
//! Planners keep their nodes in a [`NearestNeighbors`] structure, [`LinearScan`] by default;
//! [`KdTree`] is much faster in low-dimensional Euclidean spaces and [`VpTree`] in other metric
//! spaces such as [`Se3`], and both return identical results.
//! Randomness always comes from a caller-provided [`rand_core::Rng`], so a seeded RNG makes a
//! run fully reproducible. Float math goes through `libm`, so results are also identical across
//! platforms.
//!
//! Paths cost their length by default. [`RrtStar`](trees::RrtStar) also minimizes any
//! [`CostObjective`], such as the hybrid risk-density objective [`RiskDensity`].
//!
//! ## Example
//!
//! ```
//! use rand_core::SeedableRng;
//! use sampling_planning::{
//!     Euclidean, GoalRegion, Problem, StraightLine, UniformBox, Validity, ValidityChecker,
//!     trees::{RrtStar, RrtStarConfig},
//! };
//!
//! /// Free space everywhere.
//! struct Free;
//!
//! impl ValidityChecker<Euclidean<2>> for Free {
//!     fn check_state(&self, _: &[f64; 2]) -> Validity {
//!         Validity::Valid
//!     }
//!
//!     fn check_motion(&self, _: &Euclidean<2>, _: &[f64; 2], _: &[f64; 2]) -> Validity {
//!         Validity::Valid
//!     }
//! }
//!
//! let problem = Problem {
//!     space: Euclidean::<2>,
//!     sampler: UniformBox::new([0.0, 0.0], [10.0, 10.0]),
//!     checker: Free,
//!     steering: StraightLine::new(1.0),
//!     start: [1.0, 1.0],
//!     goal: GoalRegion::new([9.0, 9.0], 0.5),
//! };
//!
//! let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(7);
//! let gamma = 1.5 * RrtStarConfig::gamma_lower_bound(2, 100.0);
//! let mut planner = RrtStar::new(problem, RrtStarConfig::new(gamma, 1.0)).expect("valid problem");
//! planner.run(2000, &mut rng);
//!
//! let path = planner.best_path().expect("free space has a path");
//! assert!(path.cost < 12.0);
//! ```
//!
//! ## Planners
//!
//! - [`trees`]: [`Rrt`](trees::Rrt), [`RrtStar`](trees::RrtStar) (radius or k-nearest,
//!   optionally informed), [`RrtSharp`](trees::RrtSharp) and the bidirectional
//!   [`RrtConnect`](trees::RrtConnect), all over a [`Problem`]. RRT, RRT* and RRT-Connect can
//!   [retire](trees#vertex-retirement) vertices that keep failing to extend, and
//!   `trees::grow_forest` (feature `parallel`) grows RRT* trees on several threads and
//!   merges them.
//! - [`roadmaps`]: the multi-query [`Prm`](roadmaps::Prm) (PRM, PRM*, lazy PRM) and
//!   the sparse roadmap spanners [`Spars`](roadmaps::Spars) and [`Spars2`](roadmaps::Spars2), and the batch
//!   planners [`FmtStar`](roadmaps::FmtStar) and [`BfmtStar`](roadmaps::BfmtStar). They need no
//!   steering function.
//! - [`informed`]: the batch-informed [`BitStar`](informed::BitStar),
//!   [`AbitStar`](informed::AbitStar), [`AitStar`](informed::AitStar) and
//!   [`EitStar`](informed::EitStar).
//! - [`kinodynamic`]: [`Sst`](kinodynamic::Sst) (SST and SST*) and
//!   [`Kpiece`](kinodynamic::Kpiece) (one or several grid levels) over a
//!   [`Propagator`](kinodynamic::Propagator) that forward-simulates controls.
//!
//! [`shortcut()`] and [`shortcut_random`] post-process any [`Path`].
//!
//! Not yet implemented: compound state spaces.
//!
//! Repository: <https://github.com/sunsided/sampling-planning>

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

mod goal;
mod graph;
pub mod informed;
pub mod kinodynamic;
mod nearest;
mod objective;
#[cfg(feature = "parallel")]
mod parallel;
mod problem;
mod radius;
pub mod roadmaps;
mod sampler;
mod shortcut;
mod space;
mod steering;
pub mod trees;
mod validity;

pub use goal::GoalRegion;
pub use nearest::{KdTree, LinearScan, NearestNeighbors, RemovableNeighbors, VpTree};
pub use objective::{CostObjective, GaussianObstacle, PathLength, RiskDensity};
#[cfg(feature = "parallel")]
pub use parallel::ParallelChecker;
pub use problem::{Path, Problem};
pub use rand_core;
pub use sampler::{
    INFORMED_BOX_ATTEMPTS, LoewnerBound, ProlateHyperspheroid, Sampler, UniformBox, unit_f64,
};
pub use shortcut::{shortcut, shortcut_random};
pub use space::{
    Dubins, DubinsPath, DubinsWord, Euclidean, ReedsShepp, ReedsSheppPath, Riemannian, Se2,
    Se2State, Se3, Se3State, Segment, SegmentKind, So2, StateSpace, UniformAngle, UniformSe2,
    UniformSe3, angle_difference, wrap_angle,
};
pub use steering::{Steering, StraightLine};
pub use validity::{Validity, ValidityChecker, discretized_motion};
