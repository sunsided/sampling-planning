//! # Sampling-based motion planning
//!
//! Motion planning with sampling-based planners: RRT, RRT*, RRT#, RRT-Connect, PRM, PRM*, FMT*,
//! BFMT*, SST, SST* and KPIECE.
//!
//! This is a **motion planning** crate. It is named `sampling-planning` because the crate name `motion-planning` was already taken on crates.io.
//!
//! ## Building blocks
//!
//! A [`Problem`] bundles everything a planner needs:
//!
//! - a [`StateSpace`] with a metric and interpolation, for example [`Euclidean`];
//! - a [`Sampler`] that draws states, for example [`UniformBox`];
//! - a [`ValidityChecker`] for states and motions, which answers with a three-valued
//!   [`Validity`]. Planners accept only [`Validity::Valid`]; [`Validity::Unknown`] is rejected
//!   like [`Validity::Invalid`];
//! - a [`Steering`] function, for example [`StraightLine`];
//! - a start state and a [`GoalRegion`].
//!
//! Planners keep their nodes in a [`NearestNeighbors`] structure, [`LinearScan`] by default;
//! [`KdTree`] is much faster in low-dimensional Euclidean spaces and returns identical results.
//! Randomness always comes from a caller-provided [`rand_core::Rng`], so a seeded RNG makes a
//! run fully reproducible. Float math goes through `libm`, so results are also identical across
//! platforms.
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
//!   [`RrtConnect`](trees::RrtConnect), all over a [`Problem`].
//! - [`roadmaps`]: the multi-query [`Prm`](roadmaps::Prm) (PRM, PRM*, lazy PRM) and the batch
//!   planners [`FmtStar`](roadmaps::FmtStar) and [`BfmtStar`](roadmaps::BfmtStar). They need no
//!   steering function.
//! - [`kinodynamic`]: [`Sst`](kinodynamic::Sst) (SST and SST*) and
//!   [`Kpiece`](kinodynamic::Kpiece) over a [`Propagator`](kinodynamic::Propagator) that
//!   forward-simulates controls.
//!
//! [`shortcut`] and [`shortcut_random`] post-process any [`Path`].
//!
//! ## Planned scope
//!
//! BIT*, AIT*, ABIT*, EIT*, SPARS, SPARS2 and more state spaces (SO(2),
//! SE(2), SE(3), compound spaces).
//!
//! ## Status
//!
//! Pre-release; the API will change.
//!
//! Repository: <https://github.com/sunsided/sampling-planning>

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;

mod goal;
mod graph;
pub mod kinodynamic;
mod nearest;
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
pub use nearest::{KdTree, LinearScan, NearestNeighbors, RemovableNeighbors};
pub use problem::{Path, Problem};
pub use rand_core;
pub use sampler::{INFORMED_BOX_ATTEMPTS, ProlateHyperspheroid, Sampler, UniformBox, unit_f64};
pub use shortcut::{shortcut, shortcut_random};
pub use space::{Euclidean, StateSpace};
pub use steering::{Steering, StraightLine};
pub use validity::{Validity, ValidityChecker, discretized_motion};
