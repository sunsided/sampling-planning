//! Roadmap and batch planners: PRM, PRM*, lazy PRM, SPARS, SPARS2, FMT* and BFMT*.
//!
//! These planners do not grow a tree from the start. [`Prm`] builds a graph once and answers
//! many queries on it; FMT* marches a tree through a batch of samples drawn up front, BFMT*
//! marches two, from the start and from the goal.
//! Neither needs a [`Steering`](crate::Steering) function: edges are direct motions along
//! [`StateSpace::interpolate`](crate::StateSpace::interpolate).
//!
//! ```mermaid
//! graph LR
//!     Grow["grow: sample, check state"] --> Connect["connect to neighbors"]
//!     Connect --> Grow
//!     Connect --> Query["query: attach start and goal"]
//!     Query --> Dijkstra
//!     Dijkstra -->|lazy: invalid edge| Query
//! ```

mod bfmt_star;
mod fmt_star;
mod march;
mod prm;
mod spars;
mod spars2;
mod sparse;

use core::fmt;

pub use bfmt_star::{BfmtStar, BfmtStarConfig, Termination};
pub use fmt_star::{FmtStar, FmtStarConfig};
pub use prm::{Connection, Prm, PrmConfig};
pub use spars::{Spars, SparsConfig};
pub use spars2::{NEAR_ATTEMPTS, Spars2, Spars2Config};
pub use sparse::GuardKind;

use crate::Validity;

/// Why a roadmap planner could not be created or queried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The query start did not check as [`Validity::Valid`]; carries the actual result.
    InvalidStart(Validity),
    /// The goal center did not check as [`Validity::Valid`] for a planner that roots a tree
    /// there; carries the actual result.
    InvalidGoal(Validity),
    /// A connection radius, radius constant or k-nearest constant is not positive and finite,
    /// or a fixed neighbor count is zero.
    InvalidRadius,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStart(validity) => write!(f, "start state is not valid ({validity:?})"),
            Self::InvalidGoal(validity) => write!(f, "goal center is not valid ({validity:?})"),
            Self::InvalidRadius => f.write_str(
                "connection radius and constants must be positive and finite, k at least one",
            ),
        }
    }
}

impl core::error::Error for Error {}
