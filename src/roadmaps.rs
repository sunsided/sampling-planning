//! Roadmap and batch planners: PRM, PRM*, lazy PRM and FMT*.
//!
//! These planners do not grow a tree from the start. [`Prm`] builds a graph once and answers
//! many queries on it; FMT* marches a tree through a batch of samples drawn up front.
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

mod fmt_star;
mod graph;
mod prm;

use core::fmt;

pub use fmt_star::{FmtStar, FmtStarConfig};
pub use prm::{Connection, Prm, PrmConfig};

use crate::Validity;

/// Why a roadmap planner could not be created or queried.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The query start did not check as [`Validity::Valid`]; carries the actual result.
    InvalidStart(Validity),
    /// A connection radius or radius constant is not positive and finite.
    InvalidRadius,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStart(validity) => write!(f, "start state is not valid ({validity:?})"),
            Self::InvalidRadius => f.write_str("connection radius must be positive and finite"),
        }
    }
}

impl core::error::Error for Error {}
