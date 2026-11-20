//! Batch-informed planners: BIT* and AIT*.
//!
//! These planners process samples in batches and search the implicit random geometric graph
//! over them in order of a heuristic estimate of the solution cost through each edge, like A*.
//! After the first solution they only sample and keep states that can still improve it.
//!
//! [`BitStar`] orders its search by the geometric cost-to-go. [`AitStar`] computes the
//! cost-to-go over the graph itself with a lazy reverse search (LPA*, no collision checks) and
//! repairs it whenever the forward search finds an edge invalid; see its docs for the loop.
//!
//! ```mermaid
//! graph LR
//!     Batch["new batch: prune, sample informed set"] --> Expand["expand vertices into edge queue"]
//!     Expand --> Edge["pop best edge"]
//!     Edge -->|can improve| Check["check motion, add or rewire"]
//!     Check --> Expand
//!     Edge -->|cannot improve| Batch
//! ```

mod ait_star;
mod batch;
mod bit_star;
mod queues;

use core::fmt;

pub use ait_star::{AitStar, AitStarConfig};
pub use bit_star::{BitStar, BitStarConfig};

use crate::Validity;

/// Why a batch-informed planner could not be created.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The start state did not check as [`Validity::Valid`]; carries the actual result.
    InvalidStart(Validity),
    /// The radius constant, k-nearest constant or radius cap is not positive (the constants
    /// must also be finite).
    InvalidRadius,
    /// The batch size is zero.
    EmptyBatch,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStart(validity) => write!(f, "start state is not valid ({validity:?})"),
            Self::InvalidRadius => f.write_str("radius constants and the cap must be positive"),
            Self::EmptyBatch => f.write_str("batch size must be at least one"),
        }
    }
}

impl core::error::Error for Error {}
