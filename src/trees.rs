//! Tree-expanding planners: RRT and RRT*.
//!
//! Both planners are incremental. [`Rrt::step`] and [`RrtStar::step`] do one iteration,
//! `run` does many, and `best_path` returns the current best solution. One iteration:
//!
//! ```mermaid
//! graph LR
//!     Sample["sample (goal-biased)"] --> Nearest
//!     Nearest --> Steer
//!     Steer --> Check["check motion"]
//!     Check -->|valid| Add["add node"]
//!     Check -->|invalid or unknown| Sample
//!     Add -->|RRT*| Rewire["choose parent, rewire"]
//! ```

mod growth;
mod rrt;
mod rrt_star;
mod tree;

use core::fmt;

pub use rrt::{Rrt, RrtConfig};
pub use rrt_star::{RrtStar, RrtStarConfig};

use crate::Validity;

/// Why a tree planner could not be created.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The start state did not check as [`Validity::Valid`]; carries the actual result.
    InvalidStart(Validity),
    /// The goal bias is not in `[0, 1]`.
    GoalBiasOutOfRange,
    /// The RRT* radius constant `gamma` is not positive and finite.
    InvalidGamma,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStart(validity) => write!(f, "start state is not valid ({validity:?})"),
            Self::GoalBiasOutOfRange => f.write_str("goal bias must be in [0, 1]"),
            Self::InvalidGamma => f.write_str("gamma must be positive and finite"),
        }
    }
}

impl core::error::Error for Error {}

/// Converts a count to `f64`, saturating at `u32::MAX` (far beyond any tree size).
fn count_to_f64(count: usize) -> f64 {
    f64::from(u32::try_from(count).unwrap_or(u32::MAX))
}
