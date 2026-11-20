//! Tree-expanding planners: RRT, RRT* (optionally informed), RRT# and RRT-Connect.
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
mod rrt_connect;
mod rrt_sharp;
mod rrt_star;
mod targets;
mod tree;

use core::fmt;

pub use rrt::{Rrt, RrtConfig};
pub use rrt_connect::RrtConnect;
pub use rrt_sharp::RrtSharp;
pub use rrt_star::{INFORMED_ATTEMPTS, RrtStar, RrtStarConfig};

use crate::Validity;

/// Why a tree planner could not be created.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The start state did not check as [`Validity::Valid`]; carries the actual result.
    InvalidStart(Validity),
    /// The goal center did not check as [`Validity::Valid`] for a planner that roots a tree
    /// there; carries the actual result.
    InvalidGoal(Validity),
    /// The goal bias is not in `[0, 1]`.
    GoalBiasOutOfRange,
    /// The RRT* radius constant `gamma` is not positive and finite.
    InvalidGamma,
    /// The RRT* radius cap is not positive.
    InvalidMaxRadius,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStart(validity) => write!(f, "start state is not valid ({validity:?})"),
            Self::InvalidGoal(validity) => write!(f, "goal center is not valid ({validity:?})"),
            Self::GoalBiasOutOfRange => f.write_str("goal bias must be in [0, 1]"),
            Self::InvalidGamma => f.write_str("gamma must be positive and finite"),
            Self::InvalidMaxRadius => f.write_str("max radius must be positive"),
        }
    }
}

impl core::error::Error for Error {}
