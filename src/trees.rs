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

//! ## Vertex retirement
//!
//! [`Rrt`], [`RrtStar`] and [`RrtConnect`] can retire exhausted vertices: `with_retirement`
//! removes a vertex from the nearest-neighbor index so that later iterations stop picking it.
//! The vertex stays in the tree, so paths through it are kept. It is off by default and needs
//! an index that implements [`RemovableNeighbors`](crate::RemovableNeighbors) and [`Clone`],
//! such as [`LinearScan`](crate::LinearScan) or [`KdTree`](crate::KdTree). The roots are never
//! retired, and [`retired_count`](Rrt::retired_count) reports how many vertices were.
//!
//! - **Finite steering:** when [`Steering::successors`](crate::Steering::successors) lists
//!   the states steering can produce from a vertex, the vertex is retired as soon as all of
//!   them are in the tree. This is exact: steering from it can produce no new state.
//! - **Continuous steering:** a vertex is also retired after `failures` consecutive failed
//!   extensions from it, where an extension fails when its motion is not [`Validity::Valid`]
//!   or the steered state does not move away from the vertex. This cap is a heuristic that
//!   trades probabilistic completeness for the retired vertices: a vertex at the mouth of a
//!   narrow passage that failed `failures` times in a row is never extended again, even if a
//!   later sample would have led through. Pick a cap well above one when narrow passages
//!   matter, or [`NonZeroUsize::MAX`](core::num::NonZeroUsize::MAX) to rely on exhaustion
//!   alone.
//!
//! RRT* also stops considering a retired vertex as a parent or a rewiring target, because it
//! searches neighbors in the same index.

#[cfg(feature = "parallel")]
mod forest;
mod growth;
mod retirement;
mod rrt;
mod rrt_connect;
mod rrt_sharp;
mod rrt_star;
mod targets;
mod tree;

use core::fmt;

#[cfg(feature = "parallel")]
pub use forest::grow_forest;
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
