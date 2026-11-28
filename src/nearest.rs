//! Nearest-neighbor structures over planner nodes.

mod k_best;
mod kd_tree;
mod linear;
mod vp_tree;

use alloc::vec::Vec;

use crate::StateSpace;
pub use kd_tree::KdTree;
pub use linear::LinearScan;
pub use vp_tree::VpTree;

/// The panic message where a planner's index is never empty but `nearest` found nothing,
/// which only NaN distances cause.
pub(crate) const NO_NEAREST: &str =
    "no nearest node in a non-empty index: StateSpace::distance returned NaN for every node";

/// Stores planner nodes by id and answers proximity queries under a [`StateSpace`] metric.
///
/// Ids are the planner's node ids, so the structure needs no back-pointers. Queries break
/// ties toward the earliest inserted state, which keeps seeded runs reproducible.
pub trait NearestNeighbors<Sp: StateSpace> {
    /// Adds a node with the given id.
    fn insert(&mut self, id: usize, state: Sp::State);

    /// The id of the node closest to `query`, or `None` if the structure is empty or every
    /// distance is NaN. Nodes at a NaN distance are never returned.
    fn nearest(&self, space: &Sp, query: &Sp::State) -> Option<usize>;

    /// Replaces the contents of `out` with the ids of all nodes within `radius` of `query`,
    /// in insertion order.
    fn within_radius(&self, space: &Sp, query: &Sp::State, radius: f64, out: &mut Vec<usize>);

    /// Replaces the contents of `out` with the ids of the `k` nodes closest to `query`, closest
    /// first.
    ///
    /// Equal distances are ordered by insertion, earliest first, and the same rule decides which
    /// nodes make the cut at the `k`-th place. Fewer than `k` ids are returned only when fewer
    /// nodes are stored. Nodes at a NaN distance are never returned.
    fn k_nearest(&self, space: &Sp, query: &Sp::State, k: usize, out: &mut Vec<usize>);

    /// The number of stored nodes.
    fn len(&self) -> usize;

    /// Whether no node is stored.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A [`NearestNeighbors`] structure that can also forget nodes.
///
/// Planners that keep only a sparse set of active nodes (SST) need this. It is a separate trait
/// so that structures that cannot remove cheaply still work with every other planner.
pub trait RemovableNeighbors<Sp: StateSpace>: NearestNeighbors<Sp> {
    /// Removes the node with `id`. Returns whether such a node was stored.
    ///
    /// Ids should be unique among the stored nodes; if an id was inserted twice, which of the
    /// two nodes is removed depends on the structure. A removed id may be inserted again, and
    /// then counts as the newest node for tie breaking.
    fn remove(&mut self, id: usize) -> bool;
}
