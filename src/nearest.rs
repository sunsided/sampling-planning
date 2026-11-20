//! Nearest-neighbor structures over planner nodes.

mod kd_tree;
mod linear;

use alloc::vec::Vec;

use crate::StateSpace;
pub use kd_tree::KdTree;
pub use linear::LinearScan;

/// Stores planner nodes by id and answers proximity queries under a [`StateSpace`] metric.
///
/// Ids are the planner's node ids, so the structure needs no back-pointers. Queries break
/// ties toward the earliest inserted state, which keeps seeded runs reproducible.
pub trait NearestNeighbors<Sp: StateSpace> {
    /// Adds a node with the given id.
    fn insert(&mut self, id: usize, state: Sp::State);

    /// The id of the node closest to `query`, or `None` if the structure is empty.
    fn nearest(&self, space: &Sp, query: &Sp::State) -> Option<usize>;

    /// Replaces the contents of `out` with the ids of all nodes within `radius` of `query`,
    /// in insertion order.
    fn within_radius(&self, space: &Sp, query: &Sp::State, radius: f64, out: &mut Vec<usize>);

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
    fn remove(&mut self, id: usize) -> bool;
}
