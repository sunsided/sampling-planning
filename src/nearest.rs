//! Nearest-neighbor structures over planner nodes.

mod linear;

use alloc::vec::Vec;

use crate::StateSpace;
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
