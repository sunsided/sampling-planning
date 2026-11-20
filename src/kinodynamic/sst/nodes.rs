//! The SST node arena: stable ids, freed slots reused.

use alloc::vec::Vec;
use core::ops::{Index, IndexMut};

/// One tree node.
#[derive(Debug, Clone)]
pub(super) struct Node<S, U> {
    /// The state.
    pub(super) state: S,
    /// The parent node; `None` only for the root.
    pub(super) parent: Option<usize>,
    /// The control and duration from the parent; `None` only for the root.
    pub(super) control: Option<(U, f64)>,
    /// The total control duration from the root.
    pub(super) cost: f64,
    /// The number of stored nodes whose parent is this node.
    pub(super) children: usize,
    /// Whether the node may still be expanded; inactive nodes only stay as path ancestors.
    pub(super) active: bool,
}

/// Node storage with stable ids: freeing a node keeps every other id valid, and a later node
/// may reuse the freed id.
#[derive(Debug, Clone)]
pub(super) struct Nodes<S, U> {
    /// Slots indexed by node id; `None` for freed ids.
    slots: Vec<Option<Node<S, U>>>,
    /// Freed ids, reused last-freed first.
    free: Vec<usize>,
}

impl<S, U> Nodes<S, U> {
    /// An arena holding only `root` as node `0`.
    pub(super) fn new(root: Node<S, U>) -> Self {
        Self {
            slots: alloc::vec![Some(root)],
            free: Vec::new(),
        }
    }

    /// The id the next [`Nodes::push`] will use.
    pub(super) fn next_id(&self) -> usize {
        self.free.last().copied().unwrap_or(self.slots.len())
    }

    /// Stores `node` under [`Nodes::next_id`], counts it as a child of its parent, and returns
    /// its id.
    pub(super) fn push(&mut self, node: Node<S, U>) -> usize {
        let id = self.next_id();

        if let Some(parent) = node.parent {
            self[parent].children += 1;
        }

        if self.free.pop().is_some() {
            self.slots[id] = Some(node);
        } else {
            self.slots.push(Some(node));
        }

        id
    }

    /// Frees node `id` and returns its parent, whose child count drops by one.
    ///
    /// # Panics
    ///
    /// Panics if `id` is not stored or still has children.
    pub(super) fn free(&mut self, id: usize) -> Option<usize> {
        let node = self.slots[id].take().expect("freeing a stored node");
        assert_eq!(node.children, 0, "only leaves are freed");
        self.free.push(id);

        if let Some(parent) = node.parent {
            self[parent].children -= 1;
        }

        node.parent
    }

    /// The number of stored nodes.
    pub(super) fn len(&self) -> usize {
        self.slots.len() - self.free.len()
    }
}

impl<S, U> Index<usize> for Nodes<S, U> {
    type Output = Node<S, U>;

    /// The stored node `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` was freed or never used.
    fn index(&self, id: usize) -> &Node<S, U> {
        self.slots[id].as_ref().expect("node id is stored")
    }
}

impl<S, U> IndexMut<usize> for Nodes<S, U> {
    fn index_mut(&mut self, id: usize) -> &mut Node<S, U> {
        self.slots[id].as_mut().expect("node id is stored")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A node at `state` under `parent` with no control.
    fn node(state: u8, parent: Option<usize>) -> Node<u8, ()> {
        Node {
            state,
            parent,
            control: None,
            cost: 0.0,
            children: 0,
            active: true,
        }
    }

    #[test]
    fn freed_ids_are_reused_and_child_counts_follow() {
        let mut nodes = Nodes::new(node(0, None));
        let a = nodes.push(node(1, Some(0)));
        let b = nodes.push(node(2, Some(a)));
        assert_eq!((a, b, nodes[a].children), (1, 2, 1));

        assert_eq!(nodes.free(b), Some(a));
        assert_eq!((nodes.len(), nodes[a].children), (2, 0));
        assert_eq!(nodes.push(node(3, Some(0))), b, "the freed id is reused");
        assert_eq!(nodes[0].children, 2);
    }

    #[test]
    #[should_panic(expected = "only leaves")]
    fn refuses_to_free_inner_nodes() {
        let mut nodes = Nodes::new(node(0, None));
        nodes.push(node(1, Some(0)));
        nodes.free(0);
    }
}
