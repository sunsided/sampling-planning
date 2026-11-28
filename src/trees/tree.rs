//! The search tree shared by RRT and RRT*.

use alloc::vec::Vec;

/// A rooted tree of states with path costs. Node `0` is the root.
#[derive(Debug, Clone)]
pub(super) struct Tree<S> {
    /// All nodes, indexed by node id. Never empty.
    nodes: Vec<Node<S>>,
}

/// One node of a [`Tree`].
#[derive(Debug, Clone)]
struct Node<S> {
    /// The state at this node.
    state: S,
    /// The parent node id; `None` only for the root.
    parent: Option<usize>,
    /// Path cost from the root: the parent's cost plus the distance from the parent.
    cost: f64,
    /// Ids of the nodes whose parent is this node.
    children: Vec<usize>,
}

impl<S: Clone> Tree<S> {
    /// Creates a tree holding only the root `state` with cost zero.
    pub(super) fn new(state: S) -> Self {
        Self {
            nodes: alloc::vec![Node {
                state,
                parent: None,
                cost: 0.0,
                children: Vec::new(),
            }],
        }
    }

    /// The number of nodes, at least one.
    pub(super) fn len(&self) -> usize {
        self.nodes.len()
    }

    /// The state of node `id`.
    pub(super) fn state(&self, id: usize) -> &S {
        &self.nodes[id].state
    }

    /// The path cost from the root to node `id`.
    pub(super) fn cost(&self, id: usize) -> f64 {
        self.nodes[id].cost
    }

    /// Adds a node under `parent` with the given path cost and returns its id.
    pub(super) fn add(&mut self, state: S, parent: usize, cost: f64) -> usize {
        let id = self.nodes.len();
        self.nodes.push(Node {
            state,
            parent: Some(parent),
            cost,
            children: Vec::new(),
        });
        self.nodes[parent].children.push(id);

        id
    }

    /// Moves node `id` under `parent` with the new path cost `cost`, then recomputes the costs
    /// of all its descendants with `edge_cost(parent state, child state)`.
    ///
    /// Descendant costs use the same formula as when they were added (parent cost plus edge
    /// cost), so when `cost` is lower, no descendant cost increases, not even by rounding.
    pub(super) fn reparent(
        &mut self,
        id: usize,
        parent: usize,
        cost: f64,
        edge_cost: impl Fn(&S, &S) -> f64,
    ) {
        if let Some(old) = self.nodes[id].parent {
            self.nodes[old].children.retain(|&child| child != id);
        }

        self.nodes[id].parent = Some(parent);
        self.nodes[id].cost = cost;
        self.nodes[parent].children.push(id);
        self.propagate_costs(id, edge_cost);
    }

    /// Recomputes the costs of all descendants of `id` from their parents' costs.
    fn propagate_costs(&mut self, id: usize, edge_cost: impl Fn(&S, &S) -> f64) {
        let mut stack = self.nodes[id].children.clone();

        while let Some(child) = stack.pop() {
            let parent = self.nodes[child].parent.expect("a child has a parent");
            let edge = edge_cost(&self.nodes[parent].state, &self.nodes[child].state);
            self.nodes[child].cost = self.nodes[parent].cost + edge;
            stack.extend_from_slice(&self.nodes[child].children);
        }
    }

    /// All states in insertion order, the root first; parents come before their children.
    #[cfg(feature = "parallel")]
    pub(super) fn states(&self) -> impl Iterator<Item = &S> {
        self.nodes.iter().map(|node| &node.state)
    }

    /// The states from the root to node `id`.
    pub(super) fn path_to(&self, id: usize) -> Vec<S> {
        let mut states = Vec::new();
        let mut current = Some(id);

        while let Some(node) = current {
            states.push(self.nodes[node].state.clone());
            current = self.nodes[node].parent;
        }

        states.reverse();
        states
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Euclidean, StateSpace};

    #[test]
    fn reparent_updates_descendants_and_children() {
        let space = Euclidean::<1>;
        let mut tree = Tree::new([0.0]);
        let a = tree.add([5.0], 0, 5.0);
        let b = tree.add([6.0], a, 6.0);
        let c = tree.add([7.0], b, 7.0);
        let shortcut = tree.add([5.5], 0, 5.5);

        let distance = |a: &[f64; 1], b: &[f64; 1]| space.distance(a, b);
        tree.reparent(b, shortcut, 6.0, distance);
        assert_eq!(tree.nodes[b].parent, Some(shortcut));
        assert_eq!(tree.nodes[a].children, []);

        tree.reparent(shortcut, 0, 1.0, distance);
        assert!((tree.cost(b) - 1.5).abs() < 1e-12);
        assert!((tree.cost(c) - 2.5).abs() < 1e-12);
        assert_eq!(tree.path_to(c), [[0.0], [5.5], [6.0], [7.0]]);
    }
}
