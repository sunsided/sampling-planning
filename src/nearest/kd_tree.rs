//! An incremental k-d tree for Euclidean spaces.

use alloc::vec::Vec;

use super::NearestNeighbors;
use crate::{Euclidean, StateSpace};

/// Relative slack added to pruning bounds.
///
/// The splitting-plane gap is computed with one subtraction while
/// [`Euclidean::distance`](crate::StateSpace::distance) goes through a sum of squares and a
/// square root, so the two can disagree by a few ulps. Visiting a subtree too often only costs
/// time; pruning one too eagerly could change a result, so the bound errs on the visiting side.
const PRUNE_SLACK: f64 = 1e-9;

/// A k-d tree over `[f64; N]` states for the [`Euclidean`] metric.
///
/// Nodes are inserted one at a time without rebalancing. The split axis cycles with the
/// depth. Planners insert random samples, which keeps the tree balanced in expectation; a
/// sorted insertion order degrades it to a list.
///
/// Queries return exactly what [`LinearScan`](crate::LinearScan) returns, including its tie
/// breaking toward the earliest inserted state, because both compare the same
/// [`StateSpace::distance`] values. Expected cost per query is `O(log n)` in low dimensions;
/// beyond roughly ten dimensions the pruning stops paying off and a linear scan is as fast.
///
/// # Examples
///
/// ```
/// use sampling_planning::{Euclidean, KdTree, NearestNeighbors};
///
/// let mut tree = KdTree::<2>::default();
/// tree.insert(7, [0.0, 0.0]);
/// tree.insert(8, [3.0, 4.0]);
///
/// assert_eq!(tree.nearest(&Euclidean::<2>, &[2.5, 3.0]), Some(8));
/// ```
#[derive(Debug, Clone)]
pub struct KdTree<const N: usize> {
    /// All nodes in insertion order; index 0 is the root.
    nodes: Vec<Node<N>>,
    /// The number of nodes that were not removed.
    live: usize,
}

/// One node of a [`KdTree`].
#[derive(Debug, Clone)]
struct Node<const N: usize> {
    /// The stored state; also the splitting point.
    state: [f64; N],
    /// The caller's id for this state.
    id: usize,
    /// The split axis, `depth % N` (0 when `N == 0`).
    axis: usize,
    /// The subtree with `state[axis]` below this node's, if any.
    left: Option<usize>,
    /// The subtree with `state[axis]` at or above this node's, if any.
    right: Option<usize>,
    /// Whether the node was removed. Removed nodes still split space but never match.
    removed: bool,
}

impl<const N: usize> Default for KdTree<N> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            live: 0,
        }
    }
}

impl<const N: usize> KdTree<N> {
    /// Whether `query` belongs in the left subtree of node `index`.
    fn goes_left(&self, index: usize, query: &[f64; N]) -> bool {
        let node = &self.nodes[index];

        N > 0 && query[node.axis] < node.state[node.axis]
    }

    /// The distance from `query` to the splitting plane of node `index`.
    fn plane_gap(&self, index: usize, query: &[f64; N]) -> f64 {
        let node = &self.nodes[index];

        if N == 0 {
            0.0
        } else {
            libm::fabs(query[node.axis] - node.state[node.axis])
        }
    }

    /// The near and far children of node `index` as seen from `query`.
    fn children(&self, index: usize, query: &[f64; N]) -> (Option<usize>, Option<usize>) {
        let node = &self.nodes[index];

        if self.goes_left(index, query) {
            (node.left, node.right)
        } else {
            (node.right, node.left)
        }
    }

    /// The index of the node whose empty child slot `state` falls into, with the side.
    fn attachment_point(&self, state: &[f64; N]) -> (usize, bool) {
        let mut current = 0;

        loop {
            let left = self.goes_left(current, state);
            let next = if left {
                self.nodes[current].left
            } else {
                self.nodes[current].right
            };

            match next {
                Some(child) => current = child,
                None => return (current, left),
            }
        }
    }

    /// Whether the candidate `(distance, index)` beats the current best. Earlier insertion
    /// wins ties; NaN distances never win.
    #[expect(
        clippy::float_cmp,
        reason = "exact ties must break like LinearScan, which compares the same values"
    )]
    fn improves(best: Option<(f64, usize)>, distance: f64, index: usize) -> bool {
        best.is_none_or(|(best_distance, best_index)| {
            distance < best_distance || (distance == best_distance && index < best_index)
        })
    }
}

impl<const N: usize> NearestNeighbors<Euclidean<N>> for KdTree<N> {
    fn insert(&mut self, id: usize, state: [f64; N]) {
        let index = self.nodes.len();
        let axis = if self.nodes.is_empty() {
            0
        } else {
            let (parent, left) = self.attachment_point(&state);
            let parent_node = &mut self.nodes[parent];

            if left {
                parent_node.left = Some(index);
            } else {
                parent_node.right = Some(index);
            }

            if N == 0 {
                0
            } else {
                (parent_node.axis + 1) % N
            }
        };

        self.nodes.push(Node {
            state,
            id,
            axis,
            left: None,
            right: None,
            removed: false,
        });
        self.live += 1;
    }

    fn nearest(&self, space: &Euclidean<N>, query: &[f64; N]) -> Option<usize> {
        let mut best: Option<(f64, usize)> = None;
        // Pending subtrees with a lower bound on their distance, checked when popped so that
        // the bound is compared against the best distance found by then.
        let mut stack = Vec::new();

        if !self.nodes.is_empty() {
            stack.push((0, 0.0));
        }

        while let Some((index, lower_bound)) = stack.pop() {
            if best.is_some_and(|(d, _)| lower_bound > d * (1.0 + PRUNE_SLACK)) {
                continue;
            }

            let node = &self.nodes[index];
            let distance = space.distance(&node.state, query);

            if !node.removed && Self::improves(best, distance, index) {
                best = Some((distance, index));
            }

            // The far side goes on the stack first so that the near side is searched first.
            let (near, far) = self.children(index, query);
            let gap = self.plane_gap(index, query);
            stack.extend(far.map(|far| (far, gap)));
            stack.extend(near.map(|near| (near, 0.0)));
        }

        best.map(|(_, index)| self.nodes[index].id)
    }

    fn within_radius(
        &self,
        space: &Euclidean<N>,
        query: &[f64; N],
        radius: f64,
        out: &mut Vec<usize>,
    ) {
        let mut hits = Vec::new();
        let mut stack = Vec::new();
        let bound = radius * (1.0 + PRUNE_SLACK);

        if !self.nodes.is_empty() {
            stack.push(0);
        }

        while let Some(index) = stack.pop() {
            let node = &self.nodes[index];

            if !node.removed && space.distance(&node.state, query) <= radius {
                hits.push(index);
            }

            let (near, far) = self.children(index, query);
            stack.extend(near);

            if self.plane_gap(index, query) <= bound {
                stack.extend(far);
            }
        }

        hits.sort_unstable();
        out.clear();
        out.extend(hits.iter().map(|&index| self.nodes[index].id));
    }

    fn len(&self) -> usize {
        self.live
    }
}

impl<const N: usize> super::RemovableNeighbors<Euclidean<N>> for KdTree<N> {
    /// Marks the live node with `id` as removed.
    ///
    /// shortcut: finds the node by a linear search, `O(n)`; add an id index if removal shows up
    /// in profiles.
    ///
    /// The node keeps splitting space, so the tree shape and the cost of later queries do not
    /// shrink.
    fn remove(&mut self, id: usize) -> bool {
        let Some(node) = self.nodes.iter_mut().find(|n| n.id == id && !n.removed) else {
            return false;
        };

        node.removed = true;
        self.live -= 1;

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LinearScan, RemovableNeighbors, Sampler, UniformBox};
    use rand_core::SeedableRng;

    /// Fills a [`KdTree`] and a [`LinearScan`] with the same points, including duplicates.
    fn twins(count: usize, seed: u64) -> (KdTree<3>, LinearScan<[f64; 3]>) {
        let mut sampler = UniformBox::new([0.0; 3], [4.0; 3]);
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(seed);
        let mut kd = KdTree::default();
        let mut linear = LinearScan::default();

        for id in 0..count {
            // Snapping to a coarse grid creates many exact ties and duplicate states.
            let state = sampler.sample(&mut rng).map(libm::floor);
            kd.insert(id * 3, state);
            NearestNeighbors::<Euclidean<3>>::insert(&mut linear, id * 3, state);
        }

        (kd, linear)
    }

    #[test]
    fn matches_linear_scan_exactly() {
        let space = Euclidean::<3>;
        let (kd, linear) = twins(300, 1);
        let mut sampler = UniformBox::new([-1.0; 3], [5.0; 3]);
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(2);
        let (mut a, mut b) = (Vec::new(), Vec::new());

        for i in 0..500 {
            let mut query = sampler.sample(&mut rng);
            if i % 2 == 0 {
                query = query.map(libm::round);
            }

            assert_eq!(kd.nearest(&space, &query), linear.nearest(&space, &query));
            kd.within_radius(&space, &query, 1.5, &mut a);
            linear.within_radius(&space, &query, 1.5, &mut b);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn removed_nodes_never_match() {
        let space = Euclidean::<3>;
        let (mut kd, _) = twins(50, 3);

        let first = kd.nearest(&space, &[1.0, 1.0, 1.0]).expect("non-empty");
        assert!(kd.remove(first));
        assert!(!kd.remove(first), "removing twice is a no-op");
        assert_ne!(kd.nearest(&space, &[1.0, 1.0, 1.0]), Some(first));
        assert_eq!(NearestNeighbors::<Euclidean<3>>::len(&kd), 49);

        let mut out = Vec::new();
        kd.within_radius(&space, &[1.0, 1.0, 1.0], 10.0, &mut out);
        assert_eq!(out.len(), 49);
    }

    #[test]
    fn empty_and_zero_dimensional_trees_work() {
        let empty = KdTree::<2>::default();
        assert_eq!(empty.nearest(&Euclidean::<2>, &[0.0, 0.0]), None);

        let mut point = KdTree::<0>::default();
        point.insert(4, []);
        point.insert(5, []);
        assert_eq!(point.nearest(&Euclidean::<0>, &[]), Some(4));
    }
}
