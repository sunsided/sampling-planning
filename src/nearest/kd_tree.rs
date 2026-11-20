//! An incremental k-d tree for Euclidean spaces.

use alloc::{collections::BTreeMap, vec::Vec};

use super::{NearestNeighbors, k_best::KBest};
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
/// sorted insertion order degrades it to a list until [`KdTree::rebalance`] is called.
///
/// Removal ([`RemovableNeighbors`](crate::RemovableNeighbors)) finds the node through an id
/// index, marks it, and rebuilds the tree balanced once removed nodes outnumber live ones.
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
    /// All stored nodes, removed ones included until the next rebuild, in insertion order.
    ///
    /// The position of a node is its insertion rank, which breaks distance ties. Rebuilds drop
    /// removed nodes but keep the relative order of the rest.
    nodes: Vec<Node<N>>,
    /// The index of the root node in `nodes`; `None` while the tree has no nodes.
    root: Option<usize>,
    /// Maps each stored id to the index of its live node, for `O(log n)` removal.
    ///
    /// When an id is inserted twice, the newer node wins.
    by_id: BTreeMap<usize, usize>,
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

/// A pending subtree of the balanced build: a range of `order` and where its root attaches.
#[derive(Debug, Clone, Copy)]
struct BuildTask {
    /// The first position of the range in the build order.
    start: usize,
    /// One past the last position of the range.
    end: usize,
    /// The depth of the subtree root; its split axis is `depth % N`.
    depth: usize,
    /// The parent node and whether the subtree is its left child; `None` for the root.
    parent: Option<(usize, bool)>,
}

/// The fewest removed nodes that trigger a rebuild, so that tiny trees do not rebuild on every
/// removal.
const MIN_REBUILD_TOMBSTONES: usize = 16;

impl<const N: usize> Default for KdTree<N> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            root: None,
            by_id: BTreeMap::new(),
            live: 0,
        }
    }
}

impl<const N: usize> KdTree<N> {
    /// Rebuilds the tree as a balanced k-d tree over the live nodes, dropping removed ones.
    ///
    /// Each subtree splits at the median of its states along the axis of its depth, so the
    /// depth is `O(log n)` whatever the insertion order was. Insertion ranks keep their relative
    /// order, so query results do not change. Costs `O(n log^2 n)`.
    ///
    /// Removal calls this automatically once removed nodes outnumber live ones. Call it by hand
    /// after inserting many states in sorted order, which degrades the incremental tree to a
    /// list.
    ///
    /// # Examples
    ///
    /// ```
    /// use sampling_planning::{Euclidean, KdTree, NearestNeighbors};
    ///
    /// let mut tree = KdTree::<1>::default();
    /// for id in 0..100 {
    ///     tree.insert(id, [f64::from(u32::try_from(id).expect("small"))]);
    /// }
    ///
    /// tree.rebalance();
    /// assert_eq!(tree.nearest(&Euclidean::<1>, &[41.2]), Some(41));
    /// ```
    pub fn rebalance(&mut self) {
        self.nodes.retain(|node| !node.removed);
        self.by_id = self
            .nodes
            .iter()
            .enumerate()
            .map(|(index, node)| (node.id, index))
            .collect();
        self.live = self.nodes.len();
        self.root = self.build_balanced();
    }

    /// Links all nodes into a balanced tree and returns the root index.
    fn build_balanced(&mut self) -> Option<usize> {
        let mut order: Vec<usize> = (0..self.nodes.len()).collect();
        let mut root = None;
        let mut tasks = alloc::vec![BuildTask {
            start: 0,
            end: order.len(),
            depth: 0,
            parent: None,
        }];

        while let Some(task) = tasks.pop() {
            if task.start == task.end {
                continue;
            }

            let median = self.split_at_median(&mut order[task.start..task.end], task.depth);
            let node = order[task.start + median];
            self.attach(node, task, &mut root);

            let mid = task.start + median;
            tasks.push(BuildTask {
                start: task.start,
                end: mid,
                depth: task.depth + 1,
                parent: Some((node, true)),
            });
            tasks.push(BuildTask {
                start: mid + 1,
                end: task.end,
                depth: task.depth + 1,
                parent: Some((node, false)),
            });
        }

        root
    }

    /// Sorts `range` along the axis of `depth` and returns the position of the splitting node:
    /// the first of the nodes that share the median coordinate, so that everything before it
    /// is strictly smaller, as [`KdTree::goes_left`] expects.
    #[expect(
        clippy::float_cmp,
        reason = "only exactly equal coordinates may go to the right subtree"
    )]
    fn split_at_median(&self, range: &mut [usize], depth: usize) -> usize {
        let axis = Self::axis_of(depth);
        let coordinate = |index: usize| Self::coordinate(&self.nodes[index].state, axis);
        range.sort_by(|&a, &b| coordinate(a).total_cmp(&coordinate(b)).then(a.cmp(&b)));

        let mut median = range.len() / 2;
        while median > 0 && coordinate(range[median - 1]) == coordinate(range[median]) {
            median -= 1;
        }

        median
    }

    /// Resets node `node` as the root of the subtree described by `task` and links it to its
    /// parent, or records it as the tree root.
    fn attach(&mut self, node: usize, task: BuildTask, root: &mut Option<usize>) {
        let entry = &mut self.nodes[node];
        entry.axis = Self::axis_of(task.depth);
        entry.left = None;
        entry.right = None;

        match task.parent {
            None => *root = Some(node),
            Some((parent, true)) => self.nodes[parent].left = Some(node),
            Some((parent, false)) => self.nodes[parent].right = Some(node),
        }
    }

    /// The split axis at `depth`: `depth % N`, or 0 when `N == 0`.
    const fn axis_of(depth: usize) -> usize {
        if N == 0 { 0 } else { depth % N }
    }

    /// The coordinate of `state` along `axis`; 0 when `N == 0`.
    fn coordinate(state: &[f64; N], axis: usize) -> f64 {
        if N == 0 { 0.0 } else { state[axis] }
    }

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

    /// The index of the node whose empty child slot `state` falls into, with the side, or
    /// `None` for an empty tree.
    fn attachment_point(&self, state: &[f64; N]) -> Option<(usize, bool)> {
        let mut current = self.root?;

        loop {
            let left = self.goes_left(current, state);
            let next = if left {
                self.nodes[current].left
            } else {
                self.nodes[current].right
            };

            match next {
                Some(child) => current = child,
                None => return Some((current, left)),
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

    /// Whether removed nodes outnumber live ones by enough to make a rebuild pay off.
    fn needs_rebuild(&self) -> bool {
        let tombstones = self.nodes.len() - self.live;

        tombstones >= MIN_REBUILD_TOMBSTONES && tombstones > self.live
    }
}

impl<const N: usize> NearestNeighbors<Euclidean<N>> for KdTree<N> {
    fn insert(&mut self, id: usize, state: [f64; N]) {
        let index = self.nodes.len();
        let axis = match self.attachment_point(&state) {
            None => {
                self.root = Some(index);
                0
            }
            Some((parent, left)) => {
                let parent_node = &mut self.nodes[parent];

                if left {
                    parent_node.left = Some(index);
                } else {
                    parent_node.right = Some(index);
                }

                Self::axis_of(parent_node.axis + 1)
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
        self.by_id.insert(id, index);
        self.live += 1;
    }

    fn nearest(&self, space: &Euclidean<N>, query: &[f64; N]) -> Option<usize> {
        let mut best: Option<(f64, usize)> = None;
        // Pending subtrees with a lower bound on their distance, checked when popped so that
        // the bound is compared against the best distance found by then.
        let mut stack = Vec::new();
        stack.extend(self.root.map(|root| (root, 0.0)));

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
        stack.extend(self.root);

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

    fn k_nearest(&self, space: &Euclidean<N>, query: &[f64; N], k: usize, out: &mut Vec<usize>) {
        let mut best = KBest::new(k);
        let mut stack = Vec::new();

        if k > 0 {
            stack.extend(self.root.map(|root| (root, 0.0)));
        }

        while let Some((index, lower_bound)) = stack.pop() {
            if best
                .bound()
                .is_some_and(|worst| lower_bound > worst * (1.0 + PRUNE_SLACK))
            {
                continue;
            }

            let node = &self.nodes[index];

            if !node.removed {
                best.offer(space.distance(&node.state, query), index);
            }

            let (near, far) = self.children(index, query);
            let gap = self.plane_gap(index, query);
            stack.extend(far.map(|far| (far, gap)));
            stack.extend(near.map(|near| (near, 0.0)));
        }

        out.clear();
        out.extend(
            best.into_sorted_ranks()
                .into_iter()
                .map(|index| self.nodes[index].id),
        );
    }

    fn len(&self) -> usize {
        self.live
    }
}

impl<const N: usize> super::RemovableNeighbors<Euclidean<N>> for KdTree<N> {
    /// Removes the node with `id` in `O(log n)` amortized.
    ///
    /// The node is first marked as removed and keeps splitting space. Once removed nodes
    /// outnumber live ones, the tree is rebuilt balanced over the live nodes
    /// ([`KdTree::rebalance`]), so memory and query cost track the live node count.
    fn remove(&mut self, id: usize) -> bool {
        let Some(index) = self.by_id.remove(&id) else {
            return false;
        };

        self.nodes[index].removed = true;
        self.live -= 1;

        if self.needs_rebuild() {
            self.rebalance();
        }

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{LinearScan, RemovableNeighbors, Sampler, UniformBox};
    use rand_core::{Rng, SeedableRng};

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
        let (kd, linear) = twins(300, 1);
        let mut sampler = UniformBox::new([-1.0; 3], [5.0; 3]);
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(2);

        for i in 0..500 {
            let mut query = sampler.sample(&mut rng);
            if i % 2 == 0 {
                query = query.map(libm::round);
            }

            assert_queries_agree(&kd, &linear, &query, i % 40);
        }
    }

    /// Asserts that `kd` and `linear` give identical answers to every query type at `query`.
    fn assert_queries_agree(
        kd: &KdTree<3>,
        linear: &LinearScan<[f64; 3]>,
        query: &[f64; 3],
        k: usize,
    ) {
        let space = Euclidean::<3>;
        let (mut a, mut b) = (Vec::new(), Vec::new());

        assert_eq!(kd.nearest(&space, query), linear.nearest(&space, query));
        kd.within_radius(&space, query, 1.5, &mut a);
        linear.within_radius(&space, query, 1.5, &mut b);
        assert_eq!(a, b);
        kd.k_nearest(&space, query, k, &mut a);
        linear.k_nearest(&space, query, k, &mut b);
        assert_eq!(a, b, "k = {k}");
        assert_eq!(
            NearestNeighbors::<Euclidean<3>>::len(kd),
            NearestNeighbors::<Euclidean<3>>::len(linear)
        );
    }

    #[test]
    fn random_inserts_and_removals_across_rebuilds_match_linear_scan() {
        let mut sampler = UniformBox::new([0.0; 3], [4.0; 3]);
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(9);
        let mut kd = KdTree::default();
        let mut linear = LinearScan::default();
        let mut stored = Vec::new();
        let mut next_id = 0;

        for step in 0..3000 {
            let roll = rng.next_u64() % 10;

            if roll < 4 && !stored.is_empty() {
                let pick = usize::try_from(rng.next_u64() % 1_000_000).expect("small");
                let id = stored.swap_remove(pick % stored.len());
                assert!(kd.remove(id));
                assert!(RemovableNeighbors::<Euclidean<3>>::remove(&mut linear, id));
            } else if roll < 5 && next_id > 0 && !stored.contains(&(next_id - 1)) {
                // Re-inserting a removed id makes it the newest node in both structures.
                let state = sampler.sample(&mut rng).map(libm::floor);
                kd.insert(next_id - 1, state);
                NearestNeighbors::<Euclidean<3>>::insert(&mut linear, next_id - 1, state);
                stored.push(next_id - 1);
            } else {
                let state = sampler.sample(&mut rng).map(libm::floor);
                kd.insert(next_id, state);
                NearestNeighbors::<Euclidean<3>>::insert(&mut linear, next_id, state);
                stored.push(next_id);
                next_id += 1;
            }

            let query = sampler.sample(&mut rng).map(libm::round);
            assert_queries_agree(&kd, &linear, &query, step % 25);
        }

        assert!(kd.nodes.len() < next_id, "rebuilds dropped removed nodes");
    }

    #[test]
    fn rebalance_bounds_the_depth_of_sorted_insertions() {
        let mut kd = KdTree::<2>::default();
        for id in 0..1024_u32 {
            kd.insert(
                usize::try_from(id).expect("small"),
                [f64::from(id), f64::from(id * 37 % 1024)],
            );
        }

        assert!(depth(&kd) > 50, "sorted insertion degrades the tree");
        kd.rebalance();
        assert!(depth(&kd) <= 11, "depth {}", depth(&kd));
        assert_eq!(kd.nearest(&Euclidean::<2>, &[512.2, 511.0]), Some(512));
    }

    /// The number of nodes on the longest root-to-leaf path.
    fn depth<const N: usize>(kd: &KdTree<N>) -> usize {
        let mut deepest = 0;
        let mut stack: Vec<(usize, usize)> = kd.root.map(|root| (root, 1)).into_iter().collect();

        while let Some((index, level)) = stack.pop() {
            deepest = deepest.max(level);
            let node = &kd.nodes[index];
            stack.extend(node.left.map(|child| (child, level + 1)));
            stack.extend(node.right.map(|child| (child, level + 1)));
        }

        deepest
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
