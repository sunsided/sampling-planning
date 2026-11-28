//! A vantage-point tree: exact nearest neighbors for any metric space.

use alloc::{collections::BTreeMap, vec::Vec};
use core::cmp::Ordering;

use super::{NearestNeighbors, RemovableNeighbors, k_best::KBest};
use crate::StateSpace;

/// Relative slack added to pruning bounds.
///
/// Pruning rests on the triangle inequality over distances that were rounded separately, so
/// a bound can be off by a few ulps. Visiting a subtree too often only costs time; pruning one
/// too eagerly could change a result, so the bound errs on the visiting side.
const PRUNE_SLACK: f64 = 1e-9;

/// Subtrees of at most this many states are leaves that are scanned linearly.
const LEAF_SIZE: usize = 8;

/// The fewest states inserted since the last build that trigger a rebuild.
const MIN_REBUILD_TAIL: usize = 32;

/// The fewest removed states that trigger a rebuild, so that tiny trees do not rebuild on
/// every removal.
const MIN_REBUILD_TOMBSTONES: usize = 16;

/// A vantage-point tree over the states of any [`StateSpace`] whose distance is a metric.
///
/// [`KdTree`](crate::KdTree) needs Euclidean coordinates; a vantage-point tree only needs
/// distances. Each inner node picks a vantage state and splits the others at the median
/// distance `mu` to it, and queries skip a side whenever the triangle inequality shows that
/// nothing in it can be close enough. This works for SO(2), SE(2), SE(3) and Reeds-Shepp
/// spaces, where [`LinearScan`](crate::LinearScan) is otherwise the only choice, and turns
/// its `O(n)` queries into roughly `O(log n)` ones in low intrinsic dimension.
///
/// The space's distance must be symmetric and satisfy the triangle inequality, up to
/// rounding. [`Euclidean`](crate::Euclidean), [`So2`](crate::So2), [`Se2`](crate::Se2),
/// [`Se3`](crate::Se3) and [`ReedsShepp`](crate::ReedsShepp) do; [`Dubins`](crate::Dubins)
/// does not (it is not symmetric), and [`Riemannian`](crate::Riemannian) only approximately.
/// For such spaces use [`LinearScan`](crate::LinearScan).
///
/// The tree keeps its own copy of the space, because insertions rebuild parts of it and
/// [`NearestNeighbors::insert`] gets no space: queries must pass the same space. New states
/// go into an unindexed tail first; once the tail holds more than half as many states as the
/// tree, the whole structure is rebuilt, so a rebuild costs `O(n log n)` distances, amortized
/// `O(log^2 n)` per insertion. Removal ([`RemovableNeighbors`]) marks states and rebuilds once
/// removed states outnumber live ones.
///
/// Queries return exactly what [`LinearScan`](crate::LinearScan) returns, including its tie
/// breaking toward the earliest inserted state and its handling of NaN distances, because
/// both compare the same [`StateSpace::distance`] values.
///
/// # Examples
///
/// ```
/// use sampling_planning::{NearestNeighbors, Se3, Se3State, VpTree};
///
/// let space = Se3::new(0.5);
/// let mut tree = VpTree::new(space);
/// for (id, x) in [0.0, 1.0, 2.0, 3.0].into_iter().enumerate() {
///     tree.insert(id, Se3State::at([x, 0.0, 0.0]));
/// }
///
/// assert_eq!(tree.nearest(&space, &Se3State::at([2.2, 0.0, 0.0])), Some(2));
/// ```
#[derive(Debug, Clone)]
pub struct VpTree<Sp: StateSpace> {
    /// The space whose distance builds the tree; queries must use the same metric.
    space: Sp,
    /// All stored states, removed ones included until the next rebuild, in insertion order.
    ///
    /// The position of an entry is its insertion rank, which breaks distance ties. Rebuilds
    /// drop removed entries but keep the relative order of the rest.
    entries: Vec<Entry<Sp::State>>,
    /// The tree nodes; the tree indexes the entries before `built`.
    nodes: Vec<Node>,
    /// The index of the root node in `nodes`; `None` while nothing is indexed.
    root: Option<usize>,
    /// The number of leading entries the tree indexes; later ones form the scanned tail.
    built: usize,
    /// The number of entries that were not removed.
    live: usize,
    /// The rank of the earliest live entry of each id, for `O(log n)` removal.
    by_id: BTreeMap<usize, usize>,
    /// Whether an id was ever inserted twice; removal then also looks for a later entry.
    duplicates: bool,
}

/// One stored state.
#[derive(Debug, Clone)]
struct Entry<S> {
    /// The caller's id.
    id: usize,
    /// The state.
    state: S,
    /// Whether the entry was removed. Removed vantage states still split but never match.
    removed: bool,
}

/// One node of a [`VpTree`].
#[derive(Debug, Clone)]
enum Node {
    /// An inner node: the entries at distance at most the median `mu` from the vantage
    /// state go inside, those at distance at least `mu` outside.
    Split {
        /// The rank of the vantage state.
        vantage: usize,
        /// The smallest and largest distance of an inside state to the vantage state.
        inside_range: [f64; 2],
        /// The smallest and largest distance of an outside state to the vantage state.
        outside_range: [f64; 2],
        /// The subtree within `mu`, if any.
        inside: Option<usize>,
        /// The subtree beyond `mu`, if any.
        outside: Option<usize>,
    },
    /// A leaf: ranks that are scanned linearly.
    Leaf(Vec<usize>),
}

/// A pending subtree of a build: a range of the rank buffer and where its root attaches.
#[derive(Debug, Clone, Copy)]
struct BuildTask {
    /// The first position of the range.
    start: usize,
    /// One past the last position of the range.
    end: usize,
    /// The parent split node and whether the subtree is its inside; `None` for the root.
    parent: Option<(usize, bool)>,
}

impl<Sp: StateSpace + Default> Default for VpTree<Sp> {
    fn default() -> Self {
        Self::new(Sp::default())
    }
}

impl<Sp: StateSpace> VpTree<Sp> {
    /// An empty tree over `space`. Queries must pass the same space.
    #[must_use]
    pub const fn new(space: Sp) -> Self {
        Self {
            space,
            entries: Vec::new(),
            nodes: Vec::new(),
            root: None,
            built: 0,
            live: 0,
            by_id: BTreeMap::new(),
            duplicates: false,
        }
    }

    /// Rebuilds the tree over all live entries, dropping removed ones; insertion ranks keep
    /// their relative order, so query results do not change.
    fn rebuild(&mut self) {
        if self.live < self.entries.len() {
            self.entries.retain(|entry| !entry.removed);
            self.by_id.clear();
            for (rank, entry) in self.entries.iter().enumerate() {
                self.by_id.entry(entry.id).or_insert(rank);
            }
        }

        self.nodes.clear();
        self.built = self.entries.len();
        self.root = self.build();
    }

    /// Links all entries into a tree and returns the root index.
    fn build(&mut self) -> Option<usize> {
        let mut ranks: Vec<usize> = (0..self.entries.len()).collect();
        let mut distances = Vec::new();
        let mut root = None;
        let mut tasks = alloc::vec![BuildTask {
            start: 0,
            end: ranks.len(),
            parent: None,
        }];

        while let Some(task) = tasks.pop() {
            if task.start == task.end {
                continue;
            }

            let range = &mut ranks[task.start..task.end];
            if range.len() <= LEAF_SIZE {
                self.push_node(Node::Leaf(range.to_vec()), task.parent, &mut root);
                continue;
            }

            let (vantage, median) = self.split(range, &mut distances);
            let node = self.push_node(
                Node::Split {
                    vantage,
                    inside_range: distance_range(&distances[..median]),
                    outside_range: distance_range(&distances[median..]),
                    inside: None,
                    outside: None,
                },
                task.parent,
                &mut root,
            );

            // The vantage state sits at the start of the range; the rest is split at `median`.
            let mid = task.start + 1 + median;
            tasks.push(BuildTask {
                start: task.start + 1,
                end: mid,
                parent: Some((node, true)),
            });
            tasks.push(BuildTask {
                start: mid,
                end: task.end,
                parent: Some((node, false)),
            });
        }

        root
    }

    /// Keeps the first rank of `range` as the vantage state and partitions the rest by
    /// distance to it, leaving the partitioned `(distance, rank)` pairs in `distances`: the
    /// first `median` of them lie within the median distance `mu`, the others at `mu` or
    /// beyond. Returns `(vantage rank, median)`.
    ///
    /// Planners insert random samples, so the first rank is a random member of the subtree.
    /// Choosing a far-out vantage state instead did not prune measurably better.
    fn split(&self, range: &mut [usize], distances: &mut Vec<(f64, usize)>) -> (usize, usize) {
        let vantage = range[0];
        let vantage_state = &self.entries[vantage].state;

        distances.clear();
        distances.extend(range[1..].iter().map(|&rank| {
            let distance = self
                .space
                .distance(&self.entries[rank].state, vantage_state);
            (distance, rank)
        }));

        let median = distances.len() / 2;
        distances.select_nth_unstable_by(median, |a, b| by_distance(*a, *b));

        for (slot, &(_, rank)) in range[1..].iter_mut().zip(distances.iter()) {
            *slot = rank;
        }

        // Everything before `median` is at most `mu`; the median itself starts the outside.
        (vantage, median)
    }

    /// Appends `node`, links it to its parent (or records it as the root) and returns its
    /// index.
    fn push_node(
        &mut self,
        node: Node,
        parent: Option<(usize, bool)>,
        root: &mut Option<usize>,
    ) -> usize {
        let index = self.nodes.len();
        self.nodes.push(node);

        match parent {
            None => *root = Some(index),
            Some((parent, inside_side)) => {
                if let Node::Split {
                    inside, outside, ..
                } = &mut self.nodes[parent]
                {
                    if inside_side {
                        *inside = Some(index);
                    } else {
                        *outside = Some(index);
                    }
                }
            }
        }

        index
    }

    /// Visits every live entry that a query with the current `bound` cannot rule out, calling
    /// `visit(distance, rank)` with its distance to `query`. `bound()` is read before each
    /// subtree, so a shrinking bound prunes more as the search goes.
    fn search(
        &self,
        space: &Sp,
        query: &Sp::State,
        bound: impl Fn() -> f64,
        mut visit: impl FnMut(f64, usize),
    ) {
        let mut stack: Vec<(usize, f64)> = Vec::new();
        stack.extend(self.root.map(|root| (root, 0.0)));

        while let Some((index, lower)) = stack.pop() {
            if is_beyond(lower, bound()) {
                continue;
            }

            match &self.nodes[index] {
                Node::Leaf(ranks) => {
                    for &rank in ranks {
                        self.visit_rank(space, query, rank, &mut visit);
                    }
                }
                &Node::Split {
                    vantage,
                    inside_range,
                    outside_range,
                    inside,
                    outside,
                } => {
                    let d = space.distance(&self.entries[vantage].state, query);
                    if !self.entries[vantage].removed {
                        visit(d, vantage);
                    }

                    // A state at distance `x` in `[lo, hi]` from the vantage state is at least
                    // `max(lo - d, d - hi)` from the query, by the triangle inequality. The
                    // nearer side is pushed last, so it is searched first.
                    let inside = inside.map(|node| (node, ring_gap(inside_range, d)));
                    let outside = outside.map(|node| (node, ring_gap(outside_range, d)));
                    let (near, far) = if d <= outside_range[0] {
                        (inside, outside)
                    } else {
                        (outside, inside)
                    };
                    stack.extend(far);
                    stack.extend(near);
                }
            }
        }

        for rank in self.built..self.entries.len() {
            self.visit_rank(space, query, rank, &mut visit);
        }
    }

    /// Calls `visit` with the distance of the entry `rank` to `query`, unless it was removed.
    fn visit_rank(
        &self,
        space: &Sp,
        query: &Sp::State,
        rank: usize,
        visit: &mut impl FnMut(f64, usize),
    ) {
        let entry = &self.entries[rank];

        if !entry.removed {
            visit(space.distance(&entry.state, query), rank);
        }
    }

    /// Whether the unindexed tail has grown enough to rebuild.
    fn needs_tail_rebuild(&self) -> bool {
        let tail = self.entries.len() - self.built;

        tail >= MIN_REBUILD_TAIL && 2 * tail > self.built
    }

    /// Whether removed entries outnumber live ones by enough to make a rebuild pay off.
    fn needs_tombstone_rebuild(&self) -> bool {
        let tombstones = self.entries.len() - self.live;

        tombstones >= MIN_REBUILD_TOMBSTONES && tombstones > self.live
    }
}

impl<Sp: StateSpace> NearestNeighbors<Sp> for VpTree<Sp> {
    fn insert(&mut self, id: usize, state: Sp::State) {
        let rank = self.entries.len();

        self.entries.push(Entry {
            id,
            state,
            removed: false,
        });
        self.live += 1;
        self.duplicates |= self.by_id.contains_key(&id);
        self.by_id.entry(id).or_insert(rank);

        if self.needs_tail_rebuild() {
            self.rebuild();
        }
    }

    fn nearest(&self, space: &Sp, query: &Sp::State) -> Option<usize> {
        let mut out = Vec::with_capacity(1);
        self.k_nearest(space, query, 1, &mut out);

        out.first().copied()
    }

    fn within_radius(&self, space: &Sp, query: &Sp::State, radius: f64, out: &mut Vec<usize>) {
        let mut ranks = Vec::new();
        self.search(
            space,
            query,
            || radius,
            |distance, rank| {
                if distance <= radius {
                    ranks.push(rank);
                }
            },
        );

        ranks.sort_unstable();
        out.clear();
        out.extend(ranks.into_iter().map(|rank| self.entries[rank].id));
    }

    fn k_nearest(&self, space: &Sp, query: &Sp::State, k: usize, out: &mut Vec<usize>) {
        let best = core::cell::RefCell::new(KBest::new(k));
        self.search(
            space,
            query,
            || best.borrow().bound().unwrap_or(f64::INFINITY),
            |distance, rank| best.borrow_mut().offer(distance, rank),
        );

        out.clear();
        out.extend(
            best.into_inner()
                .into_sorted_ranks()
                .into_iter()
                .map(|rank| self.entries[rank].id),
        );
    }

    fn len(&self) -> usize {
        self.live
    }
}

impl<Sp: StateSpace> RemovableNeighbors<Sp> for VpTree<Sp> {
    /// Removes the earliest live entry with `id`, like [`LinearScan`](crate::LinearScan),
    /// in `O(log n)` while no id was inserted twice.
    fn remove(&mut self, id: usize) -> bool {
        let Some(rank) = self.by_id.remove(&id) else {
            return false;
        };

        self.entries[rank].removed = true;
        self.live -= 1;

        let later = (rank + 1..self.entries.len())
            .filter(|_| self.duplicates)
            .find(|&later| !self.entries[later].removed && self.entries[later].id == id);
        if let Some(next) = later {
            self.by_id.insert(id, next);
        }

        if self.needs_tombstone_rebuild() {
            self.rebuild();
        }

        true
    }
}

/// Orders `(distance, rank)` pairs by distance, then rank; NaN distances last.
fn by_distance(a: (f64, usize), b: (f64, usize)) -> Ordering {
    a.0.total_cmp(&b.0).then(a.1.cmp(&b.1))
}

/// The smallest and largest distance among `pairs`; `[inf, -inf]` when empty.
fn distance_range(pairs: &[(f64, usize)]) -> [f64; 2] {
    // A NaN distance poisons the range for good, so that it never prunes.
    pairs
        .iter()
        .fold([f64::INFINITY, f64::NEG_INFINITY], |[lo, hi], &(d, _)| {
            if d.is_nan() || lo.is_nan() {
                [f64::NAN, f64::NAN]
            } else {
                [lo.min(d), hi.max(d)]
            }
        })
}

/// The triangle-inequality lower bound on the distance from a query at distance `d` from a
/// vantage state to any state whose distance to that vantage state lies in `range`.
fn ring_gap(range: [f64; 2], d: f64) -> f64 {
    (range[0] - d).max(d - range[1])
}

/// Whether a subtree whose states are at least `lower` away can be skipped for `bound`.
fn is_beyond(lower: f64, bound: f64) -> bool {
    lower - PRUNE_SLACK * libm::fabs(lower).max(1.0) > bound
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Euclidean, LinearScan, So2};
    use alloc::vec;

    /// The 1D points 0, 2, 4 and 2 again with ids 10 to 13, as in the linear-scan tests.
    fn points() -> VpTree<Euclidean<1>> {
        let mut tree = VpTree::default();
        for (id, x) in [(10, 0.0), (11, 2.0), (12, 4.0), (13, 2.0)] {
            tree.insert(id, [x]);
        }

        tree
    }

    #[test]
    fn queries_break_ties_by_insertion_like_linear_scan() {
        let tree = points();
        let space = Euclidean::<1>;
        let mut out = Vec::new();

        assert_eq!(tree.nearest(&space, &[2.1]), Some(11));
        tree.within_radius(&space, &[2.0], 2.0, &mut out);
        assert_eq!(out, vec![10, 11, 12, 13]);
        tree.k_nearest(&space, &[2.0], 3, &mut out);
        assert_eq!(out, vec![11, 13, 10]);
    }

    #[test]
    fn empty_trees_have_no_neighbors() {
        let tree = VpTree::<Euclidean<2>>::default();
        let mut out = vec![1];

        assert_eq!(tree.nearest(&Euclidean::<2>, &[0.0, 0.0]), None);
        tree.within_radius(&Euclidean::<2>, &[0.0, 0.0], 1.0, &mut out);
        assert_eq!(out, Vec::<usize>::new());
        assert!(NearestNeighbors::<Euclidean<2>>::is_empty(&tree));
    }

    #[test]
    fn large_trees_rebuild_and_agree_with_a_linear_scan() {
        let space = So2;
        let mut tree = VpTree::new(space);
        let mut linear = LinearScan::default();

        // A deterministic spread of angles, wrapped and with exact repeats.
        for id in 0..2000_usize {
            let x = f64::from(u32::try_from(id).expect("small"));
            let angle = libm::sin(x * 12.9898) * 10.0 + libm::fmod(x, 7.0);
            tree.insert(id, angle);
            NearestNeighbors::<So2>::insert(&mut linear, id, angle);
        }

        assert!(tree.built > 1000, "the tail was folded into the tree");

        for removed in (0..2000).step_by(3) {
            assert!(tree.remove(removed));
            assert!(RemovableNeighbors::<So2>::remove(&mut linear, removed));
        }

        assert!(!tree.remove(0), "already removed");
        assert_eq!(tree.len(), NearestNeighbors::<So2>::len(&linear));

        let (mut a, mut b) = (Vec::new(), Vec::new());
        for step in 0..200 {
            let query = f64::from(step) * 0.0317 - 3.0;
            assert_eq!(tree.nearest(&space, &query), linear.nearest(&space, &query));
            tree.within_radius(&space, &query, 0.05, &mut a);
            linear.within_radius(&space, &query, 0.05, &mut b);
            assert_eq!(a, b);
            tree.k_nearest(&space, &query, 7, &mut a);
            linear.k_nearest(&space, &query, 7, &mut b);
            assert_eq!(a, b);
        }
    }

    #[test]
    fn removing_a_duplicated_id_removes_the_earliest_entry_first() {
        let mut tree = VpTree::<Euclidean<1>>::default();
        tree.insert(5, [0.0]);
        tree.insert(5, [3.0]);

        assert!(tree.remove(5));
        assert_eq!(tree.nearest(&Euclidean::<1>, &[0.0]), Some(5));
        assert!(tree.remove(5));
        assert_eq!(tree.nearest(&Euclidean::<1>, &[0.0]), None);
        assert!(!tree.remove(5));
    }
}
