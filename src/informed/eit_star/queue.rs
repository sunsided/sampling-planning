//! EIT*'s forward queue: edges with stored keys, ordered both by effort and by lower bound.

use alloc::collections::{BTreeMap, BTreeSet};
use core::cmp::Ordering;

/// The keys of one forward edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ForwardKey {
    /// The lower bound `g(v) + c_hat + h(x)` on a solution through the edge.
    pub(super) bound: f64,
    /// The checks still needed along the edge and from its target to the goal.
    pub(super) effort: usize,
}

/// An `f64` ordered by [`f64::total_cmp`], so it can key ordered collections.
#[derive(Debug, Clone, Copy)]
struct Ordered(f64);

impl PartialEq for Ordered {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Ordered {}

impl PartialOrd for Ordered {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ordered {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Directed edges `(vertex, target)` with their keys, ordered two ways for `O(log n)` access
/// to the least-effort and the lowest-bound edge.
#[derive(Debug, Clone, Default)]
pub(super) struct ForwardQueue {
    /// The key of every queued edge.
    keys: BTreeMap<(usize, usize), ForwardKey>,
    /// The edges by effort, then bound, then ids.
    by_effort: BTreeSet<(usize, Ordered, usize, usize)>,
    /// The edges by bound, then ids.
    by_bound: BTreeSet<(Ordered, usize, usize)>,
}

impl ForwardQueue {
    /// Whether no edge is queued.
    pub(super) fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// The number of queued edges.
    pub(super) fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether `edge` is queued.
    pub(super) fn contains(&self, edge: (usize, usize)) -> bool {
        self.keys.contains_key(&edge)
    }

    /// Queues `edge` with `key`, replacing its old key. Returns whether it was new.
    pub(super) fn insert(&mut self, edge: (usize, usize), key: ForwardKey) -> bool {
        let new = self.remove(edge).is_none();

        self.keys.insert(edge, key);
        self.by_effort
            .insert((key.effort, Ordered(key.bound), edge.0, edge.1));
        self.by_bound.insert((Ordered(key.bound), edge.0, edge.1));

        new
    }

    /// Removes `edge` and returns its key, if it was queued.
    pub(super) fn remove(&mut self, edge: (usize, usize)) -> Option<ForwardKey> {
        let key = self.keys.remove(&edge)?;

        self.by_effort
            .remove(&(key.effort, Ordered(key.bound), edge.0, edge.1));
        self.by_bound.remove(&(Ordered(key.bound), edge.0, edge.1));

        Some(key)
    }

    /// Removes every edge.
    pub(super) fn clear(&mut self) {
        self.keys.clear();
        self.by_effort.clear();
        self.by_bound.clear();
    }

    /// Every queued edge, in id order.
    pub(super) fn edges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.keys.keys().copied()
    }

    /// The edge with the lowest bound and that bound, ties by ids.
    pub(super) fn lowest_bound(&self) -> Option<((usize, usize), f64)> {
        self.by_bound
            .first()
            .map(|&(bound, from, to)| ((from, to), bound.0))
    }

    /// The edge the forward search takes next (OMPL's `ForwardQueue::getFrontIter`): the
    /// least-effort edge whose bound is within `suboptimality` times the lowest bound, ties by
    /// bound, or the lowest-bound edge when that edge is not strictly within it.
    ///
    /// shortcut: a finite `suboptimality` scans the edges in effort order until one is within
    /// the threshold, `O(n)` in the worst case, as OMPL's scan always is; infinite
    /// suboptimality takes the first edge in `O(log n)`.
    pub(super) fn front(&self, suboptimality: f64) -> Option<(usize, usize)> {
        let (lowest_edge, lowest) = self.lowest_bound()?;
        let threshold = if suboptimality.is_finite() && lowest.is_finite() {
            suboptimality * lowest
        } else {
            f64::INFINITY
        };
        let cheapest = self
            .by_effort
            .iter()
            .find(|(_, bound, _, _)| bound.0 <= threshold);

        match cheapest {
            Some(&(_, bound, from, to)) if bound.0 < threshold => Some((from, to)),
            _ => Some(lowest_edge),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn front_prefers_effort_within_the_suboptimality_bound() {
        let mut queue = ForwardQueue::default();
        let key = |bound, effort| ForwardKey { bound, effort };
        queue.insert((0, 1), key(10.0, 50));
        queue.insert((0, 2), key(12.0, 5));
        queue.insert((0, 3), key(30.0, 1));

        assert_eq!(queue.front(f64::INFINITY), Some((0, 3)));
        assert_eq!(queue.front(1.5), Some((0, 2)));
        assert_eq!(
            queue.front(1.0),
            Some((0, 1)),
            "only the lowest bound is within"
        );

        queue.insert((0, 3), key(9.0, 1));
        assert_eq!(queue.lowest_bound(), Some(((0, 3), 9.0)));
        assert_eq!(queue.remove((0, 2)), Some(key(12.0, 5)));
        assert_eq!(queue.len(), 2);
    }
}
