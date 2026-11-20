//! A bounded selection of the `k` best `(distance, rank)` candidates.

use alloc::{collections::BinaryHeap, vec::Vec};
use core::cmp::Ordering;

/// One candidate: a distance and the insertion rank that breaks ties.
#[derive(Debug, Clone, Copy)]
pub(super) struct Candidate {
    /// The distance to the query; never NaN.
    pub(super) distance: f64,
    /// The insertion rank of the stored state; lower ranks win ties.
    pub(super) rank: usize,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Candidate {}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Candidate {
    /// By distance, then by rank, both ascending: the order in which `k_nearest` reports.
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance
            .total_cmp(&other.distance)
            .then_with(|| self.rank.cmp(&other.rank))
    }
}

/// Keeps the `k` smallest candidates seen so far under the [`Candidate`] order.
///
/// A max-heap of at most `k` entries, so offering `n` candidates costs `O(n log k)`. Both
/// [`LinearScan`](crate::LinearScan) and [`KdTree`](crate::KdTree) select through this type, so
/// they agree on ties, including ties at the `k`-th place.
#[derive(Debug, Clone)]
pub(super) struct KBest {
    /// The kept candidates; the worst one is on top.
    heap: BinaryHeap<Candidate>,
    /// The number of candidates to keep.
    k: usize,
}

impl KBest {
    /// An empty selection that keeps `k` candidates.
    pub(super) fn new(k: usize) -> Self {
        Self {
            heap: BinaryHeap::with_capacity(k.saturating_add(1).min(1024)),
            k,
        }
    }

    /// Offers a candidate. NaN distances are ignored, like in [`NearestNeighbors::nearest`].
    ///
    /// [`NearestNeighbors::nearest`]: super::NearestNeighbors::nearest
    pub(super) fn offer(&mut self, distance: f64, rank: usize) {
        if distance.is_nan() || self.k == 0 {
            return;
        }

        let candidate = Candidate { distance, rank };

        if self.heap.len() < self.k {
            self.heap.push(candidate);
        } else if self.heap.peek().is_some_and(|worst| candidate < *worst) {
            self.heap.pop();
            self.heap.push(candidate);
        }
    }

    /// The distance a candidate must not exceed to enter, or `None` while fewer than `k` are
    /// kept (then everything enters).
    pub(super) fn bound(&self) -> Option<f64> {
        if self.heap.len() < self.k {
            None
        } else {
            self.heap.peek().map(|worst| worst.distance)
        }
    }

    /// The kept ranks, best first.
    pub(super) fn into_sorted_ranks(self) -> Vec<usize> {
        self.heap
            .into_sorted_vec()
            .into_iter()
            .map(|candidate| candidate.rank)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_k_smallest_and_breaks_ties_by_rank() {
        let mut best = KBest::new(3);

        for (distance, rank) in [
            (2.0, 0),
            (1.0, 1),
            (f64::NAN, 2),
            (2.0, 3),
            (0.5, 4),
            (2.0, 5),
        ] {
            best.offer(distance, rank);
        }

        assert_eq!(best.bound(), Some(2.0));
        assert_eq!(best.into_sorted_ranks(), [4, 1, 0]);
    }

    #[test]
    fn zero_k_keeps_nothing() {
        let mut best = KBest::new(0);
        best.offer(1.0, 0);

        assert_eq!(best.into_sorted_ranks(), [] as [usize; 0]);
    }
}
