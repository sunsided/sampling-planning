//! BIT*'s vertex and edge queue entries, ordered so that a max-heap pops the best first.

use core::cmp::Ordering;

/// A vertex-queue entry: ordered by `g_T(v) + h(v)`, then `g_T(v)`, then id, all ascending.
#[derive(Debug, Clone, Copy)]
pub(super) struct VertexEntry {
    /// `g_T(v) + h(v)` when queued.
    pub(super) key: f64,
    /// `g_T(v)` when queued.
    pub(super) cost: f64,
    /// The vertex.
    pub(super) vertex: usize,
}

/// An edge-queue entry: ordered by `g_T(v) + c(v, x) + h(x)`, then `g_T(v) + c(v, x)`, then
/// `g_T(v)`, then the ids, all ascending (Gammell et al. 2020, Section 3.1).
#[derive(Debug, Clone, Copy)]
pub(super) struct EdgeEntry {
    /// `[g_T(v) + c(v, x) + h(x), g_T(v) + c(v, x), g_T(v)]` when queued.
    pub(super) key: [f64; 3],
    /// The source vertex `v`.
    pub(super) from: usize,
    /// The target state `x`, a vertex or a sample.
    pub(super) to: usize,
}

impl VertexEntry {
    /// The total order, best first.
    fn order(&self, other: &Self) -> Ordering {
        self.key
            .total_cmp(&other.key)
            .then_with(|| self.cost.total_cmp(&other.cost))
            .then_with(|| self.vertex.cmp(&other.vertex))
    }
}

impl EdgeEntry {
    /// The total order, best first.
    fn order(&self, other: &Self) -> Ordering {
        self.key[0]
            .total_cmp(&other.key[0])
            .then_with(|| self.key[1].total_cmp(&other.key[1]))
            .then_with(|| self.key[2].total_cmp(&other.key[2]))
            .then_with(|| (self.from, self.to).cmp(&(other.from, other.to)))
    }
}

/// Implements the reversed order on an entry type, so that [`alloc::collections::BinaryHeap`]
/// pops the best entry.
macro_rules! reversed_order {
    ($entry:ty) => {
        impl PartialEq for $entry {
            fn eq(&self, other: &Self) -> bool {
                self.order(other) == Ordering::Equal
            }
        }

        impl Eq for $entry {}

        impl PartialOrd for $entry {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for $entry {
            fn cmp(&self, other: &Self) -> Ordering {
                other.order(self)
            }
        }
    };
}

reversed_order!(VertexEntry);
reversed_order!(EdgeEntry);

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::collections::BinaryHeap;

    #[test]
    fn heaps_pop_the_best_entry_with_documented_tie_breaks() {
        let edge = |key, from| EdgeEntry { key, from, to: 9 };
        let mut edges = BinaryHeap::from([
            edge([2.0, 1.0, 0.5], 0),
            edge([1.0, 0.9, 0.5], 1),
            edge([1.0, 0.8, 0.7], 2),
        ]);

        assert_eq!(edges.pop().map(|e| e.from), Some(2));
        assert_eq!(edges.pop().map(|e| e.from), Some(1));

        let vertex = |key, cost, vertex| VertexEntry { key, cost, vertex };
        let mut vertices = BinaryHeap::from([vertex(1.0, 0.5, 3), vertex(1.0, 0.5, 1)]);
        assert_eq!(vertices.pop().map(|v| v.vertex), Some(1));
    }
}
