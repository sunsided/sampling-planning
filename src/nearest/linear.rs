//! Brute-force nearest neighbors.

use alloc::vec::Vec;

use super::{NearestNeighbors, RemovableNeighbors, k_best::KBest};
use crate::StateSpace;

/// Checks every stored state on each query: O(n) per query, no preprocessing.
///
/// Exact for any metric and fine for a few thousand nodes. Use a spatial index for more.
#[derive(Debug, Clone)]
pub struct LinearScan<S> {
    /// Stored nodes as `(id, state)` in insertion order.
    entries: Vec<(usize, S)>,
}

impl<S> Default for LinearScan<S> {
    fn default() -> Self {
        Self {
            entries: Vec::new(),
        }
    }
}

impl<Sp: StateSpace> NearestNeighbors<Sp> for LinearScan<Sp::State> {
    fn insert(&mut self, id: usize, state: Sp::State) {
        self.entries.push((id, state));
    }

    fn nearest(&self, space: &Sp, query: &Sp::State) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;

        for (id, state) in &self.entries {
            let distance = space.distance(state, query);

            // Strict comparison keeps the earliest node on ties; NaN distances never win.
            if !distance.is_nan() && best.is_none_or(|(_, best_distance)| distance < best_distance)
            {
                best = Some((*id, distance));
            }
        }

        best.map(|(id, _)| id)
    }

    fn within_radius(&self, space: &Sp, query: &Sp::State, radius: f64, out: &mut Vec<usize>) {
        out.clear();
        out.extend(
            self.entries
                .iter()
                .filter(|(_, state)| space.distance(state, query) <= radius)
                .map(|(id, _)| *id),
        );
    }

    fn k_nearest(&self, space: &Sp, query: &Sp::State, k: usize, out: &mut Vec<usize>) {
        let mut best = KBest::new(k);

        for (rank, (_, state)) in self.entries.iter().enumerate() {
            best.offer(space.distance(state, query), rank);
        }

        out.clear();
        out.extend(
            best.into_sorted_ranks()
                .into_iter()
                .map(|rank| self.entries[rank].0),
        );
    }

    fn len(&self) -> usize {
        self.entries.len()
    }
}

impl<Sp: StateSpace> RemovableNeighbors<Sp> for LinearScan<Sp::State> {
    /// Removes the node with `id` in `O(n)`, keeping the insertion order of the rest.
    fn remove(&mut self, id: usize) -> bool {
        let Some(index) = self.entries.iter().position(|(entry, _)| *entry == id) else {
            return false;
        };

        self.entries.remove(index);

        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Euclidean;
    use alloc::vec;

    #[test]
    fn a_nan_distance_never_wins_nearest() {
        use crate::{KdTree, VpTree};

        let space = Euclidean::<1>;
        let mut linear = LinearScan::default();
        let mut kd = KdTree::default();
        let mut vp = VpTree::default();

        for (id, state) in [[f64::NAN], [1.0]].into_iter().enumerate() {
            NearestNeighbors::<Euclidean<1>>::insert(&mut linear, id, state);
            kd.insert(id, state);
            NearestNeighbors::<Euclidean<1>>::insert(&mut vp, id, state);
        }

        assert_eq!(linear.nearest(&space, &[1.0]), Some(1));
        assert_eq!(kd.nearest(&space, &[1.0]), Some(1));
        assert_eq!(vp.nearest(&space, &[1.0]), Some(1));
    }

    /// A structure holding the 1D points 0, 2, 4 and 2 again with ids 10 to 13.
    fn points() -> LinearScan<[f64; 1]> {
        let mut nn = LinearScan::default();

        for (id, x) in [(10, 0.0), (11, 2.0), (12, 4.0), (13, 2.0)] {
            NearestNeighbors::<Euclidean<1>>::insert(&mut nn, id, [x]);
        }

        nn
    }

    #[test]
    fn nearest_prefers_earliest_on_ties() {
        let nn = points();

        assert_eq!(nn.nearest(&Euclidean::<1>, &[2.1]), Some(11));
        assert_eq!(nn.nearest(&Euclidean::<1>, &[-5.0]), Some(10));
        assert_eq!(NearestNeighbors::<Euclidean<1>>::len(&nn), 4);
    }

    #[test]
    fn nearest_of_empty_is_none() {
        let nn = LinearScan::<[f64; 1]>::default();

        assert_eq!(nn.nearest(&Euclidean::<1>, &[0.0]), None);
    }

    #[test]
    fn within_radius_is_inclusive_and_ordered() {
        let mut out = vec![99];
        points().within_radius(&Euclidean::<1>, &[3.0], 1.0, &mut out);

        assert_eq!(out, [11, 12, 13]);
    }

    #[test]
    fn k_nearest_orders_by_distance_then_insertion() {
        let nn = points();
        let mut out = vec![99];

        nn.k_nearest(&Euclidean::<1>, &[2.0], 3, &mut out);
        assert_eq!(
            out,
            [11, 13, 10],
            "the tie at the cut goes to the earlier node"
        );

        nn.k_nearest(&Euclidean::<1>, &[2.9], 10, &mut out);
        assert_eq!(out, [11, 13, 12, 10]);

        nn.k_nearest(&Euclidean::<1>, &[2.9], 0, &mut out);
        assert_eq!(out, [] as [usize; 0]);
    }

    #[test]
    fn remove_forgets_one_node() {
        let mut nn = points();

        assert!(RemovableNeighbors::<Euclidean<1>>::remove(&mut nn, 11));
        assert!(!RemovableNeighbors::<Euclidean<1>>::remove(&mut nn, 11));
        assert_eq!(nn.nearest(&Euclidean::<1>, &[2.1]), Some(13));
    }
}
