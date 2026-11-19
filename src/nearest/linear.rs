//! Brute-force nearest neighbors.

use alloc::vec::Vec;

use super::NearestNeighbors;
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

            // Strict comparison keeps the earliest node on ties and skips NaN distances.
            if best.is_none_or(|(_, best_distance)| distance < best_distance) {
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

    fn len(&self) -> usize {
        self.entries.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Euclidean;
    use alloc::vec;

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
}
