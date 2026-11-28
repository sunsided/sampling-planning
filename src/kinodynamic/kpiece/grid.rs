//! One level of KPIECE's projection grid: cells, their neighbors and their importance.

use alloc::{collections::BTreeMap, vec::Vec};

/// One instantiated grid cell.
#[derive(Debug, Clone)]
pub(super) struct Cell<const K: usize> {
    /// On the finest level, the motions whose end states project into this cell, oldest first;
    /// on coarser levels, the ids of the cells of the next finer level inside this cell.
    pub(super) members: Vec<usize>,
    /// The id of the enclosing cell on the next coarser level; `None` on the coarsest level.
    pub(super) parent: Option<usize>,
    /// The total number of integration steps of those motions; positive.
    pub(super) coverage: f64,
    /// How often the cell was selected for expansion, plus one.
    pub(super) selections: f64,
    /// Starts at `(1 + ln iteration) / (1e-3 + goal distance)` and shrinks with every expansion.
    pub(super) score: f64,
    /// The iteration in which the cell was created; at least one.
    pub(super) iteration: usize,
    /// The number of instantiated axis neighbors (cells one step away along one axis).
    pub(super) neighbors: usize,
}

impl<const K: usize> Cell<K> {
    /// `score / ((neighbors + 1) * coverage * selections)`, as in OMPL's KPIECE1.
    pub(super) fn importance(&self) -> f64 {
        let neighbors = f64::from(u32::try_from(self.neighbors).unwrap_or(u32::MAX));

        self.score / ((neighbors + 1.0) * self.coverage * self.selections)
    }

    /// Whether all `2K` axis neighbors are instantiated. Other cells are exterior.
    pub(super) const fn is_interior(&self) -> bool {
        self.neighbors >= 2 * K
    }
}

/// The grid: instantiated cells by coordinate.
#[derive(Debug, Clone)]
pub(super) struct Grid<const K: usize> {
    /// All cells, by cell id in creation order.
    pub(super) cells: Vec<Cell<K>>,
    /// Cell ids by coordinate.
    index: BTreeMap<[i64; K], usize>,
}

impl<const K: usize> Grid<K> {
    /// An empty grid.
    pub(super) const fn new() -> Self {
        Self {
            cells: Vec::new(),
            index: BTreeMap::new(),
        }
    }

    /// The cell at `coord`, if instantiated.
    pub(super) fn find(&self, coord: &[i64; K]) -> Option<usize> {
        self.index.get(coord).copied()
    }

    /// Instantiates the cell at `coord`, which must not exist yet, and updates the neighbor
    /// counts on both sides. Returns its id.
    pub(super) fn add(&mut self, coord: [i64; K], iteration: usize, score: f64) -> usize {
        let id = self.cells.len();
        let mut neighbors = 0;

        for neighbor in axis_neighbors(&coord) {
            if let Some(other) = self.find(&neighbor) {
                self.cells[other].neighbors += 1;
                neighbors += 1;
            }
        }

        self.cells.push(Cell {
            members: Vec::new(),
            parent: None,
            coverage: 0.0,
            selections: 1.0,
            score,
            iteration,
            neighbors,
        });
        self.index.insert(coord, id);

        id
    }

    /// The share of exterior cells among `among` (every cell when `None`), or zero when there
    /// are none.
    pub(super) fn exterior_fraction(&self, among: Option<&[usize]>) -> f64 {
        let to_f64 = |n: usize| f64::from(u32::try_from(n).unwrap_or(u32::MAX));
        let (exterior, total) = match among {
            None => (
                self.cells.iter().filter(|cell| !cell.is_interior()).count(),
                self.cells.len(),
            ),
            Some(ids) => (
                ids.iter()
                    .filter(|&&id| !self.cells[id].is_interior())
                    .count(),
                ids.len(),
            ),
        };

        if total == 0 {
            0.0
        } else {
            to_f64(exterior) / to_f64(total)
        }
    }

    /// The most important cell among `among` (every cell when `None`) of the preferred class
    /// (exterior or interior), falling back to the other class when it is empty; the oldest
    /// cell on ties. `None` when there are no cells.
    ///
    /// shortcut: scans every candidate, `O(cells)` per selection; OMPL keeps two binary heaps
    /// updated on every change, which pays off beyond some ten thousand cells.
    pub(super) fn select(&self, among: Option<&[usize]>, exterior: bool) -> Option<usize> {
        self.best_of(among, |cell| cell.is_interior() != exterior)
            .or_else(|| self.best_of(among, |_| true))
    }

    /// The most important cell among `among` (every cell when `None`) matching `filter`,
    /// oldest on ties.
    fn best_of(&self, among: Option<&[usize]>, filter: impl Fn(&Cell<K>) -> bool) -> Option<usize> {
        let mut best: Option<(usize, f64)> = None;
        let candidates: &mut dyn Iterator<Item = usize> = match among {
            None => &mut (0..self.cells.len()),
            Some(ids) => &mut ids.iter().copied(),
        };

        for (id, cell) in candidates
            .map(|id| (id, &self.cells[id]))
            .filter(|(_, cell)| filter(cell))
        {
            let importance = cell.importance();

            if best.is_none_or(|(_, top)| importance > top) {
                best = Some((id, importance));
            }
        }

        best.map(|(id, _)| id)
    }

    /// Raises every score by `1 + ln(creation iteration)`, after scores have decayed to almost
    /// zero, as OMPL does.
    pub(super) fn refresh_scores(&mut self) {
        for cell in &mut self.cells {
            cell.score +=
                1.0 + libm::log(f64::from(u32::try_from(cell.iteration).unwrap_or(u32::MAX)));
        }
    }
}

/// The `2K` coordinates one step away from `coord` along a single axis.
fn axis_neighbors<const K: usize>(coord: &[i64; K]) -> impl Iterator<Item = [i64; K]> + '_ {
    (0..K).flat_map(move |axis| {
        [-1, 1].into_iter().map(move |delta| {
            let mut neighbor = *coord;
            neighbor[axis] = neighbor[axis].saturating_add(delta);

            neighbor
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn neighbor_counts_track_both_sides_and_interior_cells() {
        let mut grid = Grid::<2>::new();
        let center = grid.add([0, 0], 1, 1.0);

        for coord in [[1, 0], [-1, 0], [0, 1]] {
            grid.add(coord, 1, 1.0);
        }

        assert_eq!(grid.cells[center].neighbors, 3);
        assert!(!grid.cells[center].is_interior());

        grid.add([0, -1], 1, 1.0);
        assert!(grid.cells[center].is_interior());
        assert_eq!(grid.cells[grid.find(&[1, 0]).expect("added")].neighbors, 1);
        assert!((grid.exterior_fraction(None) - 0.8).abs() < 1e-12);
        assert!((grid.exterior_fraction(Some(&[center])) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn selection_prefers_the_class_and_the_importance() {
        let mut grid = Grid::<1>::new();
        let left = grid.add([0], 1, 1.0);
        let right = grid.add([1], 1, 4.0);
        grid.add([2], 1, 1.0);

        for cell in &mut grid.cells {
            cell.coverage = 1.0;
        }

        // Cells 0 and 2 have one neighbor, cell 1 two: it is the only interior cell.
        assert_eq!(grid.select(None, false), Some(right));
        assert_eq!(
            grid.select(None, true),
            Some(left),
            "oldest exterior cell on ties"
        );
        assert_eq!(
            grid.select(Some(&[2]), false),
            Some(2),
            "falls back within the subset"
        );
    }
}
