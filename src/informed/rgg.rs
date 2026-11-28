//! Radius neighborhoods of the batch graph, carried from batch to batch.
//!
//! AIT* and EIT* start a new batch whenever the graph holds no better path, often every few
//! steps, and each batch restarts a reverse search that touches nearly every state. Querying
//! every touched state's radius neighborhood again each batch made those queries the bulk of
//! the planners' time. Between two batches the graph only changes by the new samples, the
//! pruned states and a radius that usually shrinks, so a state's list from an earlier batch,
//! filtered to the new radius, plus the new samples near it, is exactly what a fresh query
//! would return.

use alloc::vec::Vec;

/// Per-state radius neighbor lists that stay valid across batches while the radius does not
/// grow.
///
/// A list is a sorted superset of the state's current neighbors: everything within the radius
/// of the batch it was built in, plus every later sample within the radius of its own batch.
/// [`NeighborCache::filter`] narrows it to the current batch on first use. When the radius
/// grows (pruning lowered the state count), every list is dropped and rebuilt by a fresh
/// query on demand.
#[derive(Debug, Clone, Default)]
pub(super) struct NeighborCache {
    /// The neighbor ids per state, sorted ascending.
    lists: Vec<Vec<usize>>,
    /// The generation each list belongs to; a list is usable while it equals `generation`.
    built: Vec<usize>,
    /// The batch each list was last narrowed to.
    filtered: Vec<usize>,
    /// The current generation; bumped whenever the radius grows. Starts at one, so that zero
    /// marks states without a list.
    generation: usize,
    /// The current batch, as counted by the planner.
    batch: usize,
    /// The radius of the current batch; NaN before the first.
    radius: f64,
}

impl NeighborCache {
    /// Starts batch `batch` over `states` states with connection radius `radius`. Returns
    /// whether the lists of earlier batches stay usable; if not, all are dropped.
    pub(super) fn begin_batch(&mut self, batch: usize, states: usize, radius: f64) -> bool {
        let keep = self.generation > 0 && radius <= self.radius;
        if !keep {
            self.generation += 1;
        }

        self.batch = batch;
        self.radius = radius;
        self.lists.resize_with(states, Vec::new);
        self.built.resize(states, 0);
        self.filtered.resize(states, 0);

        keep
    }

    /// Records `neighbors`, the sorted result of a fresh query in this batch, as the list of
    /// `id`.
    pub(super) fn store(&mut self, id: usize, neighbors: &[usize]) {
        let list = &mut self.lists[id];
        list.clear();
        list.extend_from_slice(neighbors);
        self.built[id] = self.generation;
        self.filtered[id] = self.batch;
    }

    /// Appends the new sample `sample` to the lists of the older states `older`, which a fresh
    /// query from `sample` found within this batch's radius. Sample ids grow with every batch,
    /// so the lists stay sorted when samples are added in ascending order.
    pub(super) fn add_sample(&mut self, sample: usize, older: &[usize]) {
        for &x in older {
            if self.built[x] == self.generation {
                self.lists[x].push(sample);
            }
        }
    }

    /// The list of `id`, narrowed by `keep` (a state is kept if it is live and within this
    /// batch's radius) the first time it is used in this batch, or `None` when `id` needs a
    /// fresh query.
    pub(super) fn filter(&mut self, id: usize, keep: impl Fn(usize) -> bool) -> Option<&[usize]> {
        if self.built[id] != self.generation {
            return None;
        }

        if self.filtered[id] != self.batch {
            self.lists[id].retain(|&y| keep(y));
            self.filtered[id] = self.batch;
        }

        Some(&self.lists[id])
    }
}

/// Inserts the ids of `extra` into the sorted list `list`, keeping it sorted and free of
/// duplicates.
pub(super) fn merge_sorted(list: &mut Vec<usize>, extra: &[usize]) {
    for &id in extra {
        if let Err(position) = list.binary_search(&id) {
            list.insert(position, id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn lists_survive_a_shrinking_radius_and_drop_with_a_growing_one() {
        let mut cache = NeighborCache::default();
        assert!(!cache.begin_batch(1, 3, 2.0), "nothing to keep at first");
        cache.store(0, &[1, 2]);
        assert!(cache.filter(1, |_| true).is_none(), "1 needs a query");

        assert!(cache.begin_batch(2, 5, 1.5));
        cache.add_sample(3, &[0]);
        cache.add_sample(4, &[0, 1]);
        assert_eq!(cache.filter(0, |y| y != 2), Some(&[1, 3, 4][..]));
        assert_eq!(
            cache.filter(0, |_| false),
            Some(&[1, 3, 4][..]),
            "filtered once a batch"
        );

        assert!(
            !cache.begin_batch(3, 5, 1.6),
            "a larger radius drops every list"
        );
        assert!(cache.filter(0, |_| true).is_none());
    }

    #[test]
    fn merging_keeps_the_list_sorted_and_unique() {
        let mut list = vec![1, 4, 9];
        merge_sorted(&mut list, &[4, 0, 7]);

        assert_eq!(list, vec![0, 1, 4, 7, 9]);
    }
}
