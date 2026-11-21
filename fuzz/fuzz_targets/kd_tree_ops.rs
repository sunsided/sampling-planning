//! Fuzzes [`KdTree`] with arbitrary insert, remove and query sequences against
//! [`LinearScan`], the brute-force reference it must match exactly.
//!
//! Coordinates sit on a coarse grid, so exact ties and duplicate states are common: they are
//! where the earliest-inserted tie-breaking rule is easiest to get wrong.

#![no_main]
#![forbid(unsafe_code)]

use libfuzzer_sys::fuzz_target;
use sampling_planning::{Euclidean, KdTree, LinearScan, NearestNeighbors, RemovableNeighbors};

/// The space both structures are queried in.
const SPACE: Euclidean<2> = Euclidean::<2>;

/// The kd-tree under test and the linear scan it is checked against, with the same contents.
#[derive(Default)]
struct Twins {
    /// The structure under test.
    kd: KdTree<2>,
    /// The reference.
    linear: LinearScan<[f64; 2]>,
    /// Ids currently stored, in no particular order.
    stored: Vec<usize>,
    /// The next fresh id.
    next_id: usize,
}

impl Twins {
    /// Applies one operation decoded from `(op, a, b)`.
    fn apply(&mut self, op: u8, a: i8, b: i8) {
        let point = grid_point(a, b);

        match op % 4 {
            0 | 1 => self.insert(point),
            2 => self.remove(a, b),
            _ => self.assert_queries_agree(&point, usize::from(op / 4) % 12),
        }
    }

    /// Inserts `point` under a fresh id into both structures.
    fn insert(&mut self, point: [f64; 2]) {
        self.kd.insert(self.next_id, point);
        NearestNeighbors::<Euclidean<2>>::insert(&mut self.linear, self.next_id, point);
        self.stored.push(self.next_id);
        self.next_id += 1;
    }

    /// Removes a stored id picked by `a` and `b` from both structures, if any is stored.
    fn remove(&mut self, a: i8, b: i8) {
        if self.stored.is_empty() {
            return;
        }

        let pick = usize::from(a.unsigned_abs()) * 256 + usize::from(b.unsigned_abs());
        let id = self.stored.swap_remove(pick % self.stored.len());

        assert!(self.kd.remove(id), "kd-tree lost id {id}");
        assert!(RemovableNeighbors::<Euclidean<2>>::remove(
            &mut self.linear,
            id
        ));
    }

    /// Asserts that every query type gives identical answers at `query`.
    fn assert_queries_agree(&self, query: &[f64; 2], k: usize) {
        let (mut kd_ids, mut linear_ids) = (Vec::new(), Vec::new());

        assert_eq!(
            self.kd.nearest(&SPACE, query),
            self.linear.nearest(&SPACE, query)
        );

        self.kd.within_radius(&SPACE, query, 1.5, &mut kd_ids);
        self.linear
            .within_radius(&SPACE, query, 1.5, &mut linear_ids);
        assert_eq!(kd_ids, linear_ids, "within_radius at {query:?}");

        self.kd.k_nearest(&SPACE, query, k, &mut kd_ids);
        self.linear.k_nearest(&SPACE, query, k, &mut linear_ids);
        assert_eq!(kd_ids, linear_ids, "k_nearest k = {k} at {query:?}");

        assert_eq!(
            NearestNeighbors::<Euclidean<2>>::len(&self.kd),
            self.stored.len()
        );
    }
}

/// A point on the half-unit grid in `[-8, 8]^2`.
fn grid_point(a: i8, b: i8) -> [f64; 2] {
    [f64::from(a / 8) / 2.0, f64::from(b / 8) / 2.0]
}

fuzz_target!(|ops: Vec<(u8, i8, i8)>| {
    let mut twins = Twins::default();

    for (op, a, b) in ops {
        twins.apply(op, a, b);
    }

    twins.assert_queries_agree(&[0.25, -0.25], 5);
});
