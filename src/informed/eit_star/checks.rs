//! EIT*'s edge checks: sparse state checks for the reverse search, full motion checks for the
//! forward search, and the effort that remains on an edge.

use alloc::collections::VecDeque;

use super::EitStar;
use crate::radius::count_to_f64;
use crate::{NearestNeighbors, Sampler, StateSpace, ValidityChecker};

impl<Sp, Sa, C, N> EitStar<Sp, Sa, C, N>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    N: NearestNeighbors<Sp> + Clone,
{
    /// The number of checks a full-resolution check of the edge between `a` and `b` takes:
    /// `ceil(d / resolution)`, at least one.
    pub(super) fn full_checks(&self, a: usize, b: usize) -> usize {
        let steps = libm::ceil(self.cost_estimate(a, b) / self.config.resolution);

        if steps.is_nan() || steps < 1.0 {
            1
        } else if steps >= count_to_f64(usize::MAX) {
            usize::MAX
        } else {
            #[expect(
                clippy::cast_possible_truncation,
                clippy::cast_sign_loss,
                reason = "steps is a whole number in [1, usize::MAX) after the guards above"
            )]
            let steps = steps as usize;
            steps
        }
    }

    /// The checks still needed to prove the edge from `from` to `to` valid: zero once its
    /// motion checked as valid, otherwise the full count minus the sparse checks done.
    pub(super) fn remaining_checks(&self, from: usize, to: usize) -> usize {
        if self.checks.valid.contains(&(from, to)) {
            return 0;
        }

        let done = self
            .checks
            .performed
            .get(&undirected(from, to))
            .copied()
            .unwrap_or(0);

        self.full_checks(from, to).saturating_sub(done)
    }

    /// Whether the edge from `from` to `to` passes the current sparse check: up to
    /// `sparse_checks` evenly spaced states on it, in bisection order, skipping those an
    /// earlier check already tested (OMPL's `isValidAtResolution`). A denser check reuses a
    /// sparser one only when both point counts are one below a power of two, so that the first
    /// points in bisection order are the earlier ones; any other density change checks every
    /// point again.
    pub(super) fn could_be_valid(&mut self, from: usize, to: usize) -> bool {
        if self.checks.valid.contains(&(from, to)) {
            return true;
        }

        if self.checks.invalid.contains(&(from, to)) {
            return false;
        }

        let full = self.full_checks(from, to);
        let segments = self.sparse_checks.saturating_add(1).min(full);
        let count = segments - 1;
        let key = undirected(from, to);
        let done = self.checks.performed.get(&key).copied().unwrap_or(0);
        // The same density again, or a nested denser one: the first `done` points are checked.
        let nested = (done + 1).is_power_of_two() && (count + 1).is_power_of_two();
        let skip = if count == done || nested { done } else { 0 };

        for (position, index) in bisection_order(count).enumerate() {
            if position < skip {
                continue;
            }

            let t = count_to_f64(index) / count_to_f64(segments);
            let state = self
                .space
                .interpolate(&self.nodes[from].state, &self.nodes[to].state, t);

            if !self.checker.check_state(&state).is_valid() {
                self.checks.invalid.insert((from, to));
                self.checks.invalid.insert((to, from));
                return false;
            }
        }

        self.checks.performed.insert(key, done.max(count));

        true
    }

    /// Whether the motion from `from` to `to` checks as valid; results are remembered per
    /// direction, so each motion is checked at most once.
    pub(super) fn is_valid(&mut self, from: usize, to: usize) -> bool {
        if self.checks.valid.contains(&(from, to)) {
            return true;
        }

        if self.checks.invalid.contains(&(from, to)) {
            return false;
        }

        let valid = self
            .checker
            .check_motion(&self.space, &self.nodes[from].state, &self.nodes[to].state)
            .is_valid();
        let record = if valid {
            &mut self.checks.valid
        } else {
            &mut self.checks.invalid
        };
        record.insert((from, to));

        valid
    }
}

/// The edge key `(low, high)` of the undirected edge between `a` and `b`.
fn undirected(a: usize, b: usize) -> (usize, usize) {
    (a.min(b), a.max(b))
}

/// The indices `1..=count` in bisection order: the midpoint first, then the midpoints of the
/// halves, breadth first. The first `2^j - 1` indices for `count = 2^k - 1` are the indices of
/// `count = 2^j - 1`, scaled, so a denser check reuses a sparser one.
fn bisection_order(count: usize) -> impl Iterator<Item = usize> {
    let mut pending = VecDeque::new();
    if count > 0 {
        pending.push_back((1, count));
    }

    core::iter::from_fn(move || {
        let (low, high) = pending.pop_front()?;
        let mid = usize::midpoint(low, high);

        if low < mid {
            pending.push_back((low, mid - 1));
        }

        if high > mid {
            pending.push_back((mid + 1, high));
        }

        Some(mid)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::informed::{BitStarConfig, EitStar, EitStarConfig};
    use crate::{Euclidean, GoalRegion, UniformBox, Validity, ValidityChecker};
    use alloc::vec::Vec;

    /// A 1D checker whose states within `width` of `at` are invalid. Motions are never
    /// checked here: these tests exercise the sparse state checks only.
    struct Spot {
        /// The obstacle's center.
        at: f64,
        /// Its half width.
        width: f64,
        /// How many states were checked.
        checks: core::cell::Cell<usize>,
    }

    impl Spot {
        /// An obstacle of half width `width` around `at`.
        fn new(at: f64, width: f64) -> Self {
            Self {
                at,
                width,
                checks: core::cell::Cell::new(0),
            }
        }
    }

    impl ValidityChecker<Euclidean<1>> for Spot {
        fn check_state(&self, s: &[f64; 1]) -> Validity {
            self.checks.set(self.checks.get() + 1);

            if (s[0] - self.at).abs() < self.width {
                Validity::Invalid
            } else {
                Validity::Valid
            }
        }

        fn check_motion(&self, _: &Euclidean<1>, _: &[f64; 1], _: &[f64; 1]) -> Validity {
            Validity::Invalid
        }
    }

    /// An EIT* from 0 toward `goal` on a line with the obstacle `spot`.
    fn eit(
        spot: Spot,
        goal: f64,
        initial_sparse_checks: usize,
    ) -> EitStar<Euclidean<1>, UniformBox<1>, Spot> {
        let mut config = EitStarConfig::from_bit_star(BitStarConfig::new(1.0), 0.1);
        config.initial_sparse_checks = initial_sparse_checks;

        EitStar::new(
            Euclidean::<1>,
            UniformBox::new([0.0], [10.0]),
            spot,
            [0.0],
            GoalRegion::new([goal], 0.01),
            config,
        )
        .expect("valid problem")
    }

    #[test]
    fn a_repeated_sparse_check_checks_nothing_again() {
        // Three sparse checks on a 0.25 edge are cut to two points, which do not nest.
        let mut short = eit(Spot::new(5.0, 0.1), 0.25, 3);
        assert!(short.could_be_valid(0, 1));
        let checked = short.checker.checks.get();

        assert!(short.could_be_valid(0, 1));
        assert_eq!(short.checker.checks.get(), checked);
    }

    #[test]
    fn denser_sparse_checks_test_every_point_they_did_not_reuse() {
        // A 0.25 edge at resolution 0.1 has 3 full checks, so three sparse checks are cut to
        // two points, t = 1/3 and 2/3, after one at t = 1/2. The obstacle sits at t = 1/3.
        let mut short = eit(Spot::new(0.25 / 3.0, 0.01), 0.25, 1);
        assert_eq!(short.full_checks(0, 1), 3);
        assert!(short.could_be_valid(0, 1));
        short.sparse_checks = 3;
        assert!(!short.could_be_valid(0, 1), "t = 1/3 is checked");

        // Two checks (t = 1/3, 2/3) do not nest in five (t = k/6): t = 1/2 must be checked.
        let mut long = eit(Spot::new(5.0, 0.1), 10.0, 2);
        assert!(long.could_be_valid(0, 1));
        long.sparse_checks = 5;
        assert!(!long.could_be_valid(0, 1), "t = 1/2 is checked");
    }

    #[test]
    fn bisection_order_starts_at_the_midpoint_and_covers_every_index() {
        let order: Vec<usize> = bisection_order(7).collect();
        assert_eq!(order, [4, 2, 6, 1, 3, 5, 7]);
        assert_eq!(bisection_order(0).count(), 0);
        assert_eq!(bisection_order(1).collect::<Vec<_>>(), [1]);
    }
}
