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
    /// earlier, sparser check already tested (OMPL's `isValidAtResolution`).
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

        for (position, index) in bisection_order(count).enumerate() {
            if position < done {
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
    use alloc::vec::Vec;

    #[test]
    fn bisection_order_starts_at_the_midpoint_and_covers_every_index() {
        let order: Vec<usize> = bisection_order(7).collect();
        assert_eq!(order, [4, 2, 6, 1, 3, 5, 7]);
        assert_eq!(bisection_order(0).count(), 0);
        assert_eq!(bisection_order(1).collect::<Vec<_>>(), [1]);
    }
}
