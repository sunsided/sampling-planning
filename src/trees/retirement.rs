//! Retirement of exhausted tree vertices from the nearest-neighbor index.
//!
//! A tree planner extends the vertex nearest to a random target. A vertex that cannot grow any
//! more (it sits in a dead end, against a wall, or at the edge of a bounded or discretized
//! space) keeps getting picked, and every pick is a wasted iteration. Retirement removes such a
//! vertex from the index. The vertex stays in the tree, so paths through it and its cost are
//! unaffected; it is just never picked again.

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::{NearestNeighbors, StateSpace};

/// Per-vertex failure counters, the index of every tree state and the removal hook for one
/// tree's index `N`.
///
/// Disabled by default. The planners enable it only for indices that implement
/// [`RemovableNeighbors`](crate::RemovableNeighbors), which provides `remove`.
pub(super) struct Retirement<N> {
    /// The rule when enabled.
    rule: Option<Rule<N>>,
    /// Consecutive failed extensions per vertex id since its last success; grows on demand.
    failures: Vec<usize>,
    /// Whether each vertex id was retired; grows on demand.
    retired_flags: Vec<bool>,
    /// The number of vertices removed from the index so far.
    retired: usize,
}

/// An enabled retirement rule.
struct Rule<N> {
    /// The number of consecutive failed extensions after which a vertex is retired.
    limit: NonZeroUsize,
    /// Removes a vertex id from the index; returns whether it was stored.
    remove: fn(&mut N, usize) -> bool,
    /// An index over every tree state, retired ones included, to decide whether a successor
    /// of finite steering is already in the tree.
    reached: N,
}

impl<N> Retirement<N> {
    /// Retirement switched off: recording does nothing.
    pub(super) const fn disabled() -> Self {
        Self {
            rule: None,
            failures: Vec::new(),
            retired_flags: Vec::new(),
            retired: 0,
        }
    }

    /// Retires a vertex after `limit` consecutive failed extensions, or once all its finite
    /// successors are in the tree, removing it with `remove`. `reached` must index every tree
    /// state so far, by tree id; later states are added through [`Retirement::note_added`].
    pub(super) fn enable(
        &mut self,
        limit: NonZeroUsize,
        remove: fn(&mut N, usize) -> bool,
        reached: N,
    ) {
        self.rule = Some(Rule {
            limit,
            remove,
            reached,
        });
    }

    /// Whether retirement is on.
    pub(super) const fn is_enabled(&self) -> bool {
        self.rule.is_some()
    }

    /// The number of vertices retired so far.
    pub(super) const fn retired(&self) -> usize {
        self.retired
    }

    /// Records the tree's new node `id` at `state`.
    pub(super) fn note_added<Sp>(&mut self, id: usize, state: Sp::State)
    where
        Sp: StateSpace,
        N: NearestNeighbors<Sp>,
    {
        if let Some(rule) = &mut self.rule {
            rule.reached.insert(id, state);
        }
    }

    /// Records the outcome of an extension from `vertex`: a success resets its counter, a
    /// failure counts toward its retirement after `limit` failures in a row.
    pub(super) fn record(&mut self, nn: &mut N, vertex: usize, progressed: bool) {
        let Some(rule) = &self.rule else {
            return;
        };

        let limit = rule.limit.get();
        if self.failures.len() <= vertex {
            self.failures.resize(vertex + 1, 0);
        }

        if progressed {
            self.failures[vertex] = 0;
            return;
        }

        self.failures[vertex] = self.failures[vertex].saturating_add(1);
        if self.failures[vertex] >= limit {
            self.retire(nn, vertex);
        }
    }

    /// Retires `vertex` if every state in `successors` is already in the tree (the decidable
    /// rule for finite steering).
    pub(super) fn retire_if_exhausted<Sp>(
        &mut self,
        nn: &mut N,
        space: &Sp,
        vertex: usize,
        successors: &[Sp::State],
    ) where
        Sp: StateSpace,
        N: NearestNeighbors<Sp>,
    {
        let Some(rule) = &self.rule else {
            return;
        };

        let reached = |state: &Sp::State| Self::has_exact(&rule.reached, space, state);

        if successors.iter().all(reached) {
            self.retire(nn, vertex);
        }
    }

    /// Whether `index` stores a state at distance zero from `state`.
    fn has_exact<Sp>(index: &N, space: &Sp, state: &Sp::State) -> bool
    where
        Sp: StateSpace,
        N: NearestNeighbors<Sp>,
    {
        let mut near = Vec::new();
        index.within_radius(space, state, 0.0, &mut near);

        !near.is_empty()
    }

    /// Removes `vertex` from `nn` unless it is the root (vertex `0`, so the index never runs
    /// empty) or already retired.
    fn retire(&mut self, nn: &mut N, vertex: usize) {
        let Some(rule) = &self.rule else {
            return;
        };

        if self.retired_flags.len() <= vertex {
            self.retired_flags.resize(vertex + 1, false);
        }

        if vertex != 0 && !self.retired_flags[vertex] && (rule.remove)(nn, vertex) {
            self.retired_flags[vertex] = true;
            self.retired += 1;
        }
    }
}

impl<N: Clone> Clone for Retirement<N> {
    fn clone(&self) -> Self {
        Self {
            rule: self.rule.as_ref().map(|rule| Rule {
                limit: rule.limit,
                remove: rule.remove,
                reached: rule.reached.clone(),
            }),
            failures: self.failures.clone(),
            retired_flags: self.retired_flags.clone(),
            retired: self.retired,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Euclidean, LinearScan, RemovableNeighbors};

    /// The index type the tests retire from.
    type Index = LinearScan<[f64; 1]>;

    /// An index holding the points `0, 1, 2` with ids `0, 1, 2`.
    fn three() -> Index {
        let mut nn = Index::default();
        for (id, x) in [0.0, 1.0, 2.0].into_iter().enumerate() {
            NearestNeighbors::<Euclidean<1>>::insert(&mut nn, id, [x]);
        }

        nn
    }

    #[test]
    fn retires_after_consecutive_failures_but_never_the_root() {
        let mut nn = three();
        let mut retirement = Retirement::disabled();
        retirement.record(&mut nn, 1, false);
        assert_eq!(retirement.retired(), 0, "disabled records nothing");

        let limit = NonZeroUsize::new(2).expect("non-zero");
        let remove = <Index as RemovableNeighbors<Euclidean<1>>>::remove;
        retirement.enable(limit, remove, nn.clone());

        retirement.record(&mut nn, 1, false);
        retirement.record(&mut nn, 1, true);
        retirement.record(&mut nn, 1, false);
        assert_eq!(retirement.retired(), 0, "a success resets the counter");

        retirement.record(&mut nn, 1, false);
        retirement.record(&mut nn, 1, false);
        retirement.record(&mut nn, 0, false);
        retirement.record(&mut nn, 0, false);
        assert_eq!(retirement.retired(), 1, "retired once, root kept");
        assert_eq!(NearestNeighbors::<Euclidean<1>>::len(&nn), 2);
        assert_eq!(nn.nearest(&Euclidean::<1>, &[1.1]), Some(2));
    }

    #[test]
    fn retires_once_every_finite_successor_is_reached() {
        let mut nn = three();
        let mut retirement = Retirement::disabled();
        let remove = <Index as RemovableNeighbors<Euclidean<1>>>::remove;
        retirement.enable(NonZeroUsize::MAX, remove, nn.clone());

        retirement.retire_if_exhausted(&mut nn, &Euclidean::<1>, 1, &[[0.0], [3.0]]);
        assert_eq!(retirement.retired(), 0, "3 is not in the tree yet");

        // Retired vertices still count as reached.
        retirement.retire_if_exhausted(&mut nn, &Euclidean::<1>, 2, &[[1.0]]);
        retirement.note_added::<Euclidean<1>>(3, [3.0]);
        retirement.retire_if_exhausted(&mut nn, &Euclidean::<1>, 1, &[[0.0], [2.0], [3.0]]);
        assert_eq!(retirement.retired(), 2);
    }
}
