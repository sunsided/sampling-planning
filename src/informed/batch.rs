//! Batch building blocks shared by BIT* and AIT*: the neighbor rule of a batch and informed
//! batch sampling.

use alloc::vec::Vec;

use rand_core::Rng;

use super::BitStarConfig;
use crate::radius::{dimension_f64, k_nearest_count, shrinking_radius};
use crate::trees::INFORMED_ATTEMPTS;
use crate::{GoalRegion, NearestNeighbors, Sampler, StateSpace, ValidityChecker};

/// The neighbor rule of one batch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum Neighborhood {
    /// Every state within this distance.
    Radius(f64),
    /// The `k` nearest states, each at most `max_radius` away.
    Nearest(usize),
}

impl Neighborhood {
    /// The rule for a batch whose graph holds `q` states before the newest samples (eqs. 1
    /// and 2 of Gammell et al. 2020): `min(gamma (ln q / q)^(1/d), max_radius)`, or the
    /// `ceil(k_constant ln q)` nearest.
    pub(super) fn for_batch(config: &BitStarConfig, q: usize, dimension: usize) -> Self {
        match config.k_nearest {
            Some(k_constant) => Self::Nearest(k_nearest_count(k_constant, q)),
            None => Self::Radius(
                shrinking_radius(config.gamma, q, dimension_f64(dimension)).min(config.max_radius),
            ),
        }
    }

    /// Replaces `out` with the ids in `nn` near `state` under this rule, excluding `id` (the
    /// node that holds `state`). k-nearest results whose `distance` (by id) exceeds
    /// `max_radius` are dropped.
    pub(super) fn query<Sp: StateSpace, N: NearestNeighbors<Sp>>(
        self,
        (space, nn): (&Sp, &N),
        (id, state): (usize, &Sp::State),
        (max_radius, distance): (f64, impl Fn(usize) -> f64),
        out: &mut Vec<usize>,
    ) {
        match self {
            Self::Radius(radius) => nn.within_radius(space, state, radius, out),
            Self::Nearest(k) => {
                nn.k_nearest(space, state, k + 1, out);

                if max_radius.is_finite() {
                    out.retain(|&x| distance(x) <= max_radius);
                }
            }
        }

        out.retain(|&x| x != id);
    }
}

/// What informed batch sampling needs to know about the problem.
pub(super) struct Informed<'a, Sp: StateSpace, C> {
    /// The configuration space.
    pub(super) space: &'a Sp,
    /// Decides which states are allowed.
    pub(super) checker: &'a C,
    /// The start state, one focus of the informed set.
    pub(super) start: &'a Sp::State,
    /// The goal region; its center is the other focus.
    pub(super) goal: &'a GoalRegion<Sp::State>,
    /// The cost of the current solution; infinite until one is found.
    pub(super) best_cost: f64,
}

impl<Sp: StateSpace, C: ValidityChecker<Sp>> Informed<'_, Sp, C> {
    /// Draws a valid state with `f_hat < c_best`, directly from the informed set (bound
    /// `c_best + tolerance` around start and goal center) when the sampler supports it, in at
    /// most [`INFORMED_ATTEMPTS`] draws.
    pub(super) fn draw<Sa, R>(&self, sampler: &mut Sa, rng: &mut R) -> Option<Sp::State>
    where
        Sa: Sampler<Sp::State>,
        R: Rng + ?Sized,
    {
        let bound = self.best_cost + self.goal.tolerance();

        for _ in 0..INFORMED_ATTEMPTS {
            let direct = if bound.is_finite() {
                let center = self.goal.center();
                sampler.sample_informed(self.space, rng, self.start, center, bound)
            } else {
                None
            };
            let state = direct.unwrap_or_else(|| sampler.sample(rng));

            if self.is_useful(&state) {
                return Some(state);
            }
        }

        None
    }

    /// Whether `state` is valid and could improve the current solution.
    fn is_useful(&self, state: &Sp::State) -> bool {
        let g_hat = self.space.distance(self.start, state);
        let h_hat =
            (self.space.distance(state, self.goal.center()) - self.goal.tolerance()).max(0.0);

        g_hat + h_hat < self.best_cost && self.checker.check_state(state).is_valid()
    }
}
