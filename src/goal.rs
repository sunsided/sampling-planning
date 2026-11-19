//! Goal regions.

use crate::StateSpace;

/// Every state within `tolerance` of `center` is a goal.
///
/// Goal bias samples `center`.
#[derive(Debug, Clone, PartialEq)]
pub struct GoalRegion<S> {
    /// The goal state; also the state drawn for goal-biased samples.
    center: S,
    /// The largest accepted distance from `center`. Non-negative and finite.
    tolerance: f64,
}

impl<S> GoalRegion<S> {
    /// Creates a goal region around `center`.
    ///
    /// # Panics
    ///
    /// Panics if `tolerance` is negative or not finite.
    #[must_use]
    pub fn new(center: S, tolerance: f64) -> Self {
        assert!(
            tolerance.is_finite() && tolerance >= 0.0,
            "goal tolerance must be non-negative and finite"
        );

        Self { center, tolerance }
    }

    /// The goal state.
    #[must_use]
    pub const fn center(&self) -> &S {
        &self.center
    }

    /// The largest accepted distance from [`GoalRegion::center`].
    #[must_use]
    pub const fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Whether `state` lies in the region.
    #[must_use]
    pub fn contains<Sp: StateSpace<State = S>>(&self, space: &Sp, state: &S) -> bool {
        space.distance(state, &self.center) <= self.tolerance
    }
}
