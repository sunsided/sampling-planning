//! Steering: how a planner extends a node toward a sample.

use alloc::vec::Vec;

use crate::StateSpace;

/// Extends a state toward a target state.
pub trait Steering<Sp: StateSpace> {
    /// The state reached when moving from `from` toward `toward`.
    ///
    /// May stop short of `toward`; the result must be connectable from `from` by
    /// [`StateSpace::interpolate`], since that segment is what gets validity-checked.
    fn steer(&self, space: &Sp, from: &Sp::State, toward: &Sp::State) -> Sp::State;

    /// Every state [`Steering::steer`] can return from `from`, when there are finitely many
    /// (a fixed set of motion primitives, lattice moves, the edges of a road graph); `None`
    /// when steering is continuous.
    ///
    /// Tree planners with [vertex retirement](crate::trees#vertex-retirement) use it to retire
    /// a vertex as soon as all its successors are in the tree, when steering from it can
    /// produce no new state. Successors known to be invalid (outside a bounded lattice) may be
    /// left out; any other unreachable one keeps the vertex from ever counting as exhausted.
    /// The default is `None`.
    fn successors(&self, space: &Sp, from: &Sp::State) -> Option<Vec<Sp::State>> {
        let _ = (space, from);

        None
    }
}

/// Moves along the geodesic toward the target, at most `max_step` far.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StraightLine {
    /// The longest step, in units of [`StateSpace::distance`]. Positive and finite.
    max_step: f64,
}

impl StraightLine {
    /// Creates a straight-line steering function with the given maximum step length.
    ///
    /// # Panics
    ///
    /// Panics if `max_step` is not positive and finite.
    #[must_use]
    pub fn new(max_step: f64) -> Self {
        assert!(
            max_step.is_finite() && max_step > 0.0,
            "max_step must be positive and finite"
        );

        Self { max_step }
    }

    /// The longest step, in units of [`StateSpace::distance`].
    #[must_use]
    pub const fn max_step(&self) -> f64 {
        self.max_step
    }
}

impl<Sp: StateSpace> Steering<Sp> for StraightLine {
    fn steer(&self, space: &Sp, from: &Sp::State, toward: &Sp::State) -> Sp::State {
        let distance = space.distance(from, toward);

        if distance <= self.max_step {
            toward.clone()
        } else {
            space.interpolate(from, toward, self.max_step / distance)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Euclidean;

    #[test]
    fn reaches_near_targets_and_clamps_far_ones() {
        let steering = StraightLine::new(1.0);
        let space = Euclidean::<2>;

        assert_eq!(steering.steer(&space, &[0.0, 0.0], &[0.6, 0.8]), [0.6, 0.8]);

        let far = steering.steer(&space, &[0.0, 0.0], &[30.0, 40.0]);
        assert!((space.distance(&[0.0, 0.0], &far) - 1.0).abs() < 1e-12);
        assert!((far[0] - 0.6).abs() < 1e-12 && (far[1] - 0.8).abs() < 1e-12);
    }
}
