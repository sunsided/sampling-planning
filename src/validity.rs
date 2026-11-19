//! Validity checking of states and motions with a three-valued result.

use crate::StateSpace;

/// The outcome of a validity check.
///
/// Planners accept only [`Validity::Valid`]. [`Validity::Unknown`] is rejected like
/// [`Validity::Invalid`]: a check that fails to prove validity never yields a valid state or
/// edge. This is the conservative default that reachable-set based edge checks rely on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[must_use]
pub enum Validity {
    /// Proven valid: collision-free and within all constraints.
    Valid,
    /// Proven invalid.
    Invalid,
    /// The checker could not decide, for example because an over-approximation touches an
    /// obstacle. Treated as invalid by planners.
    Unknown,
}

impl Validity {
    /// Whether a planner may accept this result. Only [`Validity::Valid`] is accepted.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        matches!(self, Self::Valid)
    }

    /// Combines two results for a compound check: [`Validity::Invalid`] dominates
    /// [`Validity::Unknown`], which dominates [`Validity::Valid`].
    pub const fn and(self, other: Self) -> Self {
        match (self, other) {
            (Self::Invalid, _) | (_, Self::Invalid) => Self::Invalid,
            (Self::Unknown, _) | (_, Self::Unknown) => Self::Unknown,
            (Self::Valid, Self::Valid) => Self::Valid,
        }
    }
}

/// Checks states and motions of a [`StateSpace`].
pub trait ValidityChecker<Sp: StateSpace> {
    /// Checks a single state.
    fn check_state(&self, state: &Sp::State) -> Validity;

    /// Checks the motion from `from` to `to`, including `to`.
    ///
    /// `from` is assumed to be valid already; planners only call this from tree or roadmap
    /// nodes. Use [`discretized_motion`] when only a state check is available.
    fn check_motion(&self, space: &Sp, from: &Sp::State, to: &Sp::State) -> Validity;
}

/// Checks a motion by checking interpolated states at most `resolution` apart.
///
/// Checks the states after `from` up to and including `to`. Stops at the first
/// [`Validity::Invalid`]; otherwise returns [`Validity::Unknown`] if any state was unknown.
/// This is a sampled check, not a proof: obstacles thinner than `resolution` can be missed.
///
/// # Panics
///
/// Panics if `resolution` is not positive and finite.
pub fn discretized_motion<Sp, F>(
    space: &Sp,
    from: &Sp::State,
    to: &Sp::State,
    resolution: f64,
    mut check_state: F,
) -> Validity
where
    Sp: StateSpace,
    F: FnMut(&Sp::State) -> Validity,
{
    assert!(
        resolution.is_finite() && resolution > 0.0,
        "resolution must be positive and finite"
    );

    let steps = step_count(space.distance(from, to), resolution);
    let mut result = Validity::Valid;

    for step in 1..=steps {
        let t = f64::from(step) / f64::from(steps);
        result = result.and(check_state(&space.interpolate(from, to, t)));

        if result == Validity::Invalid {
            break;
        }
    }

    result
}

/// The number of interpolation steps so that consecutive states are at most `resolution`
/// apart. At least one, so the end state is always checked.
fn step_count(distance: f64, resolution: f64) -> u32 {
    let steps = libm::ceil(distance / resolution);

    if steps.is_nan() || steps < 1.0 {
        1
    } else if steps >= f64::from(u32::MAX) {
        u32::MAX
    } else {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "steps is an integer in [1, u32::MAX) after the guards above"
        )]
        let steps = steps as u32;
        steps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Euclidean;

    #[test]
    fn invalid_dominates_unknown_dominates_valid() {
        use Validity::{Invalid, Unknown, Valid};

        assert_eq!(Valid.and(Valid), Valid);
        assert_eq!(Valid.and(Unknown), Unknown);
        assert_eq!(Unknown.and(Invalid), Invalid);
        assert_eq!(Invalid.and(Valid), Invalid);
        assert!(!Unknown.is_valid());
    }

    #[test]
    fn discretized_motion_finds_wall_and_unknown() {
        let wall = |s: &[f64; 1]| {
            if (4.9..5.1).contains(&s[0]) {
                Validity::Invalid
            } else if s[0] > 8.0 {
                Validity::Unknown
            } else {
                Validity::Valid
            }
        };

        let space = Euclidean::<1>;
        assert_eq!(
            discretized_motion(&space, &[0.0], &[4.0], 0.1, wall),
            Validity::Valid
        );
        assert_eq!(
            discretized_motion(&space, &[0.0], &[6.0], 0.1, wall),
            Validity::Invalid
        );
        assert_eq!(
            discretized_motion(&space, &[6.0], &[9.0], 0.1, wall),
            Validity::Unknown
        );
        assert_eq!(
            discretized_motion(&space, &[1.0], &[1.0], 0.1, wall),
            Validity::Valid
        );
    }
}
