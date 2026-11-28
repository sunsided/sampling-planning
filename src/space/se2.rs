//! The planar rigid-body space SE(2): position and heading.

use rand_core::Rng;

use super::so2::{angle_difference, interpolate_angle, uniform_angle};
use crate::{Sampler, StateSpace, UniformBox};

/// A planar pose: position `(x, y)` and heading `theta` in radians.
///
/// Shared by [`Se2`], [`Dubins`](crate::Dubins) and [`ReedsShepp`](crate::ReedsShepp). It is a
/// struct rather than `[f64; 3]` so that samplers and nearest-neighbor structures for
/// Euclidean coordinate vectors cannot be applied to it by accident.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Se2State {
    /// The x coordinate.
    pub x: f64,
    /// The y coordinate.
    pub y: f64,
    /// The heading in radians, counterclockwise from the x axis. Any finite angle; spaces
    /// return headings wrapped into `[-pi, pi)`.
    pub theta: f64,
}

impl Se2State {
    /// The pose at `(x, y)` with heading `theta`.
    #[must_use]
    pub const fn new(x: f64, y: f64, theta: f64) -> Self {
        Self { x, y, theta }
    }
}

/// SE(2) with the metric `|p_1 - p_2| + rotation_weight * |angle(theta_1, theta_2)|`.
///
/// A weighted sum of the Euclidean position metric and the SO(2) arc length, so it is a metric
/// (OMPL's `SE2StateSpace` weighs the rotation by 0.5 by default; here the weight is explicit).
/// Interpolation moves the position on a straight line and turns along the shorter arc at the
/// same time, the geodesic of this metric. `t = 0` and `t = 1` return the endpoints verbatim.
///
/// This is the space of a holonomic planar robot. Car-like robots that cannot move sideways use
/// [`Dubins`](crate::Dubins) or [`ReedsShepp`](crate::ReedsShepp) instead.
///
/// # Examples
///
/// ```
/// use sampling_planning::{Se2, Se2State, StateSpace};
///
/// let space = Se2::new(0.5);
/// let (a, b) = (Se2State::new(0.0, 0.0, 0.0), Se2State::new(3.0, 4.0, 1.0));
///
/// assert!((space.distance(&a, &b) - 5.5).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Se2 {
    /// The weight of the heading term; positive and finite. Units of distance per radian.
    rotation_weight: f64,
}

impl Se2 {
    /// The space with the given rotation weight.
    ///
    /// # Panics
    ///
    /// Panics if `rotation_weight` is not positive and finite.
    #[must_use]
    pub fn new(rotation_weight: f64) -> Self {
        assert!(
            rotation_weight.is_finite() && rotation_weight > 0.0,
            "rotation weight must be positive and finite"
        );

        Self { rotation_weight }
    }

    /// The weight of the heading term.
    #[must_use]
    pub const fn rotation_weight(&self) -> f64 {
        self.rotation_weight
    }
}

impl StateSpace for Se2 {
    type State = Se2State;

    fn is_symmetric(&self) -> bool {
        true
    }

    fn dimension(&self) -> usize {
        3
    }

    fn distance(&self, from: &Se2State, to: &Se2State) -> f64 {
        let position = libm::hypot(to.x - from.x, to.y - from.y);
        let rotation = libm::fabs(angle_difference(from.theta, to.theta));

        position + self.rotation_weight * rotation
    }

    fn interpolate(&self, from: &Se2State, to: &Se2State, t: f64) -> Se2State {
        if t <= 0.0 {
            return *from;
        }

        if t >= 1.0 {
            return *to;
        }

        Se2State {
            x: from.x + (to.x - from.x) * t,
            y: from.y + (to.y - from.y) * t,
            theta: interpolate_angle(from.theta, to.theta, t),
        }
    }
}

/// Samples poses uniformly: positions from an axis-aligned box, headings from `[-pi, pi)`.
///
/// Works for [`Se2`], [`Dubins`](crate::Dubins) and [`ReedsShepp`](crate::ReedsShepp). It does
/// not sample informed sets directly; informed planners fall back to rejection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UniformSe2 {
    /// The position box.
    position: UniformBox<2>,
}

impl UniformSe2 {
    /// A sampler over the position box spanned by `lower` and `upper` and all headings.
    ///
    /// # Panics
    ///
    /// Panics if any bound is not finite or `lower > upper` on any axis.
    #[must_use]
    pub fn new(lower: [f64; 2], upper: [f64; 2]) -> Self {
        Self {
            position: UniformBox::new(lower, upper),
        }
    }

    /// Whether the position of `state` lies in the closed position box.
    #[must_use]
    pub fn contains(&self, state: &Se2State) -> bool {
        self.position.contains(&[state.x, state.y])
    }
}

impl Sampler<Se2State> for UniformSe2 {
    fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Se2State {
        let [x, y] = self.position.sample(rng);

        Se2State {
            x,
            y,
            theta: uniform_angle(rng),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::PI;
    use rand_core::SeedableRng;

    #[test]
    fn interpolation_is_linear_in_position_and_short_in_heading() {
        let space = Se2::new(1.0);
        let a = Se2State::new(0.0, 0.0, 3.0);
        let b = Se2State::new(2.0, -2.0, -3.0);
        let mid = space.interpolate(&a, &b, 0.5);

        assert_eq!((mid.x, mid.y), (1.0, -1.0));
        assert!((libm::fabs(mid.theta) - PI).abs() < 1e-12);
        assert!((space.distance(&a, &mid) - 0.5 * space.distance(&a, &b)).abs() < 1e-12);
        assert_eq!(space.interpolate(&a, &b, 0.0), a);
        assert_eq!(space.interpolate(&a, &b, 1.0), b);
    }

    #[test]
    fn distance_is_a_metric_on_samples() {
        let space = Se2::new(0.7);
        let mut sampler = UniformSe2::new([-5.0, -5.0], [5.0, 5.0]);
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(2);

        for _ in 0..1000 {
            let [a, b, c] = [(); 3].map(|()| sampler.sample(&mut rng));
            assert!(sampler.contains(&a) && (-PI..PI).contains(&a.theta));
            assert!((space.distance(&a, &b) - space.distance(&b, &a)).abs() < 1e-12);
            assert!(
                space.distance(&a, &c) <= space.distance(&a, &b) + space.distance(&b, &c) + 1e-9
            );
        }

        let a = Se2State::new(1.0, 1.0, 1.0);
        assert!(space.distance(&a, &a).abs() < f64::EPSILON);
    }

    #[test]
    #[should_panic(expected = "rotation weight")]
    fn rejects_non_positive_weights() {
        let _ = Se2::new(0.0);
    }
}
