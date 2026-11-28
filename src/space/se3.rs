//! The spatial rigid-body space SE(3): position and orientation in 3D.

use core::f64::consts::TAU;

use rand_core::Rng;

use crate::{Sampler, StateSpace, UniformBox, unit_f64};

/// A spatial pose: a position and an orientation as a unit quaternion.
///
/// The quaternion is stored as `[w, x, y, z]` (scalar first). `q` and `-q` describe the same
/// rotation; [`Se3`] treats them as equal. It is a struct rather than `[f64; 7]` so that
/// samplers and nearest-neighbor structures for Euclidean coordinate vectors cannot be applied
/// to it by accident.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Se3State {
    /// The position `[x, y, z]`.
    pub position: [f64; 3],
    /// The orientation as a unit quaternion `[w, x, y, z]`. [`Se3`] assumes unit length;
    /// [`Se3State::new`] normalizes.
    pub rotation: [f64; 4],
}

impl Se3State {
    /// The pose at `position` with the orientation `rotation` (`[w, x, y, z]`), normalized to
    /// unit length.
    ///
    /// # Panics
    ///
    /// Panics if `rotation` is zero or not finite.
    #[must_use]
    pub fn new(position: [f64; 3], rotation: [f64; 4]) -> Self {
        let length = norm4(&rotation);
        assert!(
            length.is_finite() && length > 0.0,
            "rotation quaternion must be finite and non-zero"
        );

        Self {
            position,
            rotation: rotation.map(|c| c / length),
        }
    }

    /// The pose at `position` without rotation.
    #[must_use]
    pub const fn at(position: [f64; 3]) -> Self {
        Self {
            position,
            rotation: [1.0, 0.0, 0.0, 0.0],
        }
    }

    /// The pose at `position`, rotated by `angle` radians about `axis` (right-handed).
    ///
    /// # Panics
    ///
    /// Panics if `axis` is zero or not finite.
    #[must_use]
    pub fn from_axis_angle(position: [f64; 3], axis: [f64; 3], angle: f64) -> Self {
        let length = libm::sqrt(axis.iter().map(|a| a * a).sum());
        assert!(
            length.is_finite() && length > 0.0,
            "rotation axis must be finite and non-zero"
        );

        let (sin, cos) = (libm::sin(0.5 * angle), libm::cos(0.5 * angle));
        let [x, y, z] = axis.map(|a| sin * a / length);

        Self::new(position, [cos, x, y, z])
    }
}

/// SE(3) with the metric `|p_1 - p_2| + rotation_weight * angle(q_1, q_2)`.
///
/// `angle` is the rotation angle in `[0, pi]` of the relative rotation, the geodesic distance
/// on SO(3) under its bi-invariant metric; both quaternion signs give the same angle. A
/// weighted sum of metrics is a metric. OMPL's `SO3StateSpace` reports half this angle (the
/// arc on the unit quaternion sphere) and its `SE3StateSpace` weighs it by 1; here the angle is
/// the full rotation, like [`Se2`](crate::Se2)'s heading term, and the weight is explicit.
///
/// Interpolation moves the position on a straight line and the orientation along the shorter
/// slerp arc at the same time, the geodesic of this metric, so `d(from, interpolate(t)) =
/// t d(from, to)`. `t = 0` and `t = 1` return the endpoints verbatim.
///
/// # Examples
///
/// ```
/// use sampling_planning::{Se3, Se3State, StateSpace};
///
/// let space = Se3::new(0.5);
/// let a = Se3State::at([0.0, 0.0, 0.0]);
/// let b = Se3State::from_axis_angle([1.0, 2.0, 2.0], [0.0, 0.0, 1.0], 1.0);
///
/// assert!((space.distance(&a, &b) - 3.5).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Se3 {
    /// The weight of the rotation term; positive and finite. Units of distance per radian.
    rotation_weight: f64,
}

impl Se3 {
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

    /// The weight of the rotation term.
    #[must_use]
    pub const fn rotation_weight(&self) -> f64 {
        self.rotation_weight
    }
}

impl StateSpace for Se3 {
    type State = Se3State;

    fn dimension(&self) -> usize {
        6
    }

    fn distance(&self, from: &Se3State, to: &Se3State) -> f64 {
        let squared: f64 = (0..3)
            .map(|i| (to.position[i] - from.position[i]) * (to.position[i] - from.position[i]))
            .sum();

        libm::sqrt(squared) + self.rotation_weight * rotation_angle(&from.rotation, &to.rotation)
    }

    fn interpolate(&self, from: &Se3State, to: &Se3State, t: f64) -> Se3State {
        if t <= 0.0 {
            return *from;
        }

        if t >= 1.0 {
            return *to;
        }

        Se3State {
            position: core::array::from_fn(|i| {
                from.position[i] + (to.position[i] - from.position[i]) * t
            }),
            rotation: slerp(&from.rotation, &to.rotation, t),
        }
    }
}

/// Samples poses uniformly: positions from an axis-aligned box, orientations from the Haar
/// (uniform) measure on SO(3).
///
/// It does not sample informed sets directly; informed planners fall back to rejection.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UniformSe3 {
    /// The position box.
    position: UniformBox<3>,
}

impl UniformSe3 {
    /// A sampler over the position box spanned by `lower` and `upper` and all orientations.
    ///
    /// # Panics
    ///
    /// Panics if any bound is not finite or `lower > upper` on any axis.
    #[must_use]
    pub fn new(lower: [f64; 3], upper: [f64; 3]) -> Self {
        Self {
            position: UniformBox::new(lower, upper),
        }
    }

    /// Whether the position of `state` lies in the closed position box.
    #[must_use]
    pub fn contains(&self, state: &Se3State) -> bool {
        self.position.contains(&state.position)
    }
}

impl Sampler<Se3State> for UniformSe3 {
    fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Se3State {
        let position = self.position.sample(rng);

        Se3State {
            position,
            rotation: uniform_rotation(rng),
        }
    }
}

/// A uniformly distributed unit quaternion from three RNG draws (Shoemake 1992, "Uniform random
/// rotations"); uniform on the 3-sphere, hence Haar-distributed as a rotation.
fn uniform_rotation<R: Rng + ?Sized>(rng: &mut R) -> [f64; 4] {
    let u = unit_f64(rng);
    let (a, b) = (TAU * unit_f64(rng), TAU * unit_f64(rng));
    let (low, high) = (libm::sqrt(1.0 - u), libm::sqrt(u));

    [
        high * libm::cos(b),
        low * libm::sin(a),
        low * libm::cos(a),
        high * libm::sin(b),
    ]
}

/// The rotation angle in `[0, pi]` between the unit quaternions `a` and `b`.
///
/// The angle between the 4-vectors `a` and `±b` (the sign that makes them closer) is half the
/// rotation angle; `2 atan2(|a - b|, |a + b|)` computes it accurately near zero and near a
/// half turn, where `acos` of the dot product loses precision.
fn rotation_angle(a: &[f64; 4], b: &[f64; 4]) -> f64 {
    let b = aligned(a, b);
    let difference = norm4(&core::array::from_fn(|i| a[i] - b[i]));
    let sum = norm4(&core::array::from_fn(|i| a[i] + b[i]));

    4.0 * libm::atan2(difference, sum)
}

/// Spherical linear interpolation from `a` toward `b` along the shorter arc, normalized.
fn slerp(a: &[f64; 4], b: &[f64; 4], t: f64) -> [f64; 4] {
    let b = aligned(a, b);
    let difference = norm4(&core::array::from_fn(|i| a[i] - b[i]));
    let sum = norm4(&core::array::from_fn(|i| a[i] + b[i]));
    let omega = 2.0 * libm::atan2(difference, sum);
    let sin_omega = libm::sin(omega);

    // Below this the arc is numerically straight; a normalized lerp is exact enough.
    let (wa, wb) = if sin_omega < 1e-9 {
        (1.0 - t, t)
    } else {
        (
            libm::sin((1.0 - t) * omega) / sin_omega,
            libm::sin(t * omega) / sin_omega,
        )
    };
    let q: [f64; 4] = core::array::from_fn(|i| wa * a[i] + wb * b[i]);
    let length = norm4(&q);

    q.map(|c| c / length)
}

/// `b` or `-b`, whichever has a non-negative dot product with `a`: the same rotation, on the
/// near side of the quaternion double cover.
fn aligned(a: &[f64; 4], b: &[f64; 4]) -> [f64; 4] {
    let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();

    if dot < 0.0 { b.map(|c| -c) } else { *b }
}

/// The Euclidean norm of a quaternion.
fn norm4(q: &[f64; 4]) -> f64 {
    libm::sqrt(q.iter().map(|c| c * c).sum())
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::f64::consts::{FRAC_2_PI, FRAC_PI_2, PI};
    use rand_core::SeedableRng;

    #[test]
    fn rotation_angle_ignores_the_quaternion_sign() {
        let a = Se3State::from_axis_angle([0.0; 3], [1.0, 1.0, 0.0], 0.3);
        let b = Se3State::from_axis_angle([0.0; 3], [1.0, 1.0, 0.0], 2.8);
        let flipped = b.rotation.map(|c| -c);

        assert!((rotation_angle(&a.rotation, &b.rotation) - 2.5).abs() < 1e-12);
        assert!((rotation_angle(&a.rotation, &flipped) - 2.5).abs() < 1e-12);
        assert!(rotation_angle(&a.rotation, &a.rotation).abs() < 1e-15);
    }

    #[test]
    fn interpolation_follows_the_geodesic_and_keeps_endpoints() {
        let space = Se3::new(0.8);
        let a = Se3State::from_axis_angle([1.0, 0.0, -1.0], [0.0, 1.0, 0.0], 0.4);
        let b = Se3State::from_axis_angle([3.0, 2.0, 1.0], [1.0, 0.0, 1.0], -2.9);
        let total = space.distance(&a, &b);

        for t in [0.1, 0.25, 0.5, 0.9] {
            let x = space.interpolate(&a, &b, t);
            assert!((norm4(&x.rotation) - 1.0).abs() < 1e-12);
            assert!((space.distance(&a, &x) - t * total).abs() < 1e-9, "t = {t}");
            assert!((space.distance(&x, &b) - (1.0 - t) * total).abs() < 1e-9);
        }

        assert_eq!(space.interpolate(&a, &b, 0.0), a);
        assert_eq!(space.interpolate(&a, &b, 1.0), b);
    }

    #[test]
    fn distance_is_a_metric_on_samples() {
        let space = Se3::new(0.5);
        let mut sampler = UniformSe3::new([-5.0; 3], [5.0; 3]);
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(3);

        for _ in 0..1000 {
            let [a, b, c] = [(); 3].map(|()| sampler.sample(&mut rng));
            assert!(sampler.contains(&a) && (norm4(&a.rotation) - 1.0).abs() < 1e-12);
            assert!((space.distance(&a, &b) - space.distance(&b, &a)).abs() < 1e-12);
            assert!(
                space.distance(&a, &c) <= space.distance(&a, &b) + space.distance(&b, &c) + 1e-9
            );
        }
    }

    #[test]
    fn sampled_rotations_follow_the_haar_angle_distribution() {
        // Under the Haar measure the rotation angle has density (1 - cos θ) / π on [0, π],
        // whose mean is π / 2 + 2 / π and for which P(θ <= π / 2) = (π / 2 - 1) / π.
        let mut sampler = UniformSe3::new([0.0; 3], [1.0; 3]);
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(5);
        let identity = [1.0, 0.0, 0.0, 0.0];
        let draws = 40_000;
        let (mut mean, mut small) = (0.0, 0_u32);

        for _ in 0..draws {
            let angle = rotation_angle(&identity, &sampler.sample(&mut rng).rotation);
            mean += angle / f64::from(draws);
            small += u32::from(angle <= FRAC_PI_2);
        }

        assert!((mean - (FRAC_PI_2 + FRAC_2_PI)).abs() < 0.02, "mean {mean}");
        let share = f64::from(small) / f64::from(draws);
        assert!(
            (share - (FRAC_PI_2 - 1.0) / PI).abs() < 0.01,
            "share {share}"
        );
    }

    #[test]
    #[should_panic(expected = "rotation quaternion")]
    fn rejects_zero_quaternions() {
        let _ = Se3State::new([0.0; 3], [0.0; 4]);
    }

    #[test]
    #[should_panic(expected = "rotation weight")]
    fn rejects_non_positive_weights() {
        let _ = Se3::new(-1.0);
    }
}
