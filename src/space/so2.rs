//! The rotation group SO(2): planar orientations.

use core::f64::consts::{PI, TAU};

use rand_core::Rng;

use crate::{Sampler, StateSpace, unit_f64};

/// Wraps an angle in radians into `[-pi, pi)`.
///
/// # Examples
///
/// ```
/// use sampling_planning::wrap_angle;
///
/// assert!((wrap_angle(3.0 * std::f64::consts::PI) + std::f64::consts::PI).abs() < 1e-12);
/// assert_eq!(wrap_angle(0.5), 0.5);
/// ```
#[must_use]
pub fn wrap_angle(angle: f64) -> f64 {
    let wrapped = angle - TAU * libm::floor((angle + PI) / TAU);

    // Rounding can land exactly on +pi; that is the same orientation as -pi.
    if wrapped >= PI {
        wrapped - TAU
    } else {
        wrapped
    }
}

/// The signed shortest rotation from `from` to `to`, in `[-pi, pi)`.
///
/// A half turn is reported as `-pi`, so the result is deterministic for antipodal angles.
#[must_use]
pub fn angle_difference(from: f64, to: f64) -> f64 {
    wrap_angle(to - from)
}

/// The space of planar orientations: angles in radians, identified modulo `2 pi`.
///
/// The distance is the length of the shorter arc, at most `pi`; interpolation turns along that
/// arc. States may be any finite angle; results of [`StateSpace::interpolate`] are wrapped
/// into `[-pi, pi)`, except that `t = 0` and `t = 1` return the endpoints verbatim.
///
/// # Examples
///
/// ```
/// use sampling_planning::{So2, StateSpace};
///
/// let (a, b) = (3.0, -3.0);
/// assert!((So2.distance(&a, &b) - (2.0 * std::f64::consts::PI - 6.0)).abs() < 1e-12);
/// ```
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct So2;

impl StateSpace for So2 {
    type State = f64;

    fn is_symmetric(&self) -> bool {
        true
    }

    fn dimension(&self) -> usize {
        1
    }

    fn distance(&self, from: &f64, to: &f64) -> f64 {
        libm::fabs(angle_difference(*from, *to))
    }

    fn interpolate(&self, from: &f64, to: &f64, t: f64) -> f64 {
        interpolate_angle(*from, *to, t)
    }
}

/// Turns from `from` toward `to` along the shorter arc by the fraction `t`, wrapped into
/// `[-pi, pi)`; `t = 0` and `t = 1` return the endpoints verbatim.
pub(crate) fn interpolate_angle(from: f64, to: f64, t: f64) -> f64 {
    if t <= 0.0 {
        from
    } else if t >= 1.0 {
        to
    } else {
        wrap_angle(from + t * angle_difference(from, to))
    }
}

/// Samples orientations uniformly from `[-pi, pi)`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct UniformAngle;

impl Sampler<f64> for UniformAngle {
    fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> f64 {
        uniform_angle(rng)
    }
}

/// A uniform angle in `[-pi, pi)` from one RNG draw.
pub(crate) fn uniform_angle<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    TAU * unit_f64(rng) - PI
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::SeedableRng;

    #[test]
    fn wrapping_covers_both_directions_and_the_seam() {
        assert!((wrap_angle(PI) + PI).abs() < 1e-15);
        assert!((wrap_angle(-PI) + PI).abs() < 1e-15);
        assert!((wrap_angle(7.0) - (7.0 - TAU)).abs() < 1e-12);
        assert!((wrap_angle(-7.0) - (TAU - 7.0)).abs() < 1e-12);
        assert!((angle_difference(3.0, -3.0) - (TAU - 6.0)).abs() < 1e-12);
        assert!((angle_difference(-3.0, 3.0) + (TAU - 6.0)).abs() < 1e-12);
    }

    #[test]
    fn interpolation_takes_the_short_way_across_the_seam() {
        let (a, b) = (3.0, -3.0);
        let mid = So2.interpolate(&a, &b, 0.5);

        assert!((libm::fabs(mid) - PI).abs() < 1e-12, "{mid}");
        assert_eq!(So2.interpolate(&a, &b, 0.0), a);
        assert_eq!(So2.interpolate(&a, &b, 1.0), b);
        assert!((So2.distance(&a, &mid) - 0.5 * So2.distance(&a, &b)).abs() < 1e-12);
    }

    #[test]
    fn distance_is_a_metric_on_samples() {
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(1);

        for _ in 0..1000 {
            let [a, b, c] = [(); 3].map(|()| UniformAngle.sample(&mut rng));
            assert!((-PI..PI).contains(&a));
            assert!((So2.distance(&a, &b) - So2.distance(&b, &a)).abs() < 1e-12);
            assert!(So2.distance(&a, &b) <= PI);
            assert!(So2.distance(&a, &c) <= So2.distance(&a, &b) + So2.distance(&b, &c) + 1e-12);
        }
    }
}
