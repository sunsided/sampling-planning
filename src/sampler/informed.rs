//! Direct sampling of the informed set of a Euclidean path-length problem.

use rand_core::Rng;

use super::unit_f64;
use crate::radius::{count_to_f64, unit_ball_volume};

/// The prolate hyperspheroid `{x : |x - a| + |x - b| <= max_cost}` in R^N: every state that can
/// lie on a path from `a` to `b` of Euclidean length at most `max_cost`.
///
/// This is the informed set of Informed RRT* and BIT* (Gammell, Srinivasa and Barfoot 2014,
/// "Informed RRT*: Optimal sampling-based path planning focused via direct sampling of an
/// admissible ellipsoidal heuristic"). Its foci are `a` and `b`, its transverse diameter is
/// `max_cost`, and all conjugate diameters are `sqrt(max_cost^2 - |b - a|^2)`.
///
/// [`ProlateHyperspheroid::sample`] maps a uniform sample of the unit ball through the
/// scaling `diag(r_1, r_2, ..., r_2)` and a rotation that takes the first axis onto `b - a`. A
/// linear map keeps the distribution uniform. The rotation is a Householder reflection rather
/// than the SVD of Gammell et al.: with all conjugate radii equal, only the first axis has to
/// be placed, and any orthogonal map that places it yields the same set.
///
/// # Examples
///
/// ```
/// use rand_core::SeedableRng;
/// use sampling_planning::ProlateHyperspheroid;
///
/// let ellipse = ProlateHyperspheroid::new([0.0, 0.0], [4.0, 0.0], 5.0).expect("non-degenerate");
/// let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(1);
///
/// for _ in 0..100 {
///     assert!(ellipse.contains(&ellipse.sample(&mut rng)));
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProlateHyperspheroid<const N: usize> {
    /// The midpoint of the foci.
    center: [f64; N],
    /// The first focus.
    focus_a: [f64; N],
    /// The second focus.
    focus_b: [f64; N],
    /// The Householder vector `e_1 - (b - a) / |b - a|`, or `None` when no reflection is needed
    /// (coincident foci, or `b - a` already along the first axis).
    reflection: Option<[f64; N]>,
    /// The transverse radius `max_cost / 2`.
    major_radius: f64,
    /// The conjugate radius `sqrt(max_cost^2 - |b - a|^2) / 2`; positive.
    minor_radius: f64,
}

impl<const N: usize> ProlateHyperspheroid<N> {
    /// The hyperspheroid with foci `a` and `b` and transverse diameter `max_cost`.
    ///
    /// Returns `None` when the set has no interior: `N == 0`, `max_cost` not finite, or
    /// `max_cost <= |b - a|` (no path is that short, or only the straight segment is).
    #[must_use]
    pub fn new(a: [f64; N], b: [f64; N], max_cost: f64) -> Option<Self> {
        let axis: [f64; N] = core::array::from_fn(|i| b[i] - a[i]);
        let focal_distance = norm(&axis);

        if N == 0 || !max_cost.is_finite() || max_cost <= focal_distance {
            return None;
        }

        Some(Self {
            center: core::array::from_fn(|i| f64::midpoint(a[i], b[i])),
            focus_a: a,
            focus_b: b,
            reflection: householder_vector(&axis, focal_distance),
            major_radius: 0.5 * max_cost,
            minor_radius: 0.5
                * libm::sqrt((max_cost - focal_distance) * (max_cost + focal_distance)),
        })
    }

    /// The Lebesgue measure of the set: `zeta_N r_1 r_2^(N - 1)`.
    #[must_use]
    pub fn volume(&self) -> f64 {
        let n = count_to_f64(N);

        unit_ball_volume(n) * self.major_radius * libm::pow(self.minor_radius, n - 1.0)
    }

    /// Whether `x` lies in the closed set: `|x - a| + |x - b| <= max_cost`.
    #[must_use]
    pub fn contains(&self, x: &[f64; N]) -> bool {
        self.cost_through(x) <= 2.0 * self.major_radius
    }

    /// The length `|x - a| + |x - b|` of the shortest path from `a` through `x` to `b`.
    #[must_use]
    pub fn cost_through(&self, x: &[f64; N]) -> f64 {
        distance(&self.focus_a, x) + distance(x, &self.focus_b)
    }

    /// Draws a state uniformly from the set.
    ///
    /// Uses `N + 1` RNG draws when `N` is even and `N + 2` when it is odd (Box-Muller pairs and
    /// one draw for the radius), plus a retry in the probability-zero case of an all-zero
    /// Gaussian vector.
    pub fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> [f64; N] {
        let ball = unit_ball_sample::<N, R>(rng);
        let mut x: [f64; N] = core::array::from_fn(|i| {
            let radius = if i == 0 {
                self.major_radius
            } else {
                self.minor_radius
            };

            radius * ball[i]
        });

        if let Some(v) = &self.reflection {
            reflect(&mut x, v);
        }

        core::array::from_fn(|i| self.center[i] + x[i])
    }
}

/// The Euclidean norm of `v`.
fn norm<const N: usize>(v: &[f64; N]) -> f64 {
    libm::sqrt(v.iter().map(|x| x * x).sum())
}

/// The Euclidean distance between `a` and `b`, computed like
/// [`Euclidean::distance`](crate::StateSpace::distance).
fn distance<const N: usize>(a: &[f64; N], b: &[f64; N]) -> f64 {
    let squared: f64 = a.iter().zip(b).map(|(p, q)| (q - p) * (q - p)).sum();

    libm::sqrt(squared)
}

/// The Householder vector `v = e_1 - axis / length` whose reflection `I - 2 v v^T / v^T v`
/// maps `e_1` onto the unit vector along `axis`, or `None` if the identity already does.
fn householder_vector<const N: usize>(axis: &[f64; N], length: f64) -> Option<[f64; N]> {
    if length == 0.0 {
        return None;
    }

    let v: [f64; N] = core::array::from_fn(|i| {
        let unit = axis[i] / length;

        if i == 0 { 1.0 - unit } else { -unit }
    });

    // Below this the reflection is numerically the identity and `v^T v` would underflow.
    (norm(&v) > 1e-12).then_some(v)
}

/// Applies the Householder reflection with vector `v` to `x` in place.
fn reflect<const N: usize>(x: &mut [f64; N], v: &[f64; N]) {
    let vv: f64 = v.iter().map(|c| c * c).sum();
    let vx: f64 = v.iter().zip(x.iter()).map(|(a, b)| a * b).sum();
    let scale = 2.0 * vx / vv;

    for (xi, vi) in x.iter_mut().zip(v) {
        *xi -= scale * vi;
    }
}

/// A uniform sample of the closed unit ball in R^N: a normalized Gaussian vector (uniform on
/// the sphere) scaled by `u^(1/N)`.
fn unit_ball_sample<const N: usize, R: Rng + ?Sized>(rng: &mut R) -> [f64; N] {
    loop {
        let direction = gaussian_vector::<N, R>(rng);
        let length = norm(&direction);

        if length > 0.0 {
            let radius = libm::pow(unit_f64(rng), 1.0 / count_to_f64(N));

            return core::array::from_fn(|i| direction[i] * radius / length);
        }
    }
}

/// `N` independent standard normal values from Box-Muller pairs.
fn gaussian_vector<const N: usize, R: Rng + ?Sized>(rng: &mut R) -> [f64; N] {
    let mut out = [0.0; N];

    for pair in out.chunks_mut(2) {
        // `1 - u` lies in (0, 1], so the logarithm is finite.
        let radius = libm::sqrt(-2.0 * libm::log(1.0 - unit_f64(rng)));
        let angle = 2.0 * core::f64::consts::PI * unit_f64(rng);

        pair[0] = radius * libm::cos(angle);
        if let Some(second) = pair.get_mut(1) {
            *second = radius * libm::sin(angle);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::SeedableRng;

    #[test]
    fn degenerate_sets_are_rejected() {
        assert!(ProlateHyperspheroid::new([0.0, 0.0], [3.0, 4.0], 5.0).is_none());
        assert!(ProlateHyperspheroid::new([0.0, 0.0], [3.0, 4.0], f64::INFINITY).is_none());
        assert!(ProlateHyperspheroid::<0>::new([], [], 1.0).is_none());
    }

    #[test]
    fn volume_matches_the_closed_form_ellipse_area() {
        let ellipse = ProlateHyperspheroid::new([-3.0, 0.0], [3.0, 0.0], 10.0).expect("valid");

        // Semi-axes 5 and 4.
        assert!((ellipse.volume() - core::f64::consts::PI * 20.0).abs() < 1e-9);
    }

    #[test]
    fn samples_are_inside_and_fill_the_set_uniformly() {
        // A tilted 3D spheroid; coincident-direction and coincident-foci cases below.
        for (a, b) in [
            ([1.0, 2.0, 3.0], [4.0, -2.0, 5.0]),
            ([0.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
            ([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]),
        ] {
            let max_cost = distance(&a, &b) + 2.0;
            let set = ProlateHyperspheroid::new(a, b, max_cost).expect("valid");
            assert_uniform(&set);
        }
    }

    /// Asserts that samples of `set` stay inside and that the share of samples in the inner
    /// copy `cost_through <= c_min + (c - c_min) / 2` matches its volume share.
    fn assert_uniform(set: &ProlateHyperspheroid<3>) {
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(4);
        let c_min = distance(&set.focus_a, &set.focus_b);
        let c_max = 2.0 * set.major_radius;
        let inner_cost = f64::midpoint(c_min, c_max);
        let inner = ProlateHyperspheroid::new(set.focus_a, set.focus_b, inner_cost).expect("valid");
        let samples = 40_000;
        let mut hits = 0_u32;
        let mut mean = [0.0; 3];

        for _ in 0..samples {
            let x = set.sample(&mut rng);
            assert!(set.cost_through(&x) <= c_max + 1e-9, "{x:?} outside");

            hits += u32::from(set.cost_through(&x) <= inner_cost);
            for (m, xi) in mean.iter_mut().zip(x) {
                *m += xi / f64::from(samples);
            }
        }

        let share = f64::from(hits) / f64::from(samples);
        let expected = inner.volume() / set.volume();
        assert!((share - expected).abs() < 0.01, "{share} vs {expected}");

        for (m, c) in mean.iter().zip(set.center) {
            assert!((m - c).abs() < 0.03 * set.major_radius, "mean {mean:?}");
        }
    }
}
