//! Reeds-Shepp curves: shortest paths of a car that drives forward and backward (Reeds and
//! Shepp 1990).

use alloc::vec::Vec;
use core::f64::consts::{FRAC_PI_2, PI, TAU};

use super::Se2State;
use super::curve::{Segment, SegmentKind, local_goal, total_length, walk};
use crate::StateSpace;

/// Closed-form solutions can produce slightly negative segment lengths from rounding; values
/// above `-ZERO` count as non-negative (OMPL's `ZERO`).
const ZERO: f64 = 1e-10;

/// A local goal `(x, y, phi)` with `sin(phi)` and `cos(phi)` precomputed, since every formula
/// needs them and the symmetric variants only flip their signs.
#[derive(Debug, Clone, Copy)]
struct Goal {
    /// The goal x in units of the radius.
    x: f64,
    /// The goal y in units of the radius.
    y: f64,
    /// The heading change.
    phi: f64,
    /// `sin(phi)`.
    sin: f64,
    /// `cos(phi)`.
    cos: f64,
}

/// A formula of Reeds and Shepp: the base word lengths `(t, u, v)` for a local goal, if the
/// word is feasible.
type Formula = fn(Goal) -> Option<(f64, f64, f64)>;

/// Maps the base lengths `(t, u, v)` of a formula to the segment lengths of a word.
type Layout = fn(f64, f64, f64) -> [f64; 5];

/// The 18 Reeds-Shepp word shapes of OMPL's `reedsSheppPathType`, by index. Each is used
/// forward and time-flipped (all lengths negated), which gives the full set of words.
const WORDS: [&[SegmentKind]; 18] = {
    use SegmentKind::{Left as L, Right as R, Straight as S};

    [
        &[L, R, L],
        &[R, L, R],
        &[L, R, L, R],
        &[R, L, R, L],
        &[L, R, S, L],
        &[R, L, S, R],
        &[L, S, R, L],
        &[R, S, L, R],
        &[L, R, S, R],
        &[R, L, S, L],
        &[R, S, R, L],
        &[L, S, L, R],
        &[L, S, R],
        &[R, S, L],
        &[L, S, L],
        &[R, S, R],
        &[L, R, S, L, R],
        &[R, L, S, R, L],
    ]
};

/// A Reeds-Shepp path: up to five segments with signed lengths; negative lengths drive in
/// reverse.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReedsSheppPath {
    /// The segments in driving order; only the first `count` are used.
    segments: [Segment; 5],
    /// The number of segments, three to five.
    count: usize,
    /// The turning radius the path was computed for.
    pub radius: f64,
}

impl ReedsSheppPath {
    /// The path of word `word` with signed normalized `lengths`.
    fn new(word: usize, lengths: [f64; 5], radius: f64) -> Self {
        let kinds = WORDS[word];

        Self {
            segments: core::array::from_fn(|i| Segment {
                kind: kinds.get(i).copied().unwrap_or(SegmentKind::Straight),
                length: if i < kinds.len() { lengths[i] } else { 0.0 },
            }),
            count: kinds.len(),
            radius,
        }
    }

    /// The segments in driving order. Lengths are signed, in units of the radius.
    #[must_use]
    pub fn segments(&self) -> &[Segment] {
        &self.segments[..self.count]
    }

    /// The path length in world units: the sum of the absolute segment lengths.
    #[must_use]
    pub fn length(&self) -> f64 {
        self.radius * total_length(self.segments())
    }

    /// The pose after driving `distance` (world units, clamped to the path) from `from`.
    #[must_use]
    pub fn state_at(&self, from: &Se2State, distance: f64) -> Se2State {
        walk(
            from,
            self.segments(),
            distance.max(0.0) / self.radius,
            self.radius,
        )
    }
}

/// The Reeds-Shepp car: SE(2) poses ([`Se2State`]) of a vehicle that drives forward or
/// backward at unit speed with a minimum turning radius.
///
/// [`StateSpace::distance`] is the length of the shortest Reeds-Shepp path, a true metric
/// (symmetric, triangle inequality), and [`StateSpace::interpolate`] follows that path.
/// Ported from OMPL's `ReedsSheppStateSpace` (Mark Moll), which evaluates the formulas 8.1 to
/// 8.11 of Reeds and Shepp with the paper's typos fixed, each under time flip, reflection and
/// backward symmetries. [`ReedsShepp::candidates`] lists all of those words.
///
/// [`StraightLine`](crate::StraightLine) steering works as is, as for
/// [`Dubins`](crate::Dubins).
///
/// # Examples
///
/// ```
/// use sampling_planning::{ReedsShepp, Se2State, StateSpace};
///
/// let car = ReedsShepp::new(1.0);
/// let back = car.distance(&Se2State::new(0.0, 0.0, 0.0), &Se2State::new(-3.0, 0.0, 0.0));
///
/// assert!((back - 3.0).abs() < 1e-9, "reversing three units");
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReedsShepp {
    /// The minimum turning radius; positive and finite.
    radius: f64,
}

impl ReedsShepp {
    /// The Reeds-Shepp car with the given minimum turning radius.
    ///
    /// # Panics
    ///
    /// Panics if `radius` is not positive and finite.
    #[must_use]
    pub fn new(radius: f64) -> Self {
        assert!(
            radius.is_finite() && radius > 0.0,
            "turning radius must be positive and finite"
        );

        Self { radius }
    }

    /// The minimum turning radius.
    #[must_use]
    pub const fn radius(&self) -> f64 {
        self.radius
    }

    /// The shortest Reeds-Shepp path from `from` to `to`; the earliest candidate on ties.
    ///
    /// # Panics
    ///
    /// Never for finite poses: one of the CSC words always has a solution. Non-finite poses
    /// panic.
    #[must_use]
    pub fn path(&self, from: &Se2State, to: &Se2State) -> ReedsSheppPath {
        let (x, y, phi) = local_goal(from, to, self.radius);
        let mut best: Option<(f64, usize, [f64; 5])> = None;

        for_each_candidate(x, y, phi, |word, lengths| {
            let length: f64 = lengths.iter().map(|l| libm::fabs(*l)).sum();

            if best.is_none_or(|(b, _, _)| length < b) {
                best = Some((length, word, lengths));
            }
        });

        let (_, word, lengths) = best.expect("the CSC family always has a solution");

        ReedsSheppPath::new(word, lengths, self.radius)
    }

    /// Every feasible candidate from `from` to `to` in the order they are evaluated, for
    /// inspection and testing. [`ReedsShepp::path`] picks the shortest of them.
    #[must_use]
    pub fn candidates(&self, from: &Se2State, to: &Se2State) -> Vec<ReedsSheppPath> {
        let (x, y, phi) = local_goal(from, to, self.radius);
        let mut all = Vec::new();

        for_each_candidate(x, y, phi, |word, lengths| {
            all.push(ReedsSheppPath::new(word, lengths, self.radius));
        });

        all
    }
}

impl StateSpace for ReedsShepp {
    type State = Se2State;

    fn is_symmetric(&self) -> bool {
        true
    }

    fn dimension(&self) -> usize {
        3
    }

    fn distance(&self, from: &Se2State, to: &Se2State) -> f64 {
        self.path(from, to).length()
    }

    fn interpolate(&self, from: &Se2State, to: &Se2State, t: f64) -> Se2State {
        if t <= 0.0 {
            return *from;
        }

        if t >= 1.0 {
            return *to;
        }

        let path = self.path(from, to);

        path.state_at(from, t * path.length())
    }
}

/// Calls `emit(word, lengths)` for every feasible candidate of the local goal `(x, y, phi)`,
/// in OMPL's order: CSC, CCC, CCCC, CCSC, CCSCC.
fn for_each_candidate(x: f64, y: f64, phi: f64, mut emit: impl FnMut(usize, [f64; 5])) {
    let (sin, cos) = (libm::sin(phi), libm::cos(phi));
    let (xb, yb) = (x * cos + y * sin, x * sin - y * cos);
    let mut family = |at: (f64, f64), formula: Formula, words: (usize, usize), layout: Layout| {
        let goal = Goal {
            x: at.0,
            y: at.1,
            phi,
            sin,
            cos,
        };
        symmetric_variants(goal, formula, words, layout, &mut emit);
    };

    family((x, y), lp_sp_lp, (14, 15), |t, u, v| [t, u, v, 0.0, 0.0]);
    family((x, y), lp_sp_rp, (12, 13), |t, u, v| [t, u, v, 0.0, 0.0]);
    family((x, y), lp_rm_l, (0, 1), |t, u, v| [t, u, v, 0.0, 0.0]);
    family((xb, yb), lp_rm_l, (0, 1), |t, u, v| [v, u, t, 0.0, 0.0]);
    family((x, y), lp_rup_lum_rm, (2, 3), |t, u, v| [t, u, -u, v, 0.0]);
    family((x, y), lp_rum_lum_rp, (2, 3), |t, u, v| [t, u, u, v, 0.0]);
    family((x, y), lp_rm_sm_lm, (4, 5), |t, u, v| {
        [t, -FRAC_PI_2, u, v, 0.0]
    });
    family((x, y), lp_rm_sm_rm, (8, 9), |t, u, v| {
        [t, -FRAC_PI_2, u, v, 0.0]
    });
    family((xb, yb), lp_rm_sm_lm, (6, 7), |t, u, v| {
        [v, u, -FRAC_PI_2, t, 0.0]
    });
    family((xb, yb), lp_rm_sm_rm, (10, 11), |t, u, v| {
        [v, u, -FRAC_PI_2, t, 0.0]
    });
    family((x, y), lp_rm_s_lm_rp, (16, 17), |t, u, v| {
        [t, -FRAC_PI_2, u, -FRAC_PI_2, v]
    });
}

/// Evaluates `formula` for the goal and its time flip, reflection and both: words
/// `words.0` for the plain and time-flipped goals, `words.1` for the reflected ones; time flips
/// negate every length.
fn symmetric_variants(
    goal: Goal,
    formula: Formula,
    words: (usize, usize),
    layout: Layout,
    emit: &mut impl FnMut(usize, [f64; 5]),
) {
    let Goal {
        x,
        y,
        phi,
        sin,
        cos,
    } = goal;
    let flipped = |x, y, phi, sin| Goal {
        x,
        y,
        phi,
        sin,
        cos,
    };
    let variants = [
        (goal, words.0, 1.0),
        (flipped(-x, y, -phi, -sin), words.0, -1.0),
        (flipped(x, -y, -phi, -sin), words.1, 1.0),
        (flipped(-x, -y, phi, sin), words.1, -1.0),
    ];

    for (variant, word, sign) in variants {
        if let Some((t, u, v)) = formula(variant) {
            emit(word, layout(t, u, v).map(|length| sign * length));
        }
    }
}

/// An angle wrapped into `[-pi, pi]`, like OMPL's `fmod`-based `mod2pi` up to rounding.
fn mod2pi(x: f64) -> f64 {
    // The arguments are sums of a few angles, so one rounded quotient is exact enough and much
    // cheaper than `fmod`.
    let v = x - TAU * libm::round(x / TAU);

    if v < -PI {
        v + TAU
    } else if v > PI {
        v - TAU
    } else {
        v
    }
}

/// Polar coordinates `(r, theta)` of `(x, y)`.
fn polar(x: f64, y: f64) -> (f64, f64) {
    (libm::sqrt(x * x + y * y), libm::atan2(y, x))
}

/// The `tau` and `omega` helper of formulas 8.7 and 8.8.
fn tau_omega(u: f64, v: f64, xi: f64, eta: f64, phi: f64) -> (f64, f64) {
    let delta = mod2pi(u - v);
    let a = libm::sin(u) - libm::sin(delta);
    let b = libm::cos(u) - libm::cos(delta) - 1.0;
    let t1 = libm::atan2(eta * a - xi * b, xi * a + eta * b);
    let t2 = 2.0 * (libm::cos(delta) - libm::cos(v) - libm::cos(u)) + 3.0;
    let tau = if t2 < 0.0 {
        mod2pi(t1 + PI)
    } else {
        mod2pi(t1)
    };

    (tau, mod2pi(tau - u + v - phi))
}

/// Formula 8.1: L+ S+ L+.
fn lp_sp_lp(g: Goal) -> Option<(f64, f64, f64)> {
    let Goal {
        x,
        y,
        phi,
        sin,
        cos,
    } = g;
    let (u, t) = polar(x - sin, y - 1.0 + cos);
    let v = mod2pi(phi - t);

    (t >= -ZERO && v >= -ZERO).then_some((t, u, v))
}

/// Formula 8.2: L+ S+ R+.
fn lp_sp_rp(g: Goal) -> Option<(f64, f64, f64)> {
    let Goal {
        x,
        y,
        phi,
        sin,
        cos,
    } = g;
    let (u1, t1) = polar(x + sin, y - 1.0 - cos);
    let u1 = u1 * u1;

    if u1 < 4.0 {
        return None;
    }

    let u = libm::sqrt(u1 - 4.0);
    let t = mod2pi(t1 + libm::atan2(2.0, u));
    let v = mod2pi(t - phi);

    (t >= -ZERO && v >= -ZERO).then_some((t, u, v))
}

/// Formulas 8.3 and 8.4 (with the paper's typo fixed): L+ R- L.
fn lp_rm_l(g: Goal) -> Option<(f64, f64, f64)> {
    let Goal {
        x,
        y,
        phi,
        sin,
        cos,
    } = g;
    let (u1, theta) = polar(x - sin, y - 1.0 + cos);

    if u1 > 4.0 {
        return None;
    }

    let u = -2.0 * libm::asin(0.25 * u1);
    let t = mod2pi(theta + 0.5 * u + PI);
    let v = mod2pi(phi - t + u);

    (t >= -ZERO && u <= ZERO).then_some((t, u, v))
}

/// Formula 8.7: L+ R+ L- R-.
fn lp_rup_lum_rm(g: Goal) -> Option<(f64, f64, f64)> {
    let Goal {
        x,
        y,
        phi,
        sin,
        cos,
    } = g;
    let xi = x + sin;
    let eta = y - 1.0 - cos;
    let rho = 0.25 * (2.0 + libm::sqrt(xi * xi + eta * eta));

    if rho > 1.0 {
        return None;
    }

    let u = libm::acos(rho);
    let (t, v) = tau_omega(u, -u, xi, eta, phi);

    (t >= -ZERO && v <= ZERO).then_some((t, u, v))
}

/// Formula 8.8: L+ R- L- R+.
fn lp_rum_lum_rp(g: Goal) -> Option<(f64, f64, f64)> {
    let Goal {
        x,
        y,
        phi,
        sin,
        cos,
    } = g;
    let xi = x + sin;
    let eta = y - 1.0 - cos;
    let rho = (20.0 - xi * xi - eta * eta) / 16.0;

    if !(0.0..=1.0).contains(&rho) {
        return None;
    }

    let u = -libm::acos(rho);
    if u < -0.5 * PI {
        return None;
    }

    let (t, v) = tau_omega(u, u, xi, eta, phi);

    (t >= -ZERO && v >= -ZERO).then_some((t, u, v))
}

/// Formula 8.9: L+ R-(pi/2) S- L-.
#[expect(
    clippy::many_single_char_names,
    reason = "t, u, v, x, y follow the nomenclature of Reeds and Shepp"
)]
fn lp_rm_sm_lm(g: Goal) -> Option<(f64, f64, f64)> {
    let Goal {
        x,
        y,
        phi,
        sin,
        cos,
    } = g;
    let (rho, theta) = polar(x - sin, y - 1.0 + cos);

    if rho < 2.0 {
        return None;
    }

    let r = libm::sqrt(rho * rho - 4.0);
    let u = 2.0 - r;
    let t = mod2pi(theta + libm::atan2(r, -2.0));
    let v = mod2pi(phi - 0.5 * PI - t);

    (t >= -ZERO && u <= ZERO && v <= ZERO).then_some((t, u, v))
}

/// Formula 8.10: L+ R-(pi/2) S- R-.
fn lp_rm_sm_rm(g: Goal) -> Option<(f64, f64, f64)> {
    let Goal {
        x,
        y,
        phi,
        sin,
        cos,
    } = g;
    let xi = x + sin;
    let eta = y - 1.0 - cos;
    let (rho, theta) = polar(-eta, xi);

    if rho < 2.0 {
        return None;
    }

    let t = theta;
    let u = 2.0 - rho;
    let v = mod2pi(t + 0.5 * PI - phi);

    (t >= -ZERO && u <= ZERO && v <= ZERO).then_some((t, u, v))
}

/// Formula 8.11 (with the paper's typo fixed): L+ R-(pi/2) S- L-(pi/2) R+.
fn lp_rm_s_lm_rp(g: Goal) -> Option<(f64, f64, f64)> {
    let Goal {
        x,
        y,
        phi,
        sin,
        cos,
    } = g;
    let xi = x + sin;
    let eta = y - 1.0 - cos;
    let (rho, _) = polar(xi, eta);

    if rho < 2.0 {
        return None;
    }

    let u = 4.0 - libm::sqrt(rho * rho - 4.0);
    if u > ZERO {
        return None;
    }

    let t = mod2pi(libm::atan2(
        (4.0 - u) * xi - 2.0 * eta,
        -2.0 * xi + (u - 4.0) * eta,
    ));
    let v = mod2pi(t - phi);

    (t >= -ZERO && v >= -ZERO).then_some((t, u, v))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Dubins;
    use crate::space::curve::tests::{assert_same_pose, pose_pairs};

    #[test]
    fn known_path_lengths() {
        let car = ReedsShepp::new(1.0);
        let origin = Se2State::new(0.0, 0.0, 0.0);

        assert!((car.distance(&origin, &Se2State::new(4.0, 0.0, 0.0)) - 4.0).abs() < 1e-9);
        assert!((car.distance(&origin, &Se2State::new(-4.0, 0.0, 0.0)) - 4.0).abs() < 1e-9);
        assert!((car.distance(&origin, &Se2State::new(0.0, 2.0, PI)) - PI).abs() < 1e-9);
        assert!(car.distance(&origin, &origin).abs() < 1e-12);

        // A quarter turn reversing to the right: back and right is a reversed left arc.
        let reversed = Se2State::new(-1.0, 1.0, -FRAC_PI_2);
        assert!((car.distance(&origin, &reversed) - FRAC_PI_2).abs() < 1e-9);
    }

    #[test]
    fn every_candidate_reaches_its_target_and_all_families_occur() {
        let mut words_seen = [false; 18];

        for radius in [0.5, 1.0, 2.5] {
            let car = ReedsShepp::new(radius);

            for (from, to) in pose_pairs(600, 5) {
                for path in car.candidates(&from, &to) {
                    assert_same_pose(&path.state_at(&from, path.length()), &to, 1e-6);

                    let kinds: Vec<_> = path.segments().iter().map(|s| s.kind).collect();
                    let word = WORDS
                        .iter()
                        .position(|w| *w == kinds.as_slice())
                        .expect("known");
                    words_seen[word] = true;
                }
            }
        }

        assert_eq!(
            words_seen, [true; 18],
            "every word shape was a feasible candidate"
        );
    }

    #[test]
    fn distance_is_a_metric_no_longer_than_dubins() {
        let (car, forward) = (ReedsShepp::new(1.2), Dubins::new(1.2));
        let pairs = pose_pairs(400, 6);

        for window in pairs.windows(2) {
            let ((a, b), (c, _)) = (window[0], window[1]);
            let ab = car.distance(&a, &b);

            assert!((ab - car.distance(&b, &a)).abs() < 1e-9, "symmetric");
            // Dubins snaps heading differences below 1e-6 at one spot to zero length.
            assert!(ab <= forward.distance(&a, &b) + 1e-6);
            assert!(car.distance(&a, &c) <= ab + car.distance(&b, &c) + 1e-6);
        }
    }

    #[test]
    fn interpolation_follows_the_path() {
        let car = ReedsShepp::new(1.0);

        // OMPL tests/base/state_spaces.cpp, ReedsShepp_InterpolatedDistance.
        let from = Se2State::new(
            -1.154_810_577_189_199_2,
            -2.493_659_442_506_264,
            1.290_460_292_765_287_2,
        );
        let to = Se2State::new(
            0.680_688_762_351_618_2,
            0.957_761_849_729_337_2,
            -1.377_344_206_800_952_3,
        );
        let t = 0.043_926_969_507_538_036;
        let mid = car.interpolate(&from, &to, t);
        assert!(car.distance(&from, &mid) <= t * car.distance(&from, &to) + 1e-9);

        for (from, to) in pose_pairs(300, 7) {
            let total = car.distance(&from, &to);
            assert_eq!(car.interpolate(&from, &to, 0.0), from);
            assert_eq!(car.interpolate(&from, &to, 1.0), to);

            for t in [0.25, 0.5, 0.75] {
                let mid = car.interpolate(&from, &to, t);
                assert!((car.distance(&from, &mid) - t * total).abs() < 1e-6);
            }
        }
    }
}
