//! Dubins curves: shortest paths of a car that only drives forward (Dubins 1957).

use core::f64::consts::TAU;

use super::Se2State;
use super::curve::{Segment, SegmentKind, total_length, walk};
use crate::StateSpace;

/// Below this the normalized distance and heading difference count as zero (OMPL's
/// `DUBINS_EPS`).
const DUBINS_EPS: f64 = 1e-6;

/// Slightly negative values above this count as zero (OMPL's `DUBINS_ZERO`).
const DUBINS_ZERO: f64 = -1e-7;

/// The six Dubins words: L and R are arcs of the turning radius, S is a straight line.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DubinsWord {
    /// Left, straight, left.
    Lsl,
    /// Right, straight, right.
    Rsr,
    /// Right, straight, left.
    Rsl,
    /// Left, straight, right.
    Lsr,
    /// Right, left, right.
    Rlr,
    /// Left, right, left.
    Lrl,
}

impl DubinsWord {
    /// All words, in the order [`Dubins::candidates`] reports them.
    pub const ALL: [Self; 6] = [
        Self::Lsl,
        Self::Rsr,
        Self::Rsl,
        Self::Lsr,
        Self::Rlr,
        Self::Lrl,
    ];

    /// The segment motions of this word.
    #[must_use]
    pub const fn kinds(self) -> [SegmentKind; 3] {
        use SegmentKind::{Left as L, Right as R, Straight as S};

        match self {
            Self::Lsl => [L, S, L],
            Self::Rsr => [R, S, R],
            Self::Rsl => [R, S, L],
            Self::Lsr => [L, S, R],
            Self::Rlr => [R, L, R],
            Self::Lrl => [L, R, L],
        }
    }
}

/// A Dubins path: a word and its three non-negative segment lengths.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DubinsPath {
    /// The word.
    pub word: DubinsWord,
    /// The segments, in driving order; lengths are non-negative, in units of the radius.
    pub segments: [Segment; 3],
    /// The turning radius the path was computed for.
    pub radius: f64,
}

impl DubinsPath {
    /// The path for `word` with normalized segment lengths `lengths`.
    fn new(word: DubinsWord, lengths: [f64; 3], radius: f64) -> Self {
        let kinds = word.kinds();

        Self {
            word,
            segments: core::array::from_fn(|i| Segment {
                kind: kinds[i],
                length: lengths[i],
            }),
            radius,
        }
    }

    /// The path length in world units.
    #[must_use]
    pub fn length(&self) -> f64 {
        self.radius * total_length(&self.segments)
    }

    /// The pose after driving `distance` (world units, clamped to the path) from `from`.
    #[must_use]
    pub fn state_at(&self, from: &Se2State, distance: f64) -> Se2State {
        walk(
            from,
            &self.segments,
            distance.max(0.0) / self.radius,
            self.radius,
        )
    }
}

/// The Dubins car: SE(2) poses ([`Se2State`]) of a vehicle that drives only forward, at unit
/// speed, with a minimum turning radius.
///
/// [`StateSpace::distance`] is the length of the shortest Dubins path and
/// [`StateSpace::interpolate`] follows that path. Ported from OMPL's `DubinsStateSpace`
/// (Mark Moll), with the exhaustive search over all six words instead of OMPL's
/// classification shortcut for long paths: both return the shortest word. As in OMPL, poses
/// closer than `1e-6` turning radii with headings within `1e-6` radians count as coincident:
/// their distance is the position distance, not the loop a real car would need.
///
/// **The distance is not symmetric**: `d(a, b)` is the length from `a` to `b`, which differs
/// from the way back. Planners that measure from tree nodes toward targets (RRT, RRT*, RRT#,
/// SST) are fine; see [`StateSpace::distance`] for the planners that need symmetry. It does
/// satisfy the triangle inequality.
///
/// [`StraightLine`](crate::StraightLine) steering works as is: it interpolates the fraction
/// `max_step / d` of the way, and a prefix of a shortest path is a shortest path, so the steered
/// state is `max_step` away along the curve.
///
/// # Examples
///
/// ```
/// use sampling_planning::{Dubins, Se2State, StateSpace};
///
/// let car = Dubins::new(1.0);
/// let half_loop = car.distance(&Se2State::new(0.0, 0.0, 0.0), &Se2State::new(0.0, 2.0, std::f64::consts::PI));
///
/// assert!((half_loop - std::f64::consts::PI).abs() < 1e-9);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Dubins {
    /// The minimum turning radius; positive and finite.
    radius: f64,
}

impl Dubins {
    /// The Dubins car with the given minimum turning radius.
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

    /// The shortest Dubins path from `from` to `to`; the earliest of
    /// [`DubinsWord::ALL`] on ties.
    ///
    /// # Panics
    ///
    /// Never for finite poses: LSL and RSR always have a solution. Non-finite poses panic.
    #[must_use]
    pub fn path(&self, from: &Se2State, to: &Se2State) -> DubinsPath {
        let (d, alpha, beta) = self.normalize(from, to);

        if d < DUBINS_EPS && libm::fabs(alpha - beta) < DUBINS_EPS {
            return DubinsPath::new(DubinsWord::Lsl, [0.0, d, 0.0], self.radius);
        }

        let mut best: Option<DubinsPath> = None;
        for path in self.candidates(from, to).into_iter().flatten() {
            if best.is_none_or(|b| path.length() < b.length()) {
                best = Some(path);
            }
        }

        best.expect("LSL or RSR always exists")
    }

    /// Every feasible word from `from` to `to`, in [`DubinsWord::ALL`] order; `None` where a
    /// word has no solution (the CCC words when the poses are far apart, for example).
    #[must_use]
    pub fn candidates(&self, from: &Se2State, to: &Se2State) -> [Option<DubinsPath>; 6] {
        let (d, alpha, beta) = self.normalize(from, to);
        let trig = Trig::new(alpha, beta);

        DubinsWord::ALL.map(|word| {
            let lengths = match word {
                DubinsWord::Lsl => lsl(d, alpha, beta, &trig),
                DubinsWord::Rsr => rsr(d, alpha, beta, &trig),
                DubinsWord::Rsl => rsl(d, alpha, beta, &trig),
                DubinsWord::Lsr => lsr(d, alpha, beta, &trig),
                DubinsWord::Rlr => rlr(d, alpha, beta, &trig),
                DubinsWord::Lrl => lrl(d, alpha, beta, &trig),
            };

            lengths.map(|lengths| DubinsPath::new(word, lengths, self.radius))
        })
    }

    /// The normalized distance `d` and the headings `alpha` and `beta` relative to the line
    /// from `from` to `to`, as in Shkel and Lumelsky.
    fn normalize(self, from: &Se2State, to: &Se2State) -> (f64, f64, f64) {
        let (dx, dy) = (to.x - from.x, to.y - from.y);
        let d = libm::hypot(dx, dy) / self.radius;
        let th = libm::atan2(dy, dx);

        (d, mod2pi(from.theta - th), mod2pi(to.theta - th))
    }
}

impl StateSpace for Dubins {
    type State = Se2State;

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

/// `x` modulo `2 pi` in `[0, 2 pi)`, with OMPL's snapping of near-zero and near-`2 pi` values
/// to zero.
fn mod2pi(x: f64) -> f64 {
    if x < 0.0 && x > DUBINS_ZERO {
        return 0.0;
    }

    let xm = x - TAU * libm::floor(x / TAU);

    if TAU - xm < 0.5 * DUBINS_EPS { 0.0 } else { xm }
}

/// Sines and cosines of `alpha` and `beta`, shared by the six word formulas.
struct Trig {
    /// `cos(alpha)`.
    ca: f64,
    /// `sin(alpha)`.
    sa: f64,
    /// `cos(beta)`.
    cb: f64,
    /// `sin(beta)`.
    sb: f64,
}

impl Trig {
    /// The sines and cosines of the two headings.
    fn new(alpha: f64, beta: f64) -> Self {
        Self {
            ca: libm::cos(alpha),
            sa: libm::sin(alpha),
            cb: libm::cos(beta),
            sb: libm::sin(beta),
        }
    }
}

/// The LSL lengths, if the word is feasible.
fn lsl(d: f64, alpha: f64, beta: f64, g: &Trig) -> Option<[f64; 3]> {
    let tmp = 2.0 + d * d - 2.0 * (g.ca * g.cb + g.sa * g.sb - d * (g.sa - g.sb));
    (tmp >= DUBINS_ZERO).then(|| {
        let theta = libm::atan2(g.cb - g.ca, d + g.sa - g.sb);

        [
            mod2pi(-alpha + theta),
            libm::sqrt(tmp.max(0.0)),
            mod2pi(beta - theta),
        ]
    })
}

/// The RSR lengths, if the word is feasible.
fn rsr(d: f64, alpha: f64, beta: f64, g: &Trig) -> Option<[f64; 3]> {
    let tmp = 2.0 + d * d - 2.0 * (g.ca * g.cb + g.sa * g.sb - d * (g.sb - g.sa));
    (tmp >= DUBINS_ZERO).then(|| {
        let theta = libm::atan2(g.ca - g.cb, d - g.sa + g.sb);

        [
            mod2pi(alpha - theta),
            libm::sqrt(tmp.max(0.0)),
            mod2pi(-beta + theta),
        ]
    })
}

/// The RSL lengths, if the word is feasible.
fn rsl(d: f64, alpha: f64, beta: f64, g: &Trig) -> Option<[f64; 3]> {
    let tmp = d * d - 2.0 + 2.0 * (g.ca * g.cb + g.sa * g.sb - d * (g.sa + g.sb));
    (tmp >= DUBINS_ZERO).then(|| {
        let p = libm::sqrt(tmp.max(0.0));
        let theta = libm::atan2(g.ca + g.cb, d - g.sa - g.sb) - libm::atan2(2.0, p);

        [mod2pi(alpha - theta), p, mod2pi(beta - theta)]
    })
}

/// The LSR lengths, if the word is feasible.
fn lsr(d: f64, alpha: f64, beta: f64, g: &Trig) -> Option<[f64; 3]> {
    let tmp = -2.0 + d * d + 2.0 * (g.ca * g.cb + g.sa * g.sb + d * (g.sa + g.sb));
    (tmp >= DUBINS_ZERO).then(|| {
        let p = libm::sqrt(tmp.max(0.0));
        let theta = libm::atan2(-g.ca - g.cb, d + g.sa + g.sb) - libm::atan2(-2.0, p);

        [mod2pi(-alpha + theta), p, mod2pi(-beta + theta)]
    })
}

/// The RLR lengths, if the word is feasible.
fn rlr(d: f64, alpha: f64, beta: f64, g: &Trig) -> Option<[f64; 3]> {
    let tmp = 0.125 * (6.0 - d * d + 2.0 * (g.ca * g.cb + g.sa * g.sb + d * (g.sa - g.sb)));
    (libm::fabs(tmp) < 1.0).then(|| {
        let p = TAU - libm::acos(tmp);
        let theta = libm::atan2(g.ca - g.cb, d - g.sa + g.sb);
        let t = mod2pi(alpha - theta + 0.5 * p);

        [t, p, mod2pi(alpha - beta - t + p)]
    })
}

/// The LRL lengths, if the word is feasible.
fn lrl(d: f64, alpha: f64, beta: f64, g: &Trig) -> Option<[f64; 3]> {
    let tmp = 0.125 * (6.0 - d * d + 2.0 * (g.ca * g.cb + g.sa * g.sb - d * (g.sa - g.sb)));
    (libm::fabs(tmp) < 1.0).then(|| {
        let p = TAU - libm::acos(tmp);
        let theta = libm::atan2(-g.ca + g.cb, d + g.sa - g.sb);
        let t = mod2pi(-alpha + theta + 0.5 * p);

        [t, p, mod2pi(beta - alpha - t + p)]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::space::curve::tests::{assert_same_pose, pose_pairs};
    use core::f64::consts::{FRAC_PI_2, PI};

    /// The Dubins distance between two poses given as tuples.
    fn length(radius: f64, from: (f64, f64, f64), to: (f64, f64, f64)) -> f64 {
        Dubins::new(radius).distance(
            &Se2State::new(from.0, from.1, from.2),
            &Se2State::new(to.0, to.1, to.2),
        )
    }

    #[test]
    fn known_path_lengths() {
        // pydubins tests/test-dubins.py and Dubins-Curves tests/stableapi_tests.cpp.
        assert!((length(1.0, (0.0, 0.0, 0.0), (10.0, 0.0, 0.0)) - 10.0).abs() < 1e-9);
        assert!((length(2.0, (0.0, 0.0, 0.0), (10.0, 0.0, 0.0)) - 10.0).abs() < 1e-9);
        assert!((length(1.0, (0.0, 0.0, 0.0), (4.0, 0.0, 0.0)) - 4.0).abs() < 1e-9);
        assert!((length(2.0, (0.0, 0.0, 0.0), (0.0, -0.1, 0.0)) - (4.0 * PI + 0.1)).abs() < 1e-9);
        assert!((length(1.0, (0.0, 0.0, 0.0), (0.0, 2.0, -PI)) - PI).abs() < 1e-9);
        assert!(length(1.0, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)).abs() < 1e-12);
        assert!(
            length(2.0, (0.0, 0.0, 0.0), (10.0, 10.0, FRAC_PI_2 / 2.0))
                > length(1.0, (0.0, 0.0, 0.0), (10.0, 10.0, FRAC_PI_2 / 2.0))
        );

        // Hand-derived: a quarter turn left of radius 2, then 3 straight up.
        assert!((length(2.0, (0.0, 0.0, 0.0), (2.0, 5.0, FRAC_PI_2)) - (PI + 3.0)).abs() < 1e-9);
    }

    #[test]
    fn known_words_and_segments() {
        let car = Dubins::new(1.0);
        let (a, b) = (
            Se2State::new(0.0, 0.0, FRAC_PI_2),
            Se2State::new(1.0, 0.0, -FRAC_PI_2),
        );
        assert_eq!(
            car.path(&a, &b).word,
            DubinsWord::Lrl,
            "pydubins test_easy_lrl"
        );

        let (a, b) = (Se2State::new(0.0, 0.0, 0.0), Se2State::new(4.0, 0.0, 0.0));
        let lsl = car.candidates(&a, &b)[0].expect("LSL exists");
        let lengths = lsl.segments.map(|segment| segment.length);
        assert!(
            lengths
                .iter()
                .zip([0.0, 4.0, 0.0])
                .all(|(l, e)| (l - e).abs() < 1e-12)
        );

        let far = Se2State::new(10.0, 0.0, 0.0);
        assert_eq!(
            car.candidates(&a, &far)[5],
            None,
            "LRL is infeasible beyond 4 radii"
        );
    }

    #[test]
    fn every_candidate_reaches_its_target() {
        for radius in [0.5, 1.0, 2.5] {
            let car = Dubins::new(radius);

            for (from, to) in pose_pairs(500, 3) {
                for path in car.candidates(&from, &to).into_iter().flatten() {
                    assert!(path.segments.iter().all(|segment| segment.length >= 0.0));
                    assert_same_pose(&path.state_at(&from, path.length()), &to, 1e-6);
                }

                let best = car.path(&from, &to);
                assert_same_pose(&best.state_at(&from, best.length()), &to, 1e-6);
            }
        }
    }

    #[test]
    fn distance_satisfies_the_triangle_inequality_and_interpolates_along_the_path() {
        let car = Dubins::new(1.3);
        let pairs = pose_pairs(300, 4);

        for window in pairs.windows(2) {
            let ((a, b), (c, _)) = (window[0], window[1]);
            assert!(car.distance(&a, &c) <= car.distance(&a, &b) + car.distance(&b, &c) + 1e-6);
        }

        for (from, to) in pairs {
            let total = car.distance(&from, &to);
            assert_eq!(car.interpolate(&from, &to, 0.0), from);
            assert_eq!(car.interpolate(&from, &to, 1.0), to);

            for t in [0.1, 0.5, 0.9] {
                let mid = car.interpolate(&from, &to, t);
                assert!(
                    (car.distance(&from, &mid) - t * total).abs() < 1e-6,
                    "prefixes of shortest paths are shortest"
                );
            }
        }
    }

    #[test]
    #[should_panic(expected = "turning radius")]
    fn rejects_non_positive_radii() {
        let _ = Dubins::new(-1.0);
    }
}
