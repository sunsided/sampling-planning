//! Circular-arc and straight segments of car-like paths (Dubins, Reeds-Shepp).

use super::Se2State;
use super::so2::wrap_angle;

/// The motion of one path segment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SegmentKind {
    /// A counterclockwise arc of the turning radius.
    Left,
    /// A straight line.
    Straight,
    /// A clockwise arc of the turning radius.
    Right,
}

/// One segment of a Dubins or Reeds-Shepp path.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Segment {
    /// The motion.
    pub kind: SegmentKind,
    /// The signed length in units of the turning radius: the turned angle in radians for arcs.
    /// Negative lengths drive in reverse (Reeds-Shepp only).
    pub length: f64,
}

/// Drives `pose`, given in units of the turning radius, along `kind` for the signed normalized
/// length `v`.
fn advance(pose: (f64, f64, f64), kind: SegmentKind, v: f64) -> (f64, f64, f64) {
    let (x, y, phi) = pose;

    match kind {
        SegmentKind::Left => (
            x + libm::sin(phi + v) - libm::sin(phi),
            y - libm::cos(phi + v) + libm::cos(phi),
            phi + v,
        ),
        SegmentKind::Right => (
            x - libm::sin(phi - v) + libm::sin(phi),
            y + libm::cos(phi - v) - libm::cos(phi),
            phi - v,
        ),
        SegmentKind::Straight => (x + v * libm::cos(phi), y + v * libm::sin(phi), phi),
    }
}

/// The pose reached from `from` after driving the first `arc` (normalized, non-negative) of the
/// unsigned length of `segments`, with turning radius `radius`. The heading is wrapped into
/// `[-pi, pi)`.
pub(crate) fn walk(from: &Se2State, segments: &[Segment], arc: f64, radius: f64) -> Se2State {
    let mut pose = (0.0, 0.0, from.theta);
    let mut remaining = arc;

    for segment in segments {
        if remaining <= 0.0 {
            break;
        }

        let v = if segment.length < 0.0 {
            segment.length.max(-remaining)
        } else {
            segment.length.min(remaining)
        };
        remaining -= libm::fabs(v);
        pose = advance(pose, segment.kind, v);
    }

    Se2State {
        x: from.x + pose.0 * radius,
        y: from.y + pose.1 * radius,
        theta: wrap_angle(pose.2),
    }
}

/// The unsigned total length of `segments`, in units of the turning radius.
pub(crate) fn total_length(segments: &[Segment]) -> f64 {
    segments
        .iter()
        .map(|segment| libm::fabs(segment.length))
        .sum()
}

/// The goal `to` relative to `from`, rotated into
/// `from`'s heading and scaled by `1 / radius`: `(x, y, phi)` as in Reeds and Shepp.
pub(crate) fn local_goal(from: &Se2State, to: &Se2State, radius: f64) -> (f64, f64, f64) {
    let (dx, dy) = (to.x - from.x, to.y - from.y);
    let (s, c) = (libm::sin(from.theta), libm::cos(from.theta));

    (
        (c * dx + s * dy) / radius,
        (-s * dx + c * dy) / radius,
        to.theta - from.theta,
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::{Sampler, UniformSe2};
    use alloc::vec::Vec;
    use rand_core::SeedableRng;

    /// Asserts that `a` and `b` are the same pose up to `tolerance`, comparing headings modulo
    /// `2 pi`.
    pub(crate) fn assert_same_pose(a: &Se2State, b: &Se2State, tolerance: f64) {
        let heading = libm::fabs(super::super::angle_difference(a.theta, b.theta));

        assert!(
            libm::fabs(a.x - b.x) < tolerance
                && libm::fabs(a.y - b.y) < tolerance
                && heading < tolerance,
            "{a:?} vs {b:?}"
        );
    }

    /// `count` random pose pairs in `[-4, 4]^2`, plus coincident and nearly coincident pairs.
    pub(crate) fn pose_pairs(count: usize, seed: u64) -> Vec<(Se2State, Se2State)> {
        let mut sampler = UniformSe2::new([-4.0, -4.0], [4.0, 4.0]);
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(seed);
        let mut pairs: Vec<_> = (0..count)
            .map(|_| (sampler.sample(&mut rng), sampler.sample(&mut rng)))
            .collect();
        let a = Se2State::new(0.5, -0.5, 1.0);
        pairs.push((a, a));
        pairs.push((a, Se2State::new(0.5, -0.5, 1.0 + 1e-9)));
        pairs.push((a, Se2State::new(0.5 + 1e-9, -0.5, 1.0)));

        pairs
    }

    #[test]
    fn walking_a_quarter_circle_turns_and_moves() {
        let segment = [Segment {
            kind: SegmentKind::Left,
            length: core::f64::consts::FRAC_PI_2,
        }];
        let end = walk(&Se2State::new(1.0, 1.0, 0.0), &segment, 10.0, 2.0);

        assert_same_pose(
            &end,
            &Se2State::new(3.0, 3.0, core::f64::consts::FRAC_PI_2),
            1e-12,
        );
        assert_eq!(total_length(&segment), core::f64::consts::FRAC_PI_2);
    }
}
