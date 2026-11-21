//! Property tests for the angle helpers and the SO(2), SE(2) and car-like metrics.
//!
//! The planner tests run on a few fixed worlds; these cover the numeric building blocks over
//! their whole documented domain instead. `task test:prop` raises the case count.

use core::f64::consts::{PI, TAU};

use proptest::prelude::*;
use sampling_planning::{
    Dubins, ReedsShepp, Se2, Se2State, So2, StateSpace, angle_difference, wrap_angle,
};

/// Any angle within a thousand turns; wrapping must stay exact enough there.
fn angle() -> impl Strategy<Value = f64> {
    -1000.0 * TAU..1000.0 * TAU
}

/// A pose in `[-50, 50]^2` with an unwrapped heading.
fn pose() -> impl Strategy<Value = Se2State> {
    (-50.0..50.0, -50.0..50.0, angle()).prop_map(|(x, y, theta)| Se2State::new(x, y, theta))
}

/// Whether `a` and `b` are the same orientation up to `tolerance` radians.
fn same_orientation(a: f64, b: f64, tolerance: f64) -> bool {
    angle_difference(a, b).abs() <= tolerance
}

proptest! {
    #[test]
    fn wrap_angle_lands_in_the_half_open_range_and_keeps_the_orientation(angle in angle()) {
        let wrapped = wrap_angle(angle);

        prop_assert!((-PI..PI).contains(&wrapped), "{angle} wrapped to {wrapped}");
        prop_assert!(same_orientation(wrapped, angle, 1e-9));
        prop_assert_eq!(wrap_angle(wrapped).to_bits(), wrapped.to_bits(), "wrapping is idempotent");
    }

    #[test]
    fn angle_difference_is_the_shortest_signed_turn(from in angle(), to in angle()) {
        let turn = angle_difference(from, to);

        prop_assert!((-PI..PI).contains(&turn));
        prop_assert!(same_orientation(from + turn, to, 1e-9));
        prop_assert!((So2.distance(&from, &to) - turn.abs()).abs() < 1e-9);
    }

    #[test]
    fn se2_distance_is_a_symmetric_metric(
        a in pose(),
        b in pose(),
        c in pose(),
        weight in 0.01..10.0_f64,
    ) {
        let space = Se2::new(weight);
        let ab = space.distance(&a, &b);

        prop_assert!(ab >= 0.0);
        prop_assert!((ab - space.distance(&b, &a)).abs() < 1e-9);
        prop_assert!(space.distance(&a, &c) <= ab + space.distance(&b, &c) + 1e-9);
    }

    #[test]
    fn car_distances_are_bounded_by_the_straight_line_and_each_other(
        from in pose(),
        to in pose(),
        radius in 0.1..10.0_f64,
    ) {
        let forward = Dubins::new(radius).distance(&from, &to);
        let reversing = ReedsShepp::new(radius).distance(&from, &to);
        let straight = (to.x - from.x).hypot(to.y - from.y);
        let tolerance = 1e-7 * (1.0 + forward);

        prop_assert!(reversing >= straight - tolerance, "{reversing} < {straight}");
        prop_assert!(reversing <= forward + tolerance, "{reversing} > {forward}");
    }
}
