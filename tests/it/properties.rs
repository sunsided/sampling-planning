//! Property tests for the angle helpers and the SO(2), SE(2), SE(3) and car-like metrics.
//!
//! The planner tests run on a few fixed worlds; these cover the numeric building blocks over
//! their whole documented domain instead. `task test:prop` raises the case count.

use core::f64::consts::{PI, TAU};

use proptest::prelude::*;
use sampling_planning::{
    Dubins, LinearScan, NearestNeighbors, ReedsShepp, RemovableNeighbors, Se2, Se2State, Se3,
    Se3State, So2, StateSpace, VpTree, angle_difference, wrap_angle,
};

/// Any angle within a thousand turns; wrapping must stay exact enough there.
fn angle() -> impl Strategy<Value = f64> {
    -1000.0 * TAU..1000.0 * TAU
}

/// A pose in `[-50, 50]^2` with an unwrapped heading.
fn pose() -> impl Strategy<Value = Se2State> {
    (-50.0..50.0, -50.0..50.0, angle()).prop_map(|(x, y, theta)| Se2State::new(x, y, theta))
}

/// A spatial pose in `[-50, 50]^3` with an arbitrary (normalized) quaternion.
fn spatial_pose() -> impl Strategy<Value = Se3State> {
    (
        [-50.0..50.0, -50.0..50.0, -50.0..50.0_f64],
        [-1.0..1.0, -1.0..1.0, -1.0..1.0, -1.0..1.0_f64],
    )
        .prop_filter("non-zero quaternion", |(_, q)| {
            q.iter().map(|c| c * c).sum::<f64>() > 1e-6
        })
        .prop_map(|(position, rotation)| Se3State::new(position, rotation))
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
    fn se3_distance_is_a_metric_and_interpolation_its_geodesic(
        a in spatial_pose(),
        b in spatial_pose(),
        c in spatial_pose(),
        weight in 0.01..10.0_f64,
        t in 0.0..1.0_f64,
    ) {
        let space = Se3::new(weight);
        let ab = space.distance(&a, &b);
        let x = space.interpolate(&a, &b, t);
        let flipped = Se3State { rotation: b.rotation.map(|q| -q), ..b };

        prop_assert!(ab >= 0.0);
        prop_assert!((ab - space.distance(&b, &a)).abs() < 1e-9);
        prop_assert!((ab - space.distance(&a, &flipped)).abs() < 1e-9);
        prop_assert!(space.distance(&a, &c) <= ab + space.distance(&b, &c) + 1e-9);
        prop_assert!((space.distance(&a, &x) - t * ab).abs() < 1e-7 * (1.0 + ab));
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

    #[test]
    fn vp_tree_answers_exactly_like_a_linear_scan(
        states in prop::collection::vec(spatial_pose(), 1..300),
        removals in prop::collection::vec(0..300_usize, 0..60),
        queries in prop::collection::vec(spatial_pose(), 1..8),
        radius in 0.0..60.0_f64,
        k in 0..12_usize,
        weight in 0.01..10.0_f64,
    ) {
        assert_vp_tree_matches_linear_scan(&Se3::new(weight), &states, &removals, &queries, radius, k)?;
    }

    #[test]
    fn vp_tree_answers_exactly_like_a_linear_scan_over_se2_and_reeds_shepp(
        states in prop::collection::vec(pose(), 1..200),
        removals in prop::collection::vec(0..200_usize, 0..40),
        queries in prop::collection::vec(pose(), 1..6),
        radius in 0.0..40.0_f64,
        k in 0..10_usize,
        turning in 0.1..10.0_f64,
    ) {
        assert_vp_tree_matches_linear_scan(&Se2::new(turning), &states, &removals, &queries, radius, k)?;
        assert_vp_tree_matches_linear_scan(
            &ReedsShepp::new(turning), &states, &removals, &queries, radius, k,
        )?;
    }
}

/// Fills a [`VpTree`] and a [`LinearScan`] with `states` (every fourth id repeating an earlier
/// one, so removal order matters), removes `removals` from both, and checks that all three
/// queries agree for every query state.
fn assert_vp_tree_matches_linear_scan<Sp>(
    space: &Sp,
    states: &[Sp::State],
    removals: &[usize],
    queries: &[Sp::State],
    radius: f64,
    k: usize,
) -> Result<(), TestCaseError>
where
    Sp: StateSpace + Clone,
    Sp::State: Clone,
{
    let mut tree = VpTree::new(space.clone());
    let mut linear = LinearScan::default();

    for (id, state) in states.iter().enumerate() {
        let id = if id % 4 == 3 { id / 2 } else { id };
        tree.insert(id, state.clone());
        NearestNeighbors::<Sp>::insert(&mut linear, id, state.clone());
    }

    for &id in removals {
        prop_assert_eq!(
            tree.remove(id),
            RemovableNeighbors::<Sp>::remove(&mut linear, id)
        );
    }

    prop_assert_eq!(tree.len(), NearestNeighbors::<Sp>::len(&linear));
    let (mut a, mut b) = (Vec::new(), Vec::new());
    for query in queries {
        prop_assert_eq!(tree.nearest(space, query), linear.nearest(space, query));
        tree.within_radius(space, query, radius, &mut a);
        linear.within_radius(space, query, radius, &mut b);
        prop_assert_eq!(&a, &b);
        tree.k_nearest(space, query, k, &mut a);
        linear.k_nearest(space, query, k, &mut b);
        prop_assert_eq!(&a, &b);
    }

    Ok(())
}
