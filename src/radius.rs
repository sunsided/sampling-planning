//! Connection radii of random geometric graphs, shared by RRT*, PRM* and FMT*.

use core::f64::consts::PI;

/// Converts a count to `f64`, saturating at `u32::MAX` (far beyond any graph size).
pub(crate) fn count_to_f64(count: usize) -> f64 {
    f64::from(u32::try_from(count).unwrap_or(u32::MAX))
}

/// The volume `zeta_d` of the unit ball in `d` dimensions: `pi^(d/2) / Gamma(d/2 + 1)`.
pub(crate) fn unit_ball_volume(d: f64) -> f64 {
    libm::pow(PI, d / 2.0) / libm::tgamma(d / 2.0 + 1.0)
}

/// The dimension of a space as `f64`, at least one so that `1 / d` stays finite.
pub(crate) fn dimension_f64(dimension: usize) -> f64 {
    count_to_f64(dimension.max(1))
}

/// The shrinking radius `gamma * (ln n / n)^(1/d)` for `n` samples in `d` dimensions.
///
/// Zero for `n <= 1`, where `ln n` is zero or undefined.
pub(crate) fn shrinking_radius(gamma: f64, n: usize, d: f64) -> f64 {
    if n <= 1 {
        return 0.0;
    }

    let n = count_to_f64(n);

    gamma * libm::pow(libm::log(n) / n, 1.0 / d)
}

/// The smallest k-nearest constant for which k-nearest PRM* and RRG are asymptotically optimal
/// (Karaman and Frazzoli 2011, Theorems 35 and 37): `e (1 + 1/d)`.
pub(crate) fn k_constant_lower_bound(dimension: usize) -> f64 {
    core::f64::consts::E * (1.0 + 1.0 / dimension_f64(dimension))
}

/// The neighbor count `ceil(k_constant * ln n)` for a graph of `n` nodes, at least one.
pub(crate) fn k_nearest_count(k_constant: f64, n: usize) -> usize {
    if n <= 1 {
        return 1;
    }

    let k = libm::ceil(k_constant * libm::log(count_to_f64(n)));

    if k >= f64::from(u32::MAX) {
        usize::MAX
    } else if k >= 1.0 {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "k is a whole number in [1, u32::MAX) after the guards above"
        )]
        let k = k as u32;
        usize::try_from(k).unwrap_or(usize::MAX)
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_ball_volumes_match_closed_forms() {
        assert!((unit_ball_volume(1.0) - 2.0).abs() < 1e-12);
        assert!((unit_ball_volume(2.0) - PI).abs() < 1e-12);
        assert!((unit_ball_volume(3.0) - 4.0 * PI / 3.0).abs() < 1e-12);
    }

    #[test]
    fn k_nearest_count_grows_logarithmically() {
        assert_eq!(k_nearest_count(5.0, 1), 1);
        assert_eq!(k_nearest_count(5.0, 100), 24);
        assert_eq!(k_nearest_count(f64::NAN, 100), 1);
        assert!((k_constant_lower_bound(2) - 1.5 * core::f64::consts::E).abs() < 1e-12);
    }

    #[test]
    fn shrinking_radius_is_zero_for_tiny_graphs_and_shrinks() {
        assert!(shrinking_radius(3.0, 1, 2.0).abs() < f64::EPSILON);
        assert!(shrinking_radius(3.0, 100, 2.0) > shrinking_radius(3.0, 1000, 2.0));
    }
}
