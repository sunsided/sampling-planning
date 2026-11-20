//! The Euclidean space R^N.

use crate::StateSpace;

/// The Euclidean space R^N with states `[f64; N]` and straight-line interpolation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Euclidean<const N: usize>;

impl<const N: usize> StateSpace for Euclidean<N> {
    type State = [f64; N];

    fn dimension(&self) -> usize {
        N
    }

    fn distance(&self, from: &[f64; N], to: &[f64; N]) -> f64 {
        let squared: f64 = from.iter().zip(to).map(|(a, b)| (b - a) * (b - a)).sum();

        libm::sqrt(squared)
    }

    fn interpolate(&self, from: &[f64; N], to: &[f64; N], t: f64) -> [f64; N] {
        core::array::from_fn(|i| from[i] + (to[i] - from[i]) * t)
    }

    fn is_euclidean(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_is_euclidean() {
        assert!((Euclidean::<2>.distance(&[0.0, 0.0], &[3.0, 4.0]) - 5.0).abs() < 1e-12);
    }

    #[test]
    fn interpolate_hits_endpoints_and_midpoint() {
        let (a, b) = ([0.0, 2.0], [4.0, 6.0]);

        assert_eq!(Euclidean::<2>.interpolate(&a, &b, 0.0), a);
        assert_eq!(Euclidean::<2>.interpolate(&a, &b, 1.0), b);
        assert_eq!(Euclidean::<2>.interpolate(&a, &b, 0.5), [2.0, 4.0]);
    }
}
