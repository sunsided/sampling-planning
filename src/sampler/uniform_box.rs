//! Uniform sampling in an axis-aligned box.

use super::{Sampler, unit_f64};
use rand_core::Rng;

/// Samples uniformly from the axis-aligned box `[lower, upper)` in R^N.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UniformBox<const N: usize> {
    /// Inclusive lower corner.
    lower: [f64; N],
    /// Exclusive upper corner; at least `lower` on every axis.
    upper: [f64; N],
}

impl<const N: usize> UniformBox<N> {
    /// Creates a sampler for the box spanned by `lower` and `upper`.
    ///
    /// # Panics
    ///
    /// Panics if any bound is not finite or `lower > upper` on any axis.
    #[must_use]
    pub fn new(lower: [f64; N], upper: [f64; N]) -> Self {
        for (lo, hi) in lower.iter().zip(&upper) {
            assert!(
                lo.is_finite() && hi.is_finite(),
                "box bounds must be finite"
            );
            assert!(lo <= hi, "box lower bound exceeds upper bound");
        }

        Self { lower, upper }
    }

    /// The inclusive lower corner.
    #[must_use]
    pub const fn lower(&self) -> &[f64; N] {
        &self.lower
    }

    /// The upper corner.
    #[must_use]
    pub const fn upper(&self) -> &[f64; N] {
        &self.upper
    }

    /// Whether `state` lies inside the closed box.
    #[must_use]
    pub fn contains(&self, state: &[f64; N]) -> bool {
        (0..N).all(|i| (self.lower[i]..=self.upper[i]).contains(&state[i]))
    }
}

impl<const N: usize> Sampler<[f64; N]> for UniformBox<N> {
    fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> [f64; N] {
        core::array::from_fn(|i| self.lower[i] + (self.upper[i] - self.lower[i]) * unit_f64(rng))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::SeedableRng;

    #[test]
    fn samples_stay_in_box() {
        let mut sampler = UniformBox::new([-1.0, 2.0, 0.0], [1.0, 5.0, 0.0]);
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(3);

        for _ in 0..1000 {
            let state = sampler.sample(&mut rng);
            assert!(sampler.contains(&state));
        }
    }

    #[test]
    #[should_panic(expected = "lower bound exceeds")]
    fn rejects_inverted_box() {
        let _ = UniformBox::new([1.0], [0.0]);
    }
}
