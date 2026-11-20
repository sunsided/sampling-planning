//! Uniform sampling in an axis-aligned box.

use super::{ProlateHyperspheroid, Sampler, unit_f64};
use crate::StateSpace;
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

    /// The volume of the box.
    #[must_use]
    pub fn volume(&self) -> f64 {
        self.lower
            .iter()
            .zip(&self.upper)
            .map(|(lo, hi)| hi - lo)
            .product()
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

    /// Samples the [`ProlateHyperspheroid`] with foci `start` and `goal` and transverse diameter
    /// `max_cost`, rejecting samples outside the box, which is uniform over the intersection.
    ///
    /// Returns `None` without drawing when `space` is not
    /// [Euclidean](StateSpace::is_euclidean), the set is degenerate, or its volume is at least
    /// the box volume (then rejection from the box accepts more often). Returns `None` after
    /// [`INFORMED_BOX_ATTEMPTS`] samples outside the box, which happens when the set reaches
    /// far out of the box.
    fn sample_informed<Sp, R>(
        &mut self,
        space: &Sp,
        rng: &mut R,
        start: &[f64; N],
        goal: &[f64; N],
        max_cost: f64,
    ) -> Option<[f64; N]>
    where
        Sp: StateSpace<State = [f64; N]>,
        R: Rng + ?Sized,
    {
        if !space.is_euclidean() {
            return None;
        }

        let set = ProlateHyperspheroid::new(*start, *goal, max_cost)?;
        if set.volume() >= self.volume() {
            return None;
        }

        (0..INFORMED_BOX_ATTEMPTS)
            .map(|_| set.sample(rng))
            .find(|state| self.contains(state))
    }
}

/// The most hyperspheroid samples [`UniformBox`]'s direct informed sampling draws before it
/// gives up because they keep falling outside the box.
pub const INFORMED_BOX_ATTEMPTS: usize = 64;

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
    fn informed_samples_stay_in_the_box_and_the_set() {
        let mut sampler = UniformBox::new([0.0, 0.0], [10.0, 4.0]);
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(5);
        let (start, goal) = ([1.0, 1.0], [9.0, 1.0]);
        let space = crate::Euclidean::<2>;

        for _ in 0..1000 {
            let state = sampler
                .sample_informed(&space, &mut rng, &start, &goal, 9.0)
                .expect("a thin ellipse is sampled directly");
            assert!(sampler.contains(&state));
            assert!(space.distance(&start, &state) + space.distance(&state, &goal) <= 9.0 + 1e-9);
        }
    }

    #[test]
    fn informed_sampling_declines_without_drawing() {
        /// An RNG that panics when used.
        struct NoDraw;

        impl rand_core::TryRng for NoDraw {
            type Error = core::convert::Infallible;

            fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
                unreachable!("declining must not draw")
            }

            fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
                unreachable!("declining must not draw")
            }

            fn try_fill_bytes(&mut self, _: &mut [u8]) -> Result<(), Self::Error> {
                unreachable!("declining must not draw")
            }
        }

        let mut sampler = UniformBox::new([0.0, 0.0], [10.0, 10.0]);
        let (start, goal) = ([1.0, 1.0], [9.0, 9.0]);
        let euclidean = crate::Euclidean::<2>;

        // Larger than the box, degenerate, and a space that is not Euclidean.
        assert!(
            sampler
                .sample_informed(&euclidean, &mut NoDraw, &start, &goal, 40.0)
                .is_none()
        );
        assert!(
            sampler
                .sample_informed(&euclidean, &mut NoDraw, &start, &goal, 1.0)
                .is_none()
        );
        assert!(
            sampler
                .sample_informed(&Taxicab, &mut NoDraw, &start, &goal, 12.0)
                .is_none()
        );
    }

    /// The L1 metric on R^2, a space that is not Euclidean.
    struct Taxicab;

    impl StateSpace for Taxicab {
        type State = [f64; 2];

        fn dimension(&self) -> usize {
            2
        }

        fn distance(&self, a: &[f64; 2], b: &[f64; 2]) -> f64 {
            libm::fabs(a[0] - b[0]) + libm::fabs(a[1] - b[1])
        }

        fn interpolate(&self, a: &[f64; 2], b: &[f64; 2], t: f64) -> [f64; 2] {
            crate::Euclidean::<2>.interpolate(a, b, t)
        }
    }

    #[test]
    #[should_panic(expected = "lower bound exceeds")]
    fn rejects_inverted_box() {
        let _ = UniformBox::new([1.0], [0.0]);
    }
}
