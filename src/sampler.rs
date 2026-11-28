//! Samplers: where planners draw new states from.

mod informed;
mod loewner;
mod uniform_box;

pub use informed::ProlateHyperspheroid;
pub use loewner::LoewnerBound;
pub(crate) use loewner::{cholesky, is_symmetric};
use rand_core::Rng;
pub use uniform_box::{INFORMED_BOX_ATTEMPTS, UniformBox};

use crate::StateSpace;

/// Draws states for a planner.
///
/// `&mut self` lets stateful samplers (low-discrepancy sequences, informed samplers) advance.
/// The RNG is borrowed, never owned, so one seeded RNG drives a whole planning run.
pub trait Sampler<S> {
    /// Draws one state.
    fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> S;

    /// Draws a state directly from the informed set
    /// `{x : d(start, x) + d(x, goal) <= max_cost}` within this sampler's support, or returns
    /// `None` to make the planner fall back to rejection sampling from [`Sampler::sample`].
    ///
    /// Informed planners ([`RrtStar`](crate::trees::RrtStar) with
    /// [`informed`](crate::trees::RrtStarConfig::informed), BIT*) call this once a solution
    /// exists. A returned state must lie in the informed set under `space`'s metric and be
    /// distributed uniformly over the informed set intersected with the support of
    /// [`Sampler::sample`]; otherwise the planner loses its optimality guarantees.
    ///
    /// When direct sampling does not apply (an unsupported space, or a set so large that
    /// rejection is cheaper), return `None` **without** drawing from `rng`, so that the
    /// fallback sees the same random stream as without this method. `None` after drawing is
    /// allowed when an attempt budget runs out.
    ///
    /// The default returns `None`: rejection sampling, which works for every sampler and space.
    /// [`UniformBox`] samples a [`ProlateHyperspheroid`] in Euclidean spaces, and in spaces
    /// whose metric is bounded below by a [`LoewnerBound`].
    fn sample_informed<Sp, R>(
        &mut self,
        space: &Sp,
        rng: &mut R,
        start: &S,
        goal: &S,
        max_cost: f64,
    ) -> Option<S>
    where
        Sp: StateSpace<State = S>,
        R: Rng + ?Sized,
    {
        let _ = (space, rng, start, goal, max_cost);

        None
    }

    /// The measure of the informed set `{x : d(start, x) + d(x, goal) <= max_cost}` within
    /// this sampler's support, as a fraction in `(0, 1]` of the support's measure, or `None`
    /// when unknown.
    ///
    /// [`AitStar`](crate::informed::AitStar) and [`EitStar`](crate::informed::EitStar) scale
    /// their connection radius with it, as OMPL does with the informed measure: once a
    /// solution shrinks the informed set, the same number of samples is denser there, so a
    /// smaller radius keeps the graph's degree near `log q` instead of growing toward a
    /// complete graph.
    ///
    /// The default returns `None`, which keeps the radius of the whole support. [`UniformBox`]
    /// reports the volume of the [`ProlateHyperspheroid`] (an upper bound on its part inside
    /// the box) in Euclidean spaces and under a [`LoewnerBound`].
    fn informed_fraction<Sp>(&self, space: &Sp, start: &S, goal: &S, max_cost: f64) -> Option<f64>
    where
        Sp: StateSpace<State = S>,
    {
        let _ = (space, start, goal, max_cost);

        None
    }
}

/// Draws a uniform `f64` in `[0, 1)` from one `next_u64` call.
///
/// Uses the top 52 bits as the mantissa of a float in `[1, 2)` and subtracts one, so the
/// result is bit-for-bit reproducible on every platform.
pub fn unit_f64<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    /// Bit pattern of `1.0_f64`: zero sign, biased exponent 1023, empty mantissa.
    const ONE: u64 = 0x3FF0_0000_0000_0000;

    f64::from_bits(ONE | (rng.next_u64() >> 12)) - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::SeedableRng;

    #[test]
    fn unit_f64_stays_in_half_open_unit_interval() {
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(1);

        for _ in 0..10_000 {
            let x = unit_f64(&mut rng);
            assert!((0.0..1.0).contains(&x));
        }
    }
}
