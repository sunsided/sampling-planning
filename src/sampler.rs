//! Samplers: where planners draw new states from.

mod uniform_box;

use rand_core::Rng;
pub use uniform_box::UniformBox;

/// Draws states for a planner.
///
/// `&mut self` lets stateful samplers (low-discrepancy sequences, informed samplers) advance.
/// The RNG is borrowed, never owned, so one seeded RNG drives a whole planning run.
pub trait Sampler<S> {
    /// Draws one state.
    fn sample<R: Rng + ?Sized>(&mut self, rng: &mut R) -> S;
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
