//! Multi-threaded batch validity checks (feature `parallel`).

use alloc::vec::Vec;
use core::num::NonZeroUsize;

use crate::{StateSpace, Validity, ValidityChecker};

/// Wraps a [`ValidityChecker`] and spreads its batch checks over threads.
///
/// [`ValidityChecker::check_states`] and [`ValidityChecker::check_motions`] split the batch
/// into one contiguous chunk per thread on [`std::thread::scope`], hand each chunk to the
/// wrapped checker's own batch method (so a SIMD checker stays vectorized) and concatenate the
/// results in order, so planners get exactly the results of the wrapped checker. Single checks run on
/// the calling thread. Batches smaller than `min_batch` run on the calling thread too, since
/// spawning threads costs some microseconds: parallel checks pay off for expensive checkers
/// (meshes, robot models) and batches of tens of checks or more.
///
/// # Examples
///
/// ```
/// use core::num::NonZeroUsize;
/// use sampling_planning::{Euclidean, ParallelChecker, Validity, ValidityChecker};
///
/// /// Everything left of x = 5 is free.
/// struct HalfPlane;
///
/// impl ValidityChecker<Euclidean<1>> for HalfPlane {
///     fn check_state(&self, s: &[f64; 1]) -> Validity {
///         if s[0] < 5.0 { Validity::Valid } else { Validity::Invalid }
///     }
///
///     fn check_motion(&self, _: &Euclidean<1>, _: &[f64; 1], to: &[f64; 1]) -> Validity {
///         self.check_state(to)
///     }
/// }
///
/// let checker = ParallelChecker::new(HalfPlane, NonZeroUsize::new(4).expect("non-zero"), 1);
/// let states: Vec<[f64; 1]> = (0..10).map(|i| [f64::from(i)]).collect();
/// let mut out = Vec::new();
/// checker.check_states(&states, &mut out);
///
/// assert_eq!(out.iter().filter(|v| v.is_valid()).count(), 5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelChecker<C> {
    /// The wrapped checker.
    inner: C,
    /// The number of threads per batch, the calling thread included.
    threads: NonZeroUsize,
    /// The smallest batch that is spread over threads.
    min_batch: usize,
}

impl<C> ParallelChecker<C> {
    /// Wraps `inner`, spreading batches of at least `min_batch` checks over `threads` threads.
    #[must_use]
    pub const fn new(inner: C, threads: NonZeroUsize, min_batch: usize) -> Self {
        Self {
            inner,
            threads,
            min_batch,
        }
    }

    /// The wrapped checker.
    #[must_use]
    pub const fn inner(&self) -> &C {
        &self.inner
    }

    /// The chunk length for a batch of `len` checks, or `None` to check it on the calling
    /// thread.
    fn chunk_len(&self, len: usize) -> Option<usize> {
        let threads = self.threads.get();

        (threads > 1 && len >= self.min_batch.max(2)).then(|| len.div_ceil(threads))
    }
}

/// Runs `check_chunk` over `items` in chunks of `chunk` on scoped threads (the first chunk on
/// the calling thread) and appends the results to `out` in order.
fn scatter<T, F>(items: &[T], chunk: usize, out: &mut Vec<Validity>, check_chunk: F)
where
    T: Sync,
    F: Fn(&[T], &mut Vec<Validity>) + Sync,
{
    let check_chunk = &check_chunk;
    let mut chunks = items.chunks(chunk);
    let first = chunks.next().unwrap_or(&[]);

    std::thread::scope(|scope| {
        let handles: Vec<_> = chunks
            .map(|part| {
                scope.spawn(move || {
                    let mut results = Vec::new();
                    check_chunk(part, &mut results);
                    results
                })
            })
            .collect();

        check_chunk(first, out);

        for handle in handles {
            match handle.join() {
                Ok(results) => out.extend(results),
                Err(panic) => std::panic::resume_unwind(panic),
            }
        }
    });
}

impl<Sp, C> ValidityChecker<Sp> for ParallelChecker<C>
where
    Sp: StateSpace + Sync,
    Sp::State: Sync,
    C: ValidityChecker<Sp> + Sync,
{
    fn check_state(&self, state: &Sp::State) -> Validity {
        self.inner.check_state(state)
    }

    fn check_motion(&self, space: &Sp, from: &Sp::State, to: &Sp::State) -> Validity {
        self.inner.check_motion(space, from, to)
    }

    fn check_states(&self, states: &[Sp::State], out: &mut Vec<Validity>) {
        let Some(chunk) = self.chunk_len(states.len()) else {
            return self.inner.check_states(states, out);
        };

        out.clear();
        scatter(states, chunk, out, |part, results| {
            self.inner.check_states(part, results);
        });
    }

    fn check_motions(
        &self,
        space: &Sp,
        motions: &[(&Sp::State, &Sp::State)],
        out: &mut Vec<Validity>,
    ) {
        let Some(chunk) = self.chunk_len(motions.len()) else {
            return self.inner.check_motions(space, motions, out);
        };

        out.clear();
        scatter(motions, chunk, out, |part, results| {
            self.inner.check_motions(space, part, results);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Euclidean;

    /// Valid where the coordinate's integer part is even.
    struct Stripes;

    impl ValidityChecker<Euclidean<1>> for Stripes {
        fn check_state(&self, state: &[f64; 1]) -> Validity {
            if libm::floor(state[0]) % 2.0 == 0.0 {
                Validity::Valid
            } else {
                Validity::Invalid
            }
        }

        fn check_motion(&self, _: &Euclidean<1>, from: &[f64; 1], to: &[f64; 1]) -> Validity {
            self.check_state(from).and(self.check_state(to))
        }
    }

    #[test]
    fn parallel_batches_equal_the_sequential_results_in_order() {
        let states: Vec<[f64; 1]> = (0..103).map(|i| [0.37 * f64::from(i)]).collect();
        let motions: Vec<(&[f64; 1], &[f64; 1])> =
            states.windows(2).map(|pair| (&pair[0], &pair[1])).collect();
        let (mut sequential, mut parallel) = (Vec::new(), Vec::new());

        for threads in [1, 2, 3, 8, 200] {
            let checker =
                ParallelChecker::new(Stripes, NonZeroUsize::new(threads).expect("> 0"), 1);

            Stripes.check_states(&states, &mut sequential);
            checker.check_states(&states, &mut parallel);
            assert_eq!(sequential, parallel, "{threads} threads");

            Stripes.check_motions(&Euclidean::<1>, &motions, &mut sequential);
            checker.check_motions(&Euclidean::<1>, &motions, &mut parallel);
            assert_eq!(sequential, parallel, "{threads} threads");
        }
    }
}
