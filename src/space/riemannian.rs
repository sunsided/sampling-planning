//! Coordinate spaces with a configuration-dependent Riemannian metric.

use core::fmt;
use core::num::NonZeroUsize;

use crate::StateSpace;
use crate::radius::count_to_f64;

/// R^N with a configuration-dependent metric tensor `G(q)`: the length of a motion is
/// `integral of sqrt(dq^T G(q) dq)` along it.
///
/// Motions are straight lines in coordinates, as in [`Euclidean`](crate::Euclidean), and
/// [`StateSpace::distance`] is their length by the midpoint rule over `segments` equal pieces:
/// `sum_k sqrt(d^T G(m_k) d)` with `d = (to - from) / segments` and `m_k` the midpoint of
/// piece `k` (Kyaw and Kelly use the same approximation). Joint-space planning under a
/// kinetic-energy metric (the mass matrix) or a task-space pullback metric looks like this.
///
/// `G` must return a symmetric positive definite matrix for every state the planner visits.
/// The distance is zero only for equal states and symmetric up to rounding. It is the length
/// of the straight motion, not of the geodesic, so it can violate the triangle inequality
/// where `G` varies strongly: planners that rely on a metric (see
/// [`StateSpace::distance`]) then lose their optimality guarantee, but not their soundness,
/// since every edge is still checked and costed along the straight motion. More `segments`
/// approximate the integral better at the price of more metric evaluations per distance.
///
/// Informed planners sample its informed sets directly through
/// [`UniformBox::with_metric_bound`](crate::UniformBox::with_metric_bound) with a
/// [`LoewnerBound`](crate::LoewnerBound) of `G`: wherever the bound lies below `G`, every
/// piece is at least `||L^T d||`, so the distance is at least the bound's distance.
///
/// # Examples
///
/// ```
/// use core::num::NonZeroUsize;
/// use sampling_planning::{Riemannian, StateSpace};
///
/// // Motion along x costs four times as much as along y.
/// let space = Riemannian::new(|_: &[f64; 2]| [[16.0, 0.0], [0.0, 1.0]], NonZeroUsize::MIN);
///
/// assert!((space.distance(&[0.0, 0.0], &[1.0, 0.0]) - 4.0).abs() < 1e-12);
/// assert!((space.distance(&[0.0, 0.0], &[0.0, 1.0]) - 1.0).abs() < 1e-12);
/// ```
#[derive(Clone, Copy)]
pub struct Riemannian<const N: usize, F> {
    /// The metric tensor at a state; symmetric positive definite.
    metric: F,
    /// The number of midpoint-rule pieces per distance; at least one.
    segments: usize,
}

impl<const N: usize, F> Riemannian<N, F>
where
    F: Fn(&[f64; N]) -> [[f64; N]; N],
{
    /// The space with metric tensor `metric`, measuring distances over `segments` pieces.
    #[must_use]
    pub const fn new(metric: F, segments: NonZeroUsize) -> Self {
        Self {
            metric,
            segments: segments.get(),
        }
    }

    /// The metric tensor `G(q)` at `state`.
    #[must_use]
    pub fn metric_at(&self, state: &[f64; N]) -> [[f64; N]; N] {
        (self.metric)(state)
    }

    /// The number of midpoint-rule pieces per distance.
    #[must_use]
    pub const fn segments(&self) -> usize {
        self.segments
    }
}

impl<const N: usize, F> fmt::Debug for Riemannian<N, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Riemannian")
            .field("segments", &self.segments)
            .finish_non_exhaustive()
    }
}

impl<const N: usize, F> StateSpace for Riemannian<N, F>
where
    F: Fn(&[f64; N]) -> [[f64; N]; N],
{
    type State = [f64; N];

    fn dimension(&self) -> usize {
        N
    }

    fn distance(&self, from: &[f64; N], to: &[f64; N]) -> f64 {
        let pieces = count_to_f64(self.segments);
        let step: [f64; N] = core::array::from_fn(|i| (to[i] - from[i]) / pieces);

        (0..self.segments)
            .map(|k| {
                let t = (count_to_f64(k) + 0.5) / pieces;
                let midpoint = self.interpolate(from, to, t);
                let g = (self.metric)(&midpoint);
                let quadratic: f64 = (0..N)
                    .flat_map(|i| (0..N).map(move |j| (i, j)))
                    .map(|(i, j)| step[i] * g[i][j] * step[j])
                    .sum();

                libm::sqrt(quadratic.max(0.0))
            })
            .sum()
    }

    fn interpolate(&self, from: &[f64; N], to: &[f64; N], t: f64) -> [f64; N] {
        core::array::from_fn(|i| from[i] + (to[i] - from[i]) * t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LoewnerBound;

    /// A metric that grows quadratically away from the origin: `A + |q|^2 B`.
    fn growing(q: &[f64; 2]) -> [[f64; 2]; 2] {
        let r = q[0] * q[0] + q[1] * q[1];

        [[2.0 + r, 0.5 + 0.2 * r], [0.5 + 0.2 * r, 1.0 + 0.5 * r]]
    }

    #[test]
    fn distance_integrates_the_metric_and_converges() {
        let coarse = Riemannian::new(growing, NonZeroUsize::MIN);
        let fine = Riemannian::new(growing, NonZeroUsize::new(64).expect("non-zero"));
        let finer = Riemannian::new(growing, NonZeroUsize::new(256).expect("non-zero"));
        let (a, b) = ([-2.0, 1.0], [3.0, 0.5]);

        assert!((fine.distance(&a, &b) - finer.distance(&a, &b)).abs() < 1e-3);
        assert!((fine.distance(&a, &b) - fine.distance(&b, &a)).abs() < 1e-9);
        assert!(fine.distance(&a, &a).abs() < f64::EPSILON);
        assert!(coarse.distance(&a, &b) > 0.0);
    }

    #[test]
    fn the_bound_at_the_minimum_is_admissible() {
        // G(q) - G(0) = |q|^2 B with B positive definite, so G(0) lies below G everywhere.
        let space = Riemannian::new(growing, NonZeroUsize::new(8).expect("non-zero"));
        let bound = LoewnerBound::new(growing(&[0.0, 0.0])).expect("positive definite");
        let points = [
            [-3.0, 2.0],
            [0.5, -1.0],
            [4.0, 4.0],
            [0.0, 0.1],
            [-1.0, -3.5],
        ];

        for a in &points {
            for b in &points {
                assert!(bound.distance(a, b) <= space.distance(a, b) + 1e-12);
            }
        }
    }
}
