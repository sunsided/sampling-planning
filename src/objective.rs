//! Cost objectives: what a path costs, separate from the metric that finds neighbors.

use alloc::vec::Vec;
use core::f64::consts::PI;

use crate::StateSpace;
use crate::radius::count_to_f64;
use crate::sampler::{cholesky, is_symmetric};

/// The cost of a motion, summed over a path's edges.
///
/// Planners keep using [`StateSpace::distance`] to find neighbors and to shape informed sets;
/// the objective only prices motions. A cost that is a sum over edges is all that RRT*'s
/// choose-parent and rewiring need (and BIT*'s and FMT*'s searches would need), so whole-path
/// functionals are out of scope: integrals along the path, such as [`RiskDensity`], are sums
/// over edges.
///
/// Planners that take an objective: [`RrtStar`](crate::trees::RrtStar) through
/// [`RrtStar::with_objective`](crate::trees::RrtStar::with_objective). All others price paths
/// by [`PathLength`].
pub trait CostObjective<Sp: StateSpace> {
    /// The cost of the motion from `from` to `to` along [`StateSpace::interpolate`].
    ///
    /// Non-negative, and zero for equal states.
    fn motion_cost(&self, space: &Sp, from: &Sp::State, to: &Sp::State) -> f64;

    /// The largest `w >= 0` with `motion_cost(a, b) >= w * distance(a, b)` for every motion.
    ///
    /// Informed sampling needs it: a path of cost `c` through `x` then has
    /// `d(start, x) + d(x, goal) <= c / w`, so the informed set of the metric with bound
    /// `c / w` contains every state that can improve the solution. Zero, the default, switches
    /// informed sampling off, which is always admissible.
    fn distance_weight(&self) -> f64 {
        0.0
    }
}

/// The default objective: the motion cost is [`StateSpace::distance`], so a path costs its
/// length.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PathLength;

impl<Sp: StateSpace> CostObjective<Sp> for PathLength {
    fn motion_cost(&self, space: &Sp, from: &Sp::State, to: &Sp::State) -> f64 {
        space.distance(from, to)
    }

    fn distance_weight(&self) -> f64 {
        1.0
    }
}

/// A Gaussian uncertain obstacle of [`RiskDensity`]: its position is distributed as
/// `N(mean, covariance)`, where the covariance combines the obstacle's and the robot's
/// position uncertainty.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GaussianObstacle<const N: usize> {
    /// The mean position.
    mean: [f64; N],
    /// The inverse of the covariance matrix.
    precision: [[f64; N]; N],
    /// `scale / sqrt((2 pi)^N det(covariance))`: the density's normalizer times the scale.
    factor: f64,
}

impl<const N: usize> GaussianObstacle<N> {
    /// An obstacle at `mean` with covariance `covariance`, whose risk density along a path is
    /// multiplied by `scale`.
    ///
    /// For a disk robot and obstacle in the plane, Paiola et al.'s first-order collision
    /// probability of a path is its risk density times the combined radius `r`, and the risk
    /// density is twice the line integral of the Gaussian density. So `scale = 2 r` makes the
    /// cost of a path that collision probability, and `scale = 2 r xi` weighs it by a severity
    /// `xi`.
    ///
    /// Returns `None` unless `covariance` is finite, symmetric (up to `1e-9` relative) and
    /// positive definite and `scale` is non-negative and finite.
    #[must_use]
    pub fn new(mean: [f64; N], covariance: [[f64; N]; N], scale: f64) -> Option<Self> {
        if !(is_symmetric(&covariance) && scale.is_finite() && scale >= 0.0) {
            return None;
        }

        let factor_l = cholesky(&covariance)?;
        let determinant: f64 = (0..N).map(|i| factor_l[i][i] * factor_l[i][i]).product();
        let normalizer = libm::sqrt(libm::pow(2.0 * PI, count_to_f64(N)) * determinant);

        Some(Self {
            mean,
            precision: inverse_from_cholesky(&factor_l),
            factor: scale / normalizer,
        })
    }

    /// The scaled line integral of the obstacle's density along the straight segment from
    /// `from` to `to`, by arc length: `scale * integral_0^1 N(from + s (to - from)) |to - from| ds`
    /// in closed form (Paiola et al., eq. 26, in N dimensions).
    #[must_use]
    pub fn segment_risk(&self, from: &[f64; N], to: &[f64; N]) -> f64 {
        let offset: [f64; N] = core::array::from_fn(|i| from[i] - self.mean[i]);
        let delta: [f64; N] = core::array::from_fn(|i| to[i] - from[i]);
        let a = self.quadratic(&offset, &offset);
        let b = self.quadratic(&offset, &delta);
        let c = self.quadratic(&delta, &delta);
        let length = libm::sqrt(delta.iter().map(|d| d * d).sum());

        if c <= 0.0 || length <= 0.0 {
            return 0.0;
        }

        // integral_0^1 exp(-(a + 2 b s + c s^2) / 2) ds, completing the square in s.
        let root = libm::sqrt(0.5 * c);
        let integral = libm::exp(-0.5 * (a - b * b / c))
            * libm::sqrt(PI / (2.0 * c))
            * (libm::erf(root * (1.0 + b / c)) - libm::erf(root * (b / c)));

        self.factor * length * integral.max(0.0)
    }

    /// `u^T P v` with the precision matrix `P`.
    fn quadratic(&self, u: &[f64; N], v: &[f64; N]) -> f64 {
        (0..N)
            .flat_map(|i| (0..N).map(move |j| (i, j)))
            .map(|(i, j)| u[i] * self.precision[i][j] * v[j])
            .sum()
    }
}

/// The inverse of `L L^T` from its lower-triangular Cholesky factor `L`.
fn inverse_from_cholesky<const N: usize>(l: &[[f64; N]; N]) -> [[f64; N]; N] {
    // Columns of L^-1 by forward substitution, then (L L^T)^-1 = L^-T L^-1.
    let mut inverse_l = [[0.0; N]; N];

    for column in 0..N {
        for i in 0..N {
            let unit = if i == column { 1.0 } else { 0.0 };
            let head: f64 = (0..i).map(|k| l[i][k] * inverse_l[k][column]).sum();
            inverse_l[i][column] = (unit - head) / l[i][i];
        }
    }

    core::array::from_fn(|i| {
        core::array::from_fn(|j| (0..N).map(|k| inverse_l[k][i] * inverse_l[k][j]).sum())
    })
}

/// The hybrid risk-density objective of Paiola, Grioli and Bicchi ("Risk Density RRT*:
/// Planning in the space of probability", IJRR, arXiv 2611.22654): the risk of a motion
/// among uncertain obstacles plus `length_weight` times its length.
///
/// The risk of a straight motion is the sum over the [`GaussianObstacle`]s of their scaled
/// line integrals ([`GaussianObstacle::segment_risk`]), a first-order approximation of the
/// probability of colliding with them along the motion. Line integrals add up over edges, so
/// the objective decomposes as RRT*'s rewiring needs (their eqs. 27 and 28). The length term
/// (their eq. 30) keeps paths from making arbitrarily long detours around low-risk regions,
/// and is what makes informed sampling admissible: [`CostObjective::distance_weight`] is
/// `length_weight`, so a zero weight switches informed sampling off. Deterministic obstacles
/// stay with the [`ValidityChecker`](crate::ValidityChecker), as in the paper.
///
/// The risk is computed on the coordinates of `[f64; N]` states along straight motions, so
/// use it with spaces whose interpolation is the straight line, such as
/// [`Euclidean`](crate::Euclidean). The length term uses the space's distance. The motion cost
/// is not a metric (a straight edge through high risk can cost more than a detour), so
/// RRT*'s asymptotic optimality, which needs a metric cost, is not guaranteed; the paper
/// makes the same caveat.
///
/// # Examples
///
/// ```
/// use sampling_planning::{CostObjective, Euclidean, GaussianObstacle, RiskDensity};
///
/// let obstacle = GaussianObstacle::new([5.0, 0.0], [[0.25, 0.0], [0.0, 0.25]], 1.0)
///     .expect("positive definite");
/// let objective = RiskDensity::new(vec![obstacle], 0.01);
///
/// let through = objective.motion_cost(&Euclidean::<2>, &[0.0, 0.0], &[10.0, 0.0]);
/// let around = objective.motion_cost(&Euclidean::<2>, &[0.0, 3.0], &[10.0, 3.0]);
/// assert!(through > 5.0 * around);
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RiskDensity<const N: usize> {
    /// The uncertain obstacles.
    obstacles: Vec<GaussianObstacle<N>>,
    /// The weight `lambda` of the length term; non-negative and finite.
    length_weight: f64,
}

impl<const N: usize> RiskDensity<N> {
    /// The objective over `obstacles` with length weight `length_weight` (`lambda` in the
    /// paper, which uses `1e-8` in its benchmarks).
    ///
    /// # Panics
    ///
    /// Panics if `length_weight` is negative or not finite.
    #[must_use]
    pub fn new(obstacles: Vec<GaussianObstacle<N>>, length_weight: f64) -> Self {
        assert!(
            length_weight.is_finite() && length_weight >= 0.0,
            "length weight must be non-negative and finite"
        );

        Self {
            obstacles,
            length_weight,
        }
    }

    /// The risk of the straight motion from `from` to `to`, without the length term.
    #[must_use]
    pub fn risk(&self, from: &[f64; N], to: &[f64; N]) -> f64 {
        self.obstacles
            .iter()
            .map(|obstacle| obstacle.segment_risk(from, to))
            .sum()
    }
}

impl<const N: usize, Sp> CostObjective<Sp> for RiskDensity<N>
where
    Sp: StateSpace<State = [f64; N]>,
{
    fn motion_cost(&self, space: &Sp, from: &[f64; N], to: &[f64; N]) -> f64 {
        self.risk(from, to) + self.length_weight * space.distance(from, to)
    }

    fn distance_weight(&self) -> f64 {
        self.length_weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Euclidean;

    /// A unit-variance obstacle at the origin of the plane, unscaled.
    fn unit() -> GaussianObstacle<2> {
        GaussianObstacle::new([0.0, 0.0], [[1.0, 0.0], [0.0, 1.0]], 1.0).expect("valid")
    }

    #[test]
    fn segment_risk_matches_numerical_integration_and_adds_up() {
        let obstacle = GaussianObstacle::new([1.0, 0.5], [[0.5, 0.2], [0.2, 0.8]], 3.0)
            .expect("positive definite");
        let (from, to) = ([-2.0, -1.0], [3.0, 2.0]);
        let steps = 20_000;
        let length = Euclidean::<2>.distance(&from, &to);
        let numeric: f64 = (0..steps)
            .map(|k| {
                let t = (f64::from(k) + 0.5) / f64::from(steps);
                let x = Euclidean::<2>.interpolate(&from, &to, t);
                let d = [x[0] - 1.0, x[1] - 0.5];
                let exponent = obstacle.quadratic(&d, &d);

                obstacle.factor * libm::exp(-0.5 * exponent) * length / f64::from(steps)
            })
            .sum();

        let exact = obstacle.segment_risk(&from, &to);
        assert!((exact - numeric).abs() < 1e-9, "{exact} vs {numeric}");

        let mid = Euclidean::<2>.interpolate(&from, &to, 0.3);
        let split = obstacle.segment_risk(&from, &mid) + obstacle.segment_risk(&mid, &to);
        assert!(
            (exact - split).abs() < 1e-12,
            "line integrals add up over edges"
        );
    }

    #[test]
    fn a_line_through_the_mean_integrates_to_the_marginal() {
        // The infinite line through the mean of a standard 2D Gaussian integrates to
        // 1 / sqrt(2 pi); a long segment gets all of it.
        let risk = unit().segment_risk(&[-20.0, 0.0], &[20.0, 0.0]);

        assert!((risk - 1.0 / libm::sqrt(2.0 * PI)).abs() < 1e-12);
        assert!(unit().segment_risk(&[1.0, 1.0], &[1.0, 1.0]).abs() < f64::EPSILON);
    }

    #[test]
    fn hybrid_cost_adds_weighted_length_and_bounds_informed_sampling() {
        let objective = RiskDensity::new(alloc::vec![unit()], 0.5);
        let (a, b) = ([5.0, 5.0], [8.0, 9.0]);
        let cost = objective.motion_cost(&Euclidean::<2>, &a, &b);

        assert!((cost - objective.risk(&a, &b) - 2.5).abs() < 1e-12);
        assert!(cost >= CostObjective::<Euclidean<2>>::distance_weight(&objective) * 5.0);
        assert!((CostObjective::<Euclidean<2>>::distance_weight(&PathLength) - 1.0).abs() < 1e-15);
        assert!(GaussianObstacle::new([0.0], [[-1.0]], 1.0).is_none());
    }
}
