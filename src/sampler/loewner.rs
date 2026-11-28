//! Constant lower bounds of a configuration-dependent metric tensor in the Loewner order.

/// A constant symmetric positive definite matrix `G_lower = L L^T` with `G_lower <= G(q)` in
/// the Loewner order (`G(q) - G_lower` positive semidefinite) for the metric tensors `G(q)`
/// it was built from.
///
/// This is the bound of Kyaw and Kelly, "Direct Informed Sampling on Riemannian Manifolds via
/// Loewner Order Lower Bounds" (RA-L 11(12), 2026, arXiv 2606.02879). Under it the length of
/// every path is at least `||L^T (b - a)||` ([`LoewnerBound::distance`], their Theorem 2), so
/// the map `q -> L^T q` turns the informed set of a Riemannian path-length problem into a
/// superset that is a prolate hyperspheroid, which
/// [`UniformBox::with_metric_bound`](crate::UniformBox::with_metric_bound) samples directly.
///
/// [`LoewnerBound::meet`] is their Algorithm 1: it lowers the bound just enough to lie below
/// one more metric evaluation, keeping it a maximal lower bound of all evaluations seen. The
/// Loewner order is partial, so a maximal bound is not unique and depends on the order of
/// the evaluations.
///
/// **The bound is only as admissible as the evaluations it has seen.** It lies below `G` at
/// the configurations passed to [`LoewnerBound::new`] and [`LoewnerBound::meet`], not
/// necessarily anywhere else. The paper searches for the configuration that violates the
/// bound most (multi-start gradient descent on `lambda_min(L^-1 G(q) L^-T)`) and meets it until
/// none is found; this crate leaves choosing the configurations to the caller. Where `G` is
/// smallest at known points (a constant metric, or one that grows away from a point), evaluate
/// there. Otherwise evaluate densely and check with [`LoewnerBound::is_below`].
///
/// # Examples
///
/// ```
/// use sampling_planning::LoewnerBound;
///
/// let mut bound = LoewnerBound::new([[4.0, 0.0], [0.0, 1.0]]).expect("positive definite");
/// bound.meet(&[[1.0, 0.0], [0.0, 9.0]]);
///
/// // The meet of two diagonal metrics is their elementwise minimum.
/// let lower = bound.matrix();
/// assert!((lower[0][0] - 1.0).abs() < 1e-12 && (lower[1][1] - 1.0).abs() < 1e-12);
/// assert!(bound.is_below(&[[4.0, 0.0], [0.0, 1.0]], 1e-9));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoewnerBound<const N: usize> {
    /// The lower-triangular Cholesky factor `L` of the bound, with a positive diagonal.
    factor: [[f64; N]; N],
}

impl<const N: usize> LoewnerBound<N> {
    /// The bound equal to the metric tensor `metric`, the first evaluation.
    ///
    /// Returns `None` unless `metric` is finite, symmetric (up to `1e-9` relative) and
    /// positive definite, and `N > 0`.
    #[must_use]
    pub fn new(metric: [[f64; N]; N]) -> Option<Self> {
        if N == 0 || !is_symmetric(&metric) {
            return None;
        }

        cholesky(&metric).map(|factor| Self { factor })
    }

    /// The bound built from several metric evaluations: the first, met with all the others.
    ///
    /// Returns `None` if there are none or one is not symmetric positive definite.
    #[must_use]
    pub fn from_metrics(metrics: impl IntoIterator<Item = [[f64; N]; N]>) -> Option<Self> {
        let mut metrics = metrics.into_iter();
        let mut bound = Self::new(metrics.next()?)?;

        for metric in metrics {
            Self::new(metric)?;
            bound.meet(&metric);
        }

        Some(bound)
    }

    /// Lowers the bound to also lie below `metric` (Algorithm 1 of Kyaw and Kelly): with
    /// `S = L^-1 G L^-T = V diag(lambda) V^T`, the new bound is
    /// `L V diag(min(lambda, 1)) V^T L^T`. Returns whether the bound changed; it does not when
    /// it already lies below `metric`.
    ///
    /// # Panics
    ///
    /// Panics if `metric` is not finite, symmetric and positive definite.
    pub fn meet(&mut self, metric: &[[f64; N]; N]) -> bool {
        assert!(
            Self::new(*metric).is_some(),
            "a metric tensor must be finite, symmetric and positive definite"
        );

        let (eigenvalues, vectors) = symmetric_eigen(&self.relative(metric));
        if eigenvalues.iter().all(|&lambda| lambda >= 1.0) {
            return false;
        }

        let clamped = eigenvalues.map(|lambda| lambda.min(1.0));
        let inner: [[f64; N]; N] = core::array::from_fn(|i| {
            core::array::from_fn(|j| {
                (0..N)
                    .map(|k| vectors[i][k] * clamped[k] * vectors[j][k])
                    .sum()
            })
        });
        let lower = symmetrize(&congruence(&self.factor, &inner));

        // The clamped eigenvalues are positive, so the product is positive definite; should
        // rounding make it fail, keep the old bound below the new metric by scaling it.
        self.factor = cholesky(&lower).unwrap_or_else(|| {
            let scale = libm::sqrt(eigenvalues.iter().copied().fold(1.0, f64::min));
            self.factor.map(|row| row.map(|value| value * scale))
        });

        true
    }

    /// Whether the bound lies below `metric` up to `tolerance`: the smallest eigenvalue of
    /// `L^-1 G L^-T` is at least `1 - tolerance`.
    #[must_use]
    pub fn is_below(&self, metric: &[[f64; N]; N], tolerance: f64) -> bool {
        let (eigenvalues, _) = symmetric_eigen(&self.relative(metric));

        eigenvalues.iter().all(|&lambda| lambda >= 1.0 - tolerance)
    }

    /// The bound `G_lower = L L^T`.
    #[must_use]
    pub fn matrix(&self) -> [[f64; N]; N] {
        let l = &self.factor;

        core::array::from_fn(|i| core::array::from_fn(|j| (0..N).map(|k| l[i][k] * l[j][k]).sum()))
    }

    /// The lower-triangular Cholesky factor `L` of the bound.
    #[must_use]
    pub const fn factor(&self) -> &[[f64; N]; N] {
        &self.factor
    }

    /// The constant-metric distance `||L^T (b - a)||`, a lower bound on the length of every
    /// path from `a` to `b` wherever the bound lies below the metric.
    #[must_use]
    pub fn distance(&self, a: &[f64; N], b: &[f64; N]) -> f64 {
        let embedded_a = self.embed(a);
        let embedded_b = self.embed(b);

        libm::sqrt(
            (0..N)
                .map(|i| (embedded_b[i] - embedded_a[i]) * (embedded_b[i] - embedded_a[i]))
                .sum(),
        )
    }

    /// The image `L^T q` of `q` in the coordinates where the bound is the identity.
    pub(crate) fn embed(&self, q: &[f64; N]) -> [f64; N] {
        core::array::from_fn(|i| (i..N).map(|k| self.factor[k][i] * q[k]).sum())
    }

    /// The preimage `L^-T x` of embedded coordinates `x`, by back substitution.
    pub(crate) fn unembed(&self, x: &[f64; N]) -> [f64; N] {
        let mut q = [0.0; N];

        for i in (0..N).rev() {
            let tail: f64 = (i + 1..N).map(|k| self.factor[k][i] * q[k]).sum();
            q[i] = (x[i] - tail) / self.factor[i][i];
        }

        q
    }

    /// `det L = sqrt(det G_lower)`, the factor by which [`LoewnerBound::embed`] scales volumes.
    pub(crate) fn volume_scale(&self) -> f64 {
        (0..N).map(|i| self.factor[i][i]).product()
    }

    /// `S = L^-1 G L^-T`, symmetrized: the metric in embedded coordinates.
    fn relative(&self, metric: &[[f64; N]; N]) -> [[f64; N]; N] {
        // M = L^-1 G column by column, then S = L^-1 M^T, since G is symmetric.
        let m = self.solve_lower_columns(metric);
        let transposed: [[f64; N]; N] = core::array::from_fn(|i| core::array::from_fn(|j| m[j][i]));

        symmetrize(&self.solve_lower_columns(&transposed))
    }

    /// `L^-1 B` by forward substitution on each column of `B`.
    fn solve_lower_columns(&self, b: &[[f64; N]; N]) -> [[f64; N]; N] {
        let mut x = [[0.0; N]; N];

        for column in 0..N {
            for i in 0..N {
                let head: f64 = (0..i).map(|k| self.factor[i][k] * x[k][column]).sum();
                x[i][column] = (b[i][column] - head) / self.factor[i][i];
            }
        }

        x
    }
}

/// Whether `m` is finite and symmetric up to `1e-9` relative to its largest entry.
pub(crate) fn is_symmetric<const N: usize>(m: &[[f64; N]; N]) -> bool {
    let scale = m
        .iter()
        .flatten()
        .fold(0.0_f64, |acc, v| acc.max(libm::fabs(*v)));

    m.iter().flatten().all(|v| v.is_finite())
        && (0..N).all(|i| (0..i).all(|j| libm::fabs(m[i][j] - m[j][i]) <= 1e-9 * scale))
}

/// `(m + m^T) / 2`.
fn symmetrize<const N: usize>(m: &[[f64; N]; N]) -> [[f64; N]; N] {
    core::array::from_fn(|i| core::array::from_fn(|j| f64::midpoint(m[i][j], m[j][i])))
}

/// `L A L^T`.
fn congruence<const N: usize>(l: &[[f64; N]; N], a: &[[f64; N]; N]) -> [[f64; N]; N] {
    let la: [[f64; N]; N] =
        core::array::from_fn(|i| core::array::from_fn(|j| (0..N).map(|k| l[i][k] * a[k][j]).sum()));

    core::array::from_fn(|i| core::array::from_fn(|j| (0..N).map(|k| la[i][k] * l[j][k]).sum()))
}

/// The lower-triangular Cholesky factor of a symmetric matrix, or `None` unless it is
/// positive definite.
pub(crate) fn cholesky<const N: usize>(m: &[[f64; N]; N]) -> Option<[[f64; N]; N]> {
    let mut l = [[0.0; N]; N];

    for i in 0..N {
        for j in 0..=i {
            let dot: f64 = (0..j).map(|k| l[i][k] * l[j][k]).sum();

            if i == j {
                let pivot = m[i][i] - dot;
                if !(pivot.is_finite() && pivot > 0.0) {
                    return None;
                }

                l[i][i] = libm::sqrt(pivot);
            } else {
                l[i][j] = (m[i][j] - dot) / l[j][j];
            }
        }
    }

    Some(l)
}

/// The eigenvalues and eigenvectors (as the columns of the second result) of a symmetric
/// matrix, by cyclic Jacobi rotations.
///
/// Accurate to rounding for the small, well-conditioned matrices of configuration spaces;
/// sweeps stop once the off-diagonal mass drops below `1e-30` of the total, or after 64
/// sweeps.
fn symmetric_eigen<const N: usize>(m: &[[f64; N]; N]) -> ([f64; N], [[f64; N]; N]) {
    let mut a = *m;
    let mut v: [[f64; N]; N] =
        core::array::from_fn(|i| core::array::from_fn(|j| if i == j { 1.0 } else { 0.0 }));
    let total: f64 = a.iter().flatten().map(|x| x * x).sum();

    for _ in 0..64 {
        let off: f64 = (0..N)
            .flat_map(|i| (0..N).filter(move |&j| j != i).map(move |j| (i, j)))
            .map(|(i, j)| a[i][j] * a[i][j])
            .sum();

        if off <= 1e-30 * total {
            break;
        }

        for p in 0..N {
            for q in p + 1..N {
                rotate(&mut a, &mut v, p, q);
            }
        }
    }

    (core::array::from_fn(|i| a[i][i]), v)
}

/// One Jacobi rotation in the `(p, q)` plane that zeroes `matrix[p][q]`, applied to `matrix`
/// from both sides and accumulated into the eigenvector columns `vectors`.
fn rotate<const N: usize>(
    matrix: &mut [[f64; N]; N],
    vectors: &mut [[f64; N]; N],
    p: usize,
    q: usize,
) {
    if matrix[p][q] == 0.0 {
        return;
    }

    let theta = (matrix[q][q] - matrix[p][p]) / (2.0 * matrix[p][q]);
    let tangent =
        libm::copysign(1.0, theta) / (libm::fabs(theta) + libm::sqrt(theta * theta + 1.0));
    let cos = 1.0 / libm::sqrt(tangent * tangent + 1.0);
    let sin = tangent * cos;
    let turn = |x: f64, y: f64| (cos * x - sin * y, sin * x + cos * y);

    for row in matrix.iter_mut() {
        (row[p], row[q]) = turn(row[p], row[q]);
    }

    for column in 0..N {
        (matrix[p][column], matrix[q][column]) = turn(matrix[p][column], matrix[q][column]);
    }

    for row in vectors.iter_mut() {
        (row[p], row[q]) = turn(row[p], row[q]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tilted positive definite 3 x 3 matrix.
    const TILTED: [[f64; 3]; 3] = [[4.0, 1.0, 0.5], [1.0, 3.0, -0.4], [0.5, -0.4, 2.0]];

    #[test]
    fn cholesky_and_embedding_round_trip() {
        let bound = LoewnerBound::new(TILTED).expect("positive definite");
        let product = bound.matrix();

        for (row, expected) in product.iter().zip(&TILTED) {
            for (a, b) in row.iter().zip(expected) {
                assert!((a - b).abs() < 1e-12);
            }
        }

        let q = [0.3, -1.2, 2.5];
        let back = bound.unembed(&bound.embed(&q));
        assert!((0..3).all(|i| (back[i] - q[i]).abs() < 1e-12), "{back:?}");

        // ||L^T q||^2 = q^T G q.
        let quadratic: f64 = (0..3)
            .flat_map(|i| (0..3).map(move |j| q[i] * TILTED[i][j] * q[j]))
            .sum();
        assert!((bound.distance(&[0.0; 3], &q).powi(2) - quadratic).abs() < 1e-9);
    }

    #[test]
    fn eigen_decomposition_reconstructs_the_matrix() {
        let (lambda, v) = symmetric_eigen(&TILTED);

        for i in 0..3 {
            for j in 0..3 {
                let value: f64 = (0..3).map(|k| v[i][k] * lambda[k] * v[j][k]).sum();
                assert!((value - TILTED[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn meet_lies_below_both_inputs_and_keeps_larger_metrics() {
        let other = [[1.0, -0.3, 0.0], [-0.3, 5.0, 1.0], [0.0, 1.0, 2.5]];
        let mut bound = LoewnerBound::new(TILTED).expect("positive definite");

        assert!(bound.meet(&other));
        assert!(bound.is_below(&TILTED, 1e-9) && bound.is_below(&other, 1e-9));

        // The meet is maximal: it touches each input along some direction.
        let touches = |g: &[[f64; 3]; 3]| !bound.is_below(g, -1e-6);
        assert!(touches(&TILTED) || touches(&other));

        let larger = TILTED.map(|row| row.map(|value| 3.0 * value));
        assert!(!bound.meet(&larger), "a larger metric changes nothing");
    }

    #[test]
    fn invalid_metrics_are_rejected() {
        assert!(
            LoewnerBound::new([[1.0, 2.0], [2.0, 1.0]]).is_none(),
            "indefinite"
        );
        assert!(
            LoewnerBound::new([[1.0, 0.5], [0.0, 1.0]]).is_none(),
            "asymmetric"
        );
        assert!(LoewnerBound::<2>::from_metrics([]).is_none());
    }
}
