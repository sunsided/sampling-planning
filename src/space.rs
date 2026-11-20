//! State spaces: the metric and interpolation planners work in.

mod euclidean;

pub use euclidean::Euclidean;

/// A continuous configuration space with a metric and a geodesic interpolation.
///
/// Planners use [`StateSpace::distance`] both to find neighbors and as the path cost. Bounds
/// are not part of the space: the [`Sampler`](crate::Sampler) decides where states come from
/// and the [`ValidityChecker`](crate::ValidityChecker) decides where they may be.
pub trait StateSpace {
    /// A configuration. Cloned whenever a planner stores a node.
    type State: Clone;

    /// The number of degrees of freedom; feeds the RRT*/PRM* connection radius.
    fn dimension(&self) -> usize;

    /// The distance between two configurations.
    ///
    /// Must be a metric: non-negative, zero only for equal states, symmetric and satisfying the
    /// triangle inequality.
    fn distance(&self, from: &Self::State, to: &Self::State) -> f64;

    /// The configuration at fraction `t` in `[0, 1]` of the way from `from` to `to`.
    ///
    /// `t = 0` yields `from` and `t = 1` yields `to`.
    fn interpolate(&self, from: &Self::State, to: &Self::State, t: f64) -> Self::State;

    /// Whether states are coordinate vectors, [`StateSpace::distance`] is the Euclidean norm of
    /// their difference and [`StateSpace::interpolate`] the straight line.
    ///
    /// Samplers that exploit Euclidean geometry, such as the direct informed sampling of
    /// [`UniformBox`](crate::UniformBox), check this before they assume it. The default is
    /// `false`, which is always safe.
    fn is_euclidean(&self) -> bool {
        false
    }
}
