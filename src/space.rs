//! State spaces: the metric and interpolation planners work in.

mod curve;
mod dubins;
mod euclidean;
mod reeds_shepp;
mod se2;
mod so2;

pub use curve::{Segment, SegmentKind};
pub use dubins::{Dubins, DubinsPath, DubinsWord};
pub use euclidean::Euclidean;
pub use reeds_shepp::{ReedsShepp, ReedsSheppPath};
pub use se2::{Se2, Se2State, UniformSe2};
pub use so2::{So2, UniformAngle, angle_difference, wrap_angle};

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
    /// The length of the shortest motion from `from` to `to`.
    ///
    /// Must be non-negative, zero for equal states and satisfy the triangle inequality. It
    /// should be symmetric; [`Dubins`] is the exception, because a forward-only car's way back
    /// is longer. Forward tree planners ([`Rrt`](crate::trees::Rrt),
    /// [`RrtStar`](crate::trees::RrtStar), [`RrtSharp`](crate::trees::RrtSharp),
    /// [`Sst`](crate::kinodynamic::Sst)) always measure from a tree node to the target and
    /// work with asymmetric distances. Planners that need symmetry: [`Prm`](crate::roadmaps::Prm)
    /// stores one cost per edge for both directions, the goal tree of
    /// [`RrtConnect`](crate::trees::RrtConnect) measures from the target to its nodes,
    /// [`Spars2`](crate::roadmaps::Spars2) keeps undirected edges, and
    /// [`FmtStar`](crate::roadmaps::FmtStar), [`BfmtStar`](crate::roadmaps::BfmtStar),
    /// [`BitStar`](crate::informed::BitStar) and [`KdTree`](crate::KdTree) assume a metric
    /// for their neighborhoods and heuristics.
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
