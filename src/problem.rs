//! Planning problems and their solutions.

use alloc::vec::Vec;

use crate::{GoalRegion, StateSpace};

/// Everything a planner needs to know about the world.
///
/// `Sa` is a [`Sampler`](crate::Sampler), `C` a [`ValidityChecker`](crate::ValidityChecker) and
/// `St` a [`Steering`](crate::Steering) function, all over the space `Sp`. Planners take the
/// problem by value and check the start state when they are created.
#[derive(Debug, Clone)]
pub struct Problem<Sp: StateSpace, Sa, C, St> {
    /// The configuration space.
    pub space: Sp,
    /// Draws the states the planner grows toward.
    pub sampler: Sa,
    /// Decides which states and motions are allowed.
    pub checker: C,
    /// Extends nodes toward samples.
    pub steering: St,
    /// The initial state. Must be [`Validity::Valid`](crate::Validity::Valid).
    pub start: Sp::State,
    /// Where the path has to end.
    pub goal: GoalRegion<Sp::State>,
}

/// A solution path.
#[derive(Debug, Clone, PartialEq)]
pub struct Path<S> {
    /// The states from the start to a state in the goal region. Never empty.
    pub states: Vec<S>,
    /// The path length under [`StateSpace::distance`].
    pub cost: f64,
}
