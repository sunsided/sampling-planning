//! Kinodynamic planners: SST, SST* and KPIECE.
//!
//! Kinodynamic planners cannot steer exactly between two states. They grow a tree by forward
//! simulation instead: a [`Propagator`] applies a control for some time. Edges are therefore
//! trajectories, solutions are [`Trajectory`] values with the controls that produce them, and
//! the cost is the total control duration.

mod kpiece;
mod sst;

use alloc::vec::Vec;
use core::fmt;

use rand_core::Rng;

pub use kpiece::{Kpiece, KpieceConfig, Projection};
pub use sst::{Sst, SstConfig, SstStarSchedule};

use crate::{GoalRegion, StateSpace, Validity};

/// Forward-simulates a system: the kinodynamic counterpart of [`Steering`](crate::Steering).
pub trait Propagator<Sp: StateSpace> {
    /// A control input, for example a velocity or an acceleration vector.
    type Control: Clone;

    /// The number of control dimensions; feeds the SST* epoch schedule.
    fn control_dimension(&self) -> usize;

    /// Draws a random admissible control.
    fn sample_control<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Self::Control;

    /// The state reached from `from` by applying `control` for `duration` time units.
    ///
    /// Planners call this with short durations (one integration step) and check every step
    /// with [`ValidityChecker::check_motion`](crate::ValidityChecker::check_motion), so it must
    /// be deterministic: the same arguments always yield the same state.
    fn propagate(
        &self,
        space: &Sp,
        from: &Sp::State,
        control: &Self::Control,
        duration: f64,
    ) -> Sp::State;
}

/// Everything a kinodynamic planner needs to know about the world.
#[derive(Debug, Clone)]
pub struct KinodynamicProblem<Sp: StateSpace, Sa, C, P> {
    /// The state space, also used as the metric for neighbor queries and the goal region.
    pub space: Sp,
    /// Draws the states the planner grows toward.
    pub sampler: Sa,
    /// Decides which states and motions are allowed.
    pub checker: C,
    /// Simulates the system.
    pub propagator: P,
    /// The initial state. Must be [`Validity::Valid`].
    pub start: Sp::State,
    /// Where the trajectory has to end.
    pub goal: GoalRegion<Sp::State>,
}

/// A solution trajectory: states and the controls that connect them.
#[derive(Debug, Clone, PartialEq)]
pub struct Trajectory<S, U> {
    /// The states from the start to a state in the goal region. Never empty.
    pub states: Vec<S>,
    /// `controls[i]` is the control and its duration that take `states[i]` to `states[i + 1]`.
    /// One shorter than `states`.
    pub controls: Vec<(U, f64)>,
    /// The total duration of all controls.
    pub cost: f64,
}

/// Why a kinodynamic planner could not be created.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// The start state did not check as [`Validity::Valid`]; carries the actual result.
    InvalidStart(Validity),
    /// The goal bias is not in `[0, 1]`.
    GoalBiasOutOfRange,
    /// A radius or KPIECE cell size is not positive and finite.
    InvalidRadius,
    /// The integration step is not positive and finite, or the step count is zero.
    InvalidPropagation,
    /// The SST* shrink factor is not in `(0, 1)` or the first epoch is empty; or a KPIECE score
    /// factor is not in `(0, 1]` or its exterior bias not in `[0, 1]`.
    InvalidSchedule,
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStart(validity) => write!(f, "start state is not valid ({validity:?})"),
            Self::GoalBiasOutOfRange => f.write_str("goal bias must be in [0, 1]"),
            Self::InvalidRadius => f.write_str("radii must be positive and finite"),
            Self::InvalidPropagation => {
                f.write_str("step size must be positive and finite, step count at least one")
            }
            Self::InvalidSchedule => {
                f.write_str("shrink factor must be in (0, 1), first epoch at least one iteration")
            }
        }
    }
}

impl core::error::Error for Error {}
