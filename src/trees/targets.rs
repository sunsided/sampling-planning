//! Growth targets: goal-biased samples, optionally restricted to the informed set.

use rand_core::Rng;

use super::INFORMED_ATTEMPTS;
use crate::{Problem, Sampler, StateSpace, unit_f64};

/// Draws the next growth target of a tree planner.
///
/// With probability `goal_bias` this is the goal center. Otherwise it is a sampler state; once
/// `informed_bound` is `Some(c)`, a state of the informed set
/// `d(start, x) + d(x, goal center) < c`, drawn directly through
/// [`Sampler::sample_informed`] when the sampler supports it and by rejection otherwise.
///
/// Rejection redraws with goal bias, up to [`INFORMED_ATTEMPTS`] samples in total, and then
/// settles for the last one. A sampler that declines without drawing leaves the random stream
/// exactly as pure rejection sampling uses it, and without a bound the stream is that of plain
/// goal-biased sampling.
pub(super) fn draw_target<Sp, Sa, C, St, R>(
    problem: &mut Problem<Sp, Sa, C, St>,
    goal_bias: f64,
    informed_bound: Option<f64>,
    rng: &mut R,
) -> Sp::State
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    R: Rng + ?Sized,
{
    if unit_f64(rng) < goal_bias {
        return problem.goal.center().clone();
    }

    let Some(bound) = informed_bound else {
        return problem.sampler.sample(rng);
    };

    let direct = problem.sampler.sample_informed(
        &problem.space,
        rng,
        &problem.start,
        problem.goal.center(),
        bound,
    );
    if let Some(state) = direct {
        return state;
    }

    let mut sample = problem.sampler.sample(rng);
    for _ in 1..INFORMED_ATTEMPTS {
        if informed_cost(problem, &sample) < bound {
            break;
        }

        sample = draw_target(problem, goal_bias, None, rng);
    }

    sample
}

/// The informed heuristic `d(start, x) + d(x, goal center)`: a lower bound on the cost of a
/// solution through `x`, plus the goal tolerance.
pub(super) fn informed_cost<Sp: StateSpace, Sa, C, St>(
    problem: &Problem<Sp, Sa, C, St>,
    state: &Sp::State,
) -> f64 {
    problem.space.distance(&problem.start, state)
        + problem.space.distance(state, problem.goal.center())
}
