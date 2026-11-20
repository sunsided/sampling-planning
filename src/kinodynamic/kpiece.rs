//! KPIECE: kinodynamic planning by interior-exterior cell exploration (Sucan and Kavraki).

mod grid;

use alloc::vec::Vec;

use rand_core::Rng;

use super::{Error, KinodynamicProblem, Propagator, Trajectory, sst::random_step_count};
use crate::radius::count_to_f64;
use crate::{StateSpace, ValidityChecker, unit_f64};
use grid::Grid;

/// Maps a state to a point in a low-dimensional projection space, where KPIECE lays its grid.
///
/// A good projection keeps the coordinates that tell progress apart (the position of a robot,
/// not its velocity). Any `Fn(&S) -> [f64; K]` closure is a projection.
pub trait Projection<S, const K: usize> {
    /// The projection of `state`.
    fn project(&self, state: &S) -> [f64; K];
}

impl<S, const K: usize, F: Fn(&S) -> [f64; K]> Projection<S, K> for F {
    fn project(&self, state: &S) -> [f64; K] {
        self(state)
    }
}

/// Settings for [`Kpiece`]. The defaults follow OMPL's `control::KPIECE1`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct KpieceConfig<const K: usize> {
    /// The edge lengths of a grid cell in projection space. Positive and finite.
    pub cell_size: [f64; K],
    /// The integration step: every propagation is a sequence of steps of this duration, each
    /// checked with [`ValidityChecker::check_motion`]. Positive and finite.
    pub step_size: f64,
    /// The most integration steps per propagation; the step count is uniform in
    /// `1..=max_steps`. At least `min_steps`.
    pub max_steps: usize,
    /// The fewest valid steps for a propagation to count; shorter ones are discarded and
    /// penalize their cell. At least one.
    pub min_steps: usize,
    /// Probability in `[0, 1]` of expanding one of the motions closest to the goal instead of
    /// a grid selection. 5 % by default.
    pub goal_bias: f64,
    /// The least probability of selecting an exterior cell; the share of exterior cells is used
    /// when it is higher. In `[0, 1]`; 0.8 by default (the paper suggests 0.7 to 0.8).
    pub exterior_bias: f64,
    /// The factor applied to a cell's score after a successful expansion. In `(0, 1]`; 0.9 by
    /// default.
    pub good_score_factor: f64,
    /// The factor applied after a failed expansion. In `(0, 1]`; 0.45 by default.
    pub bad_score_factor: f64,
    /// How many motions closest to the goal the goal bias chooses from. 30 by default.
    pub close_samples: usize,
}

impl<const K: usize> KpieceConfig<K> {
    /// A config with the given cell size, integration step and step range `1..=max_steps`, and
    /// OMPL's defaults for the rest.
    #[must_use]
    pub const fn new(cell_size: [f64; K], step_size: f64, max_steps: usize) -> Self {
        Self {
            cell_size,
            step_size,
            max_steps,
            min_steps: 1,
            goal_bias: 0.05,
            exterior_bias: 0.8,
            good_score_factor: 0.9,
            bad_score_factor: 0.45,
            close_samples: 30,
        }
    }

    /// Checks every field against its documented range.
    fn validate(&self) -> Result<(), Error> {
        let positive = |value: f64| value.is_finite() && value > 0.0;
        let factor = |value: f64| value > 0.0 && value <= 1.0;
        let probability = |value: f64| (0.0..=1.0).contains(&value);

        if !probability(self.goal_bias) {
            return Err(Error::GoalBiasOutOfRange);
        }

        if !self.cell_size.iter().all(|&size| positive(size)) {
            return Err(Error::InvalidRadius);
        }

        if !positive(self.step_size) || self.min_steps == 0 || self.max_steps < self.min_steps {
            return Err(Error::InvalidPropagation);
        }

        let scores = factor(self.good_score_factor) && factor(self.bad_score_factor);
        if !(scores && probability(self.exterior_bias)) {
            return Err(Error::InvalidSchedule);
        }

        Ok(())
    }
}

/// One tree edge: a constant control applied for a number of steps, ending in one cell.
#[derive(Debug, Clone)]
struct Motion<S, U> {
    /// The state reached at the end of the motion.
    state: S,
    /// The motion this one continues; `None` only for the start.
    parent: Option<usize>,
    /// The control and its duration; `None` only for the start.
    control: Option<(U, f64)>,
    /// The total duration from the start.
    cost: f64,
    /// The grid cell of `state`.
    cell: usize,
}

/// KPIECE: grows a tree of motions by forward propagation, guided by a grid over a projection
/// of the state space (Sucan and Kavraki 2008, 2012).
///
/// Each motion's end state falls into a grid cell. Cells with all `2K` axis neighbors are
/// interior, the others exterior, the frontier of exploration. Each iteration:
///
/// ```mermaid
/// graph LR
///     Select["select cell: exterior with prob. max(bias, share)"] --> Motion["pick motion, newest preferred"]
///     Motion --> Propagate["random control, random steps, stop at first invalid step"]
///     Propagate -->|long enough| Split["split at cell changes, add pieces"]
///     Propagate --> Score["scale cell score by good or bad factor"]
/// ```
///
/// Cells are ranked by importance `score / ((neighbors + 1) coverage selections)`; a new cell
/// starts with score `(1 + ln i) / (1e-3 + goal distance)`. Within a cell, motions are picked
/// from a half-normal distribution centered on the newest one. This follows OMPL's
/// `control::KPIECE1` with a single grid level, which also expands from the end state of the
/// chosen motion rather than a random state along it. With probability `goal_bias` one of the
/// `close_samples` motions closest to the goal is expanded instead.
///
/// KPIECE is probabilistically complete but not asymptotically optimal: [`Kpiece::solve`]
/// stops at the first motion that reaches the goal region. It needs no state sampler; the
/// problem's sampler is not used.
///
/// # Examples
///
/// ```
/// use rand_core::{Rng, SeedableRng};
/// use sampling_planning::{
///     Euclidean, GoalRegion, UniformBox, Validity, ValidityChecker, unit_f64,
///     kinodynamic::{KinodynamicProblem, Kpiece, KpieceConfig, Propagator},
/// };
///
/// /// Free space everywhere.
/// struct Free;
///
/// impl ValidityChecker<Euclidean<2>> for Free {
///     fn check_state(&self, _: &[f64; 2]) -> Validity {
///         Validity::Valid
///     }
///
///     fn check_motion(&self, _: &Euclidean<2>, _: &[f64; 2], _: &[f64; 2]) -> Validity {
///         Validity::Valid
///     }
/// }
///
/// /// A point that moves with a velocity of at most 1 per axis.
/// struct Velocity;
///
/// impl Propagator<Euclidean<2>> for Velocity {
///     type Control = [f64; 2];
///
///     fn control_dimension(&self) -> usize {
///         2
///     }
///
///     fn sample_control<R: Rng + ?Sized>(&mut self, rng: &mut R) -> [f64; 2] {
///         [2.0 * unit_f64(rng) - 1.0, 2.0 * unit_f64(rng) - 1.0]
///     }
///
///     fn propagate(&self, _: &Euclidean<2>, from: &[f64; 2], u: &[f64; 2], dt: f64) -> [f64; 2] {
///         [from[0] + u[0] * dt, from[1] + u[1] * dt]
///     }
/// }
///
/// let problem = KinodynamicProblem {
///     space: Euclidean::<2>,
///     sampler: UniformBox::new([0.0, 0.0], [10.0, 10.0]),
///     checker: Free,
///     propagator: Velocity,
///     start: [1.0, 1.0],
///     goal: GoalRegion::new([9.0, 9.0], 0.5),
/// };
///
/// // The identity projection with half-unit cells.
/// let projection = |state: &[f64; 2]| *state;
/// let config = KpieceConfig::new([0.5, 0.5], 0.1, 10);
/// let mut kpiece = Kpiece::new(problem, projection, config).expect("valid");
///
/// let trajectory = kpiece
///     .solve(20_000, &mut rand_pcg::Pcg64Mcg::seed_from_u64(1))
///     .expect("free space");
/// assert_eq!(trajectory.controls.len() + 1, trajectory.states.len());
/// ```
#[derive(Clone)]
pub struct Kpiece<Sp, Sa, C, P, Pr, const K: usize>
where
    Sp: StateSpace,
    P: Propagator<Sp>,
{
    /// The problem being solved.
    problem: KinodynamicProblem<Sp, Sa, C, P>,
    /// Maps states into the grid's space.
    projection: Pr,
    /// The validated settings.
    config: KpieceConfig<K>,
    /// All motions; motion `0` is the start.
    motions: Vec<Motion<Sp::State, P::Control>>,
    /// The grid over the motions' end states.
    grid: Grid<K>,
    /// Up to `close_samples` motions closest to the goal as `(goal distance, motion)`, closest
    /// first.
    close: Vec<(f64, usize)>,
    /// The first motion that reached the goal region.
    solution: Option<usize>,
    /// The number of iterations run, including the current one.
    iteration: usize,
}

impl<Sp, Sa, C, P, Pr, const K: usize> Kpiece<Sp, Sa, C, P, Pr, K>
where
    Sp: StateSpace,
    C: ValidityChecker<Sp>,
    P: Propagator<Sp>,
    Pr: Projection<Sp::State, K>,
{
    /// Creates a planner whose tree holds only the start.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidStart`] if the start state is not valid.
    /// - [`Error::GoalBiasOutOfRange`] if `goal_bias` is not in `[0, 1]`.
    /// - [`Error::InvalidRadius`] if a cell size is not positive and finite.
    /// - [`Error::InvalidPropagation`] if the step size is not positive and finite or the step
    ///   range is empty.
    /// - [`Error::InvalidSchedule`] if a score factor is not in `(0, 1]` or `exterior_bias` not
    ///   in `[0, 1]`.
    pub fn new(
        problem: KinodynamicProblem<Sp, Sa, C, P>,
        projection: Pr,
        config: KpieceConfig<K>,
    ) -> Result<Self, Error> {
        config.validate()?;

        let validity = problem.checker.check_state(&problem.start);
        if !validity.is_valid() {
            return Err(Error::InvalidStart(validity));
        }

        let mut planner = Self {
            problem,
            projection,
            config,
            motions: Vec::new(),
            grid: Grid::new(),
            close: Vec::new(),
            solution: None,
            iteration: 1,
        };
        let start = planner.problem.start.clone();
        planner.add_motion(start, None, None, 0.0, 0);

        Ok(planner)
    }

    /// Runs iterations until a motion reaches the goal region or `max_iterations` have run, and
    /// returns the solution, if any.
    pub fn solve<R: Rng + ?Sized>(
        &mut self,
        max_iterations: usize,
        rng: &mut R,
    ) -> Option<Trajectory<Sp::State, P::Control>> {
        for _ in 0..max_iterations {
            if self.solution.is_some() {
                break;
            }

            self.step(rng);
        }

        self.best_trajectory()
    }

    /// Runs one iteration. Returns whether motions were added.
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> bool {
        self.iteration += 1;

        let (from, cell) = self.choose_motion(rng);
        let control = self.problem.propagator.sample_control(rng);
        let steps = random_step_count(self.config.max_steps, rng);
        let states = self.propagate_while_valid(from, &control, steps);
        let good = states.len() >= self.config.min_steps;

        let added = good && self.add_if_interesting(from, &control, &states, rng);
        self.rescore(cell, good);

        added
    }

    /// Picks the motion to expand and the cell to score: a close-to-goal motion with
    /// probability `goal_bias`, otherwise a grid selection.
    fn choose_motion<R: Rng + ?Sized>(&mut self, rng: &mut R) -> (usize, usize) {
        if !self.close.is_empty() && unit_f64(rng) < self.config.goal_bias {
            let motion = self.take_close_sample();

            return (motion, self.motions[motion].cell);
        }

        let exterior = unit_f64(rng) < self.config.exterior_bias.max(self.grid.exterior_fraction());
        let cell = self
            .grid
            .select(exterior)
            .expect("the start cell always exists");

        if self.grid.cells[cell].score < f64::EPSILON {
            self.grid.refresh_scores();
        }

        let entry = &mut self.grid.cells[cell];
        entry.selections += 1.0;
        let pick = half_normal_index(entry.motions.len(), rng);

        (entry.motions[pick], cell)
    }

    /// The closest-to-goal motion; it stays a close sample with its distance inflated to
    /// `1.1 (best + worst) / 2`, so that other close samples get their turn (as in OMPL).
    fn take_close_sample(&mut self) -> usize {
        let (best, motion) = self.close.remove(0);
        let worst = self.close.last().map_or(best, |&(distance, _)| distance);
        self.insert_close(0.55 * (best + worst), motion);

        motion
    }

    /// Applies `control` from the end of motion `from` for up to `steps` integration steps and
    /// returns the states after each valid step, stopping at the first invalid one.
    fn propagate_while_valid(
        &self,
        from: usize,
        control: &P::Control,
        steps: usize,
    ) -> Vec<Sp::State> {
        let (space, propagator) = (&self.problem.space, &self.problem.propagator);
        let mut states: Vec<Sp::State> = Vec::new();
        let mut state = self.motions[from].state.clone();

        for _ in 0..steps {
            let next = propagator.propagate(space, &state, control, self.config.step_size);

            if !self
                .problem
                .checker
                .check_motion(space, &state, &next)
                .is_valid()
            {
                break;
            }

            states.push(next.clone());
            state = next;
        }

        states
    }

    /// Adds the propagated `states` as motions, one per run of consecutive states in the same
    /// cell, if they are interesting (OMPL's rule): one reaches a new cell or a cell with at
    /// most two thirds of the average motion count, or a 5 % coin flip says so.
    fn add_if_interesting<R: Rng + ?Sized>(
        &mut self,
        from: usize,
        control: &P::Control,
        states: &[Sp::State],
        rng: &mut R,
    ) -> bool {
        let coords: Vec<[i64; K]> = states.iter().map(|state| self.coordinates(state)).collect();

        if !self.is_interesting(&coords) && unit_f64(rng) >= 0.05 {
            return false;
        }

        let mut parent = from;
        let mut start = 0;

        while start < states.len() {
            let end = run_end(&coords, start);
            let duration = count_to_f64(end - start) * self.config.step_size;
            let cost = self.motions[parent].cost + duration;
            let cell = self.cell_for(coords[start], &states[end - 1]);
            parent = self.add_motion(
                states[end - 1].clone(),
                Some(parent),
                Some((control.clone(), duration)),
                cost,
                cell,
            );
            self.grid.cells[cell].coverage += count_to_f64(end - start);
            start = end;
        }

        true
    }

    /// Whether a propagation through cells `coords` adds information: a new cell, or a sparsely
    /// populated one.
    fn is_interesting(&self, coords: &[[i64; K]]) -> bool {
        let threshold = 2 * self.motions.len() / (3 * self.grid.cells.len());

        coords.iter().any(|coord| {
            self.grid
                .find(coord)
                .is_none_or(|cell| self.grid.cells[cell].motions.len() <= threshold)
        })
    }

    /// The cell at `coord`, created with a goal-distance score for `state` if needed.
    fn cell_for(&mut self, coord: [i64; K], state: &Sp::State) -> usize {
        if let Some(cell) = self.grid.find(&coord) {
            return cell;
        }

        let score =
            (1.0 + libm::log(count_to_f64(self.iteration))) / (1e-3 + self.goal_distance(state));

        self.grid.add(coord, self.iteration, score)
    }

    /// Stores a motion in its cell, the close samples and, if it reaches the goal first, as the
    /// solution. `cell` must be the cell of `state`, except for the start, whose cell is
    /// created here.
    fn add_motion(
        &mut self,
        state: Sp::State,
        parent: Option<usize>,
        control: Option<(P::Control, f64)>,
        cost: f64,
        cell: usize,
    ) -> usize {
        let id = self.motions.len();
        let distance = self.goal_distance(&state);
        let cell = if parent.is_none() {
            let coord = self.coordinates(&state);
            let cell = self.cell_for(coord, &state);
            self.grid.cells[cell].coverage = 1.0;

            cell
        } else {
            cell
        };

        if distance <= 0.0 && self.solution.is_none() {
            self.solution = Some(id);
        }

        self.grid.cells[cell].motions.push(id);
        self.insert_close(distance, id);
        self.motions.push(Motion {
            state,
            parent,
            control,
            cost,
            cell,
        });

        id
    }

    /// Keeps `motion` among the `close_samples` closest motions if it is close enough.
    fn insert_close(&mut self, distance: f64, motion: usize) {
        let at = self.close.partition_point(|&(other, _)| other <= distance);

        if at < self.config.close_samples {
            self.close.insert(at, (distance, motion));
            self.close.truncate(self.config.close_samples);
        }
    }

    /// Scales the score of `cell` by the good or bad factor.
    fn rescore(&mut self, cell: usize, good: bool) {
        self.grid.cells[cell].score *= if good {
            self.config.good_score_factor
        } else {
            self.config.bad_score_factor
        };
    }

    /// The grid coordinates of `state`.
    fn coordinates(&self, state: &Sp::State) -> [i64; K] {
        let point = self.projection.project(state);

        core::array::from_fn(|axis| cell_index(point[axis] / self.config.cell_size[axis]))
    }

    /// The distance from `state` to the goal region; zero inside.
    fn goal_distance(&self, state: &Sp::State) -> f64 {
        let goal = &self.problem.goal;

        (self.problem.space.distance(state, goal.center()) - goal.tolerance()).max(0.0)
    }

    /// The first trajectory found into the goal region, or `None` if none was found yet.
    #[must_use]
    pub fn best_trajectory(&self) -> Option<Trajectory<Sp::State, P::Control>> {
        let goal = self.solution?;
        let mut states = Vec::new();
        let mut controls = Vec::new();
        let mut current = Some(goal);

        while let Some(id) = current {
            let motion = &self.motions[id];
            states.push(motion.state.clone());
            controls.extend(motion.control.clone());
            current = motion.parent;
        }

        states.reverse();
        controls.reverse();

        Some(Trajectory {
            states,
            controls,
            cost: self.motions[goal].cost,
        })
    }

    /// The number of motions in the tree, including the start.
    #[must_use]
    pub fn motion_count(&self) -> usize {
        self.motions.len()
    }

    /// The number of instantiated grid cells.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.grid.cells.len()
    }
}

/// One past the last index of the run of equal coordinates starting at `start`.
fn run_end<const K: usize>(coords: &[[i64; K]], start: usize) -> usize {
    coords[start..]
        .iter()
        .position(|coord| *coord != coords[start])
        .map_or(coords.len(), |offset| start + offset)
}

/// The integer cell index of a scaled coordinate: its floor, saturated; NaN maps to zero.
#[expect(
    clippy::cast_possible_truncation,
    reason = "float-to-int casts saturate, which is the intended clamping"
)]
fn cell_index(scaled: f64) -> i64 {
    libm::floor(scaled) as i64
}

/// An index into a list of `len >= 1` items from a half-normal distribution peaked at the last
/// item, the newest motion: OMPL's `halfNormalInt(0, len - 1)` with focus 3.
fn half_normal_index<R: Rng + ?Sized>(len: usize, rng: &mut R) -> usize {
    let mean = count_to_f64(len);
    let mut v = mean + mean / 3.0 * standard_normal(rng);

    if v > mean {
        v = 2.0 * mean - v;
    }

    let value = libm::floor(v.max(0.0));
    let index = if value >= mean {
        len - 1
    } else {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "value is a whole number in [0, len) after the guards above"
        )]
        let index = value as usize;
        index
    };

    index.min(len - 1)
}

/// A standard normal value from one Box-Muller transform.
fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    // `1 - u` lies in (0, 1], so the logarithm is finite.
    let radius = libm::sqrt(-2.0 * libm::log(1.0 - unit_f64(rng)));

    radius * libm::cos(2.0 * core::f64::consts::PI * unit_f64(rng))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::SeedableRng;

    #[test]
    fn half_normal_indices_stay_in_range_and_favor_the_newest() {
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(1);
        let mut counts = [0_u32; 10];

        for _ in 0..10_000 {
            counts[half_normal_index(10, &mut rng)] += 1;
        }

        assert!(counts[9] > counts[5] && counts[5] > counts[0], "{counts:?}");
        assert_eq!(half_normal_index(1, &mut rng), 0);
    }

    #[test]
    fn runs_split_at_cell_changes() {
        let coords = [[0], [0], [1], [1], [1], [0]];

        assert_eq!(run_end(&coords, 0), 2);
        assert_eq!(run_end(&coords, 2), 5);
        assert_eq!(run_end(&coords, 5), 6);
        assert_eq!(cell_index(-0.5), -1);
        assert_eq!(cell_index(f64::NAN), 0);
    }
}
