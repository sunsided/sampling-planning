//! SST and SST*: stable sparse RRT (Li, Littlefield and Bekris 2016).

use alloc::vec::Vec;

use rand_core::Rng;

use super::{Error, KinodynamicProblem, Propagator, Trajectory};
use crate::radius::count_to_f64;
use crate::{LinearScan, RemovableNeighbors, Sampler, StateSpace, ValidityChecker, unit_f64};

/// The SST* schedule: SST runs in epochs, and after each epoch both radii shrink.
///
/// Epoch `0` lasts `first_epoch` iterations, epoch `j >= 1` lasts
/// `(1 + ln j) * shrink^(-(d + l + 1) j) * first_epoch` iterations, where `d` is the state and
/// `l` the control dimension. After epoch `j` both radii are multiplied by `shrink` (Li et al.
/// 2016, Algorithm 7). With the radii shrinking toward zero, SST* is asymptotically optimal.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SstStarSchedule {
    /// The radius factor per epoch, `xi` in the paper. In `(0, 1)`.
    pub shrink: f64,
    /// The length of the first epoch in iterations, `K_0` in the paper. At least one.
    pub first_epoch: usize,
}

/// Settings for [`Sst`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SstConfig {
    /// Probability in `[0, 1]` of growing toward the goal center instead of a random sample.
    pub goal_bias: f64,
    /// `delta_BN`: the cheapest active node within this distance of a sample is expanded; the
    /// nearest active node if there is none. Positive and finite.
    pub selection_radius: f64,
    /// `delta_s`: the witness radius. Of all nodes near one witness only the cheapest stays
    /// active. Positive and finite; usually smaller than `selection_radius`.
    pub pruning_radius: f64,
    /// The integration step: every propagation is a sequence of steps of this duration, each
    /// checked with [`ValidityChecker::check_motion`]. Positive and finite.
    pub step_size: f64,
    /// The most integration steps per propagation; the step count is uniform in
    /// `1..=max_steps`. At least one.
    pub max_steps: usize,
    /// Turns SST into SST*; `None` keeps the radii fixed.
    pub star: Option<SstStarSchedule>,
}

impl SstConfig {
    /// Creates an SST config with a goal bias of 5 % and fixed radii.
    #[must_use]
    pub const fn new(
        selection_radius: f64,
        pruning_radius: f64,
        step_size: f64,
        max_steps: usize,
    ) -> Self {
        Self {
            goal_bias: 0.05,
            selection_radius,
            pruning_radius,
            step_size,
            max_steps,
            star: None,
        }
    }

    /// This config with the SST* schedule.
    #[must_use]
    pub const fn star(self, schedule: SstStarSchedule) -> Self {
        Self {
            star: Some(schedule),
            ..self
        }
    }

    /// Checks every field against its documented range.
    fn validate(&self) -> Result<(), Error> {
        let positive = |value: f64| value.is_finite() && value > 0.0;

        if !(0.0..=1.0).contains(&self.goal_bias) {
            return Err(Error::GoalBiasOutOfRange);
        }

        if !(positive(self.selection_radius) && positive(self.pruning_radius)) {
            return Err(Error::InvalidRadius);
        }

        if !positive(self.step_size) || self.max_steps == 0 {
            return Err(Error::InvalidPropagation);
        }

        match self.star {
            Some(s) if !(s.shrink > 0.0 && s.shrink < 1.0) || s.first_epoch == 0 => {
                Err(Error::InvalidSchedule)
            }
            _ => Ok(()),
        }
    }
}

/// One tree node.
#[derive(Debug, Clone)]
struct Node<S, U> {
    /// The state.
    state: S,
    /// The parent node; `None` only for the root.
    parent: Option<usize>,
    /// The control and duration from the parent; `None` only for the root.
    control: Option<(U, f64)>,
    /// The total control duration from the root.
    cost: f64,
}

/// A witness: a point that keeps only the cheapest node within the pruning radius active.
#[derive(Debug, Clone)]
struct Witness<S> {
    /// The witness point: the state of the node that created it.
    state: S,
    /// The node that represents this witness; always active.
    representative: usize,
}

/// Where a new node falls relative to the witnesses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WitnessSlot {
    /// Within the pruning radius of this existing witness.
    Existing(usize),
    /// Farther than the pruning radius from every witness; becomes a new witness.
    New,
}

/// Progress through the SST* epochs.
#[derive(Debug, Clone, Copy)]
struct Epoch {
    /// The current epoch index `j`.
    index: usize,
    /// The iteration at which the current epoch ends.
    end: usize,
}

/// SST: an RRT-like tree grown by random forward propagation, kept sparse by witnesses.
///
/// Each iteration picks the cheapest active node near a goal-biased sample, applies a random
/// control for a random number of integration steps, and keeps the new node only if it is the
/// cheapest node near its witness; the previous representative then becomes inactive and is
/// never expanded again (Li, Littlefield and Bekris 2016). SST is asymptotically near-optimal;
/// with an [`SstStarSchedule`] it becomes SST*, which is asymptotically optimal.
///
/// The cost is the total control duration. [`Sst::best_trajectory`] keeps the best solution
/// found so far, so its cost never increases.
///
/// shortcut: inactive nodes are only removed from the active index, not freed, so memory grows
/// like RRT's; free inactive leaves if memory matters.
///
/// # Examples
///
/// ```
/// use rand_core::{Rng, SeedableRng};
/// use sampling_planning::{
///     Euclidean, GoalRegion, UniformBox, Validity, ValidityChecker, unit_f64,
///     kinodynamic::{KinodynamicProblem, Propagator, Sst, SstConfig},
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
/// let mut sst = Sst::new(problem, SstConfig::new(1.0, 0.3, 0.1, 10)).expect("valid");
/// sst.run(5000, &mut rand_pcg::Pcg64Mcg::seed_from_u64(1));
///
/// let trajectory = sst.best_trajectory().expect("free space");
/// assert_eq!(trajectory.controls.len() + 1, trajectory.states.len());
/// ```
#[derive(Clone)]
pub struct Sst<Sp, Sa, C, P, N = LinearScan<<Sp as StateSpace>::State>>
where
    Sp: StateSpace,
    P: Propagator<Sp>,
{
    /// The problem being solved.
    problem: KinodynamicProblem<Sp, Sa, C, P>,
    /// The settings; the radii shrink here under SST*.
    config: SstConfig,
    /// All nodes ever added, indexed by node id; node `0` is the start.
    nodes: Vec<Node<Sp::State, P::Control>>,
    /// Index over the active nodes, by node id.
    active: N,
    /// All witnesses, indexed by witness id.
    witnesses: Vec<Witness<Sp::State>>,
    /// Index over the witness states, by witness id.
    witness_nn: N,
    /// The cheapest node in the goal region so far.
    best_goal: Option<usize>,
    /// The number of iterations run.
    iteration: usize,
    /// SST* epoch progress; unused without a schedule.
    epoch: Epoch,
    /// Reused buffer for neighbor queries.
    near: Vec<usize>,
}

impl<Sp, Sa, C, P> Sst<Sp, Sa, C, P>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    P: Propagator<Sp>,
{
    /// Creates a planner with [`LinearScan`] indices.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidStart`] if the start state is not valid.
    /// - [`Error::GoalBiasOutOfRange`], [`Error::InvalidRadius`],
    ///   [`Error::InvalidPropagation`] or [`Error::InvalidSchedule`] if a config field is out of
    ///   its documented range.
    pub fn new(
        problem: KinodynamicProblem<Sp, Sa, C, P>,
        config: SstConfig,
    ) -> Result<Self, Error> {
        Self::with_nearest_neighbors(problem, config, LinearScan::default())
    }
}

impl<Sp, Sa, C, P, N> Sst<Sp, Sa, C, P, N>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    P: Propagator<Sp>,
    N: RemovableNeighbors<Sp> + Clone,
{
    /// Creates a planner; the active-node and witness indices are clones of the given empty
    /// index.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidStart`] if the start state is not valid.
    /// - [`Error::GoalBiasOutOfRange`], [`Error::InvalidRadius`],
    ///   [`Error::InvalidPropagation`] or [`Error::InvalidSchedule`] if a config field is out of
    ///   its documented range.
    pub fn with_nearest_neighbors(
        problem: KinodynamicProblem<Sp, Sa, C, P>,
        config: SstConfig,
        nn: N,
    ) -> Result<Self, Error> {
        config.validate()?;

        let validity = problem.checker.check_state(&problem.start);
        if !validity.is_valid() {
            return Err(Error::InvalidStart(validity));
        }

        let start = problem.start.clone();
        let mut active = nn.clone();
        let mut witness_nn = nn;
        active.insert(0, start.clone());
        witness_nn.insert(0, start.clone());

        Ok(Self {
            best_goal: problem.goal.contains(&problem.space, &start).then_some(0),
            nodes: alloc::vec![Node {
                state: start.clone(),
                parent: None,
                control: None,
                cost: 0.0,
            }],
            problem,
            active,
            witnesses: alloc::vec![Witness {
                state: start.clone(),
                representative: 0,
            }],
            witness_nn,
            iteration: 0,
            epoch: Epoch {
                index: 0,
                end: config.star.map_or(usize::MAX, |s| s.first_epoch),
            },
            config,
            near: Vec::new(),
        })
    }

    /// Runs one iteration. Returns whether a node was added.
    pub fn step<R: Rng + ?Sized>(&mut self, rng: &mut R) -> bool {
        self.advance_schedule();
        self.iteration += 1;

        let target = self.draw(rng);
        let parent = self.select(&target);
        let Some((state, control, duration)) = self.propagate(parent, rng) else {
            return false;
        };

        let cost = self.nodes[parent].cost + duration;
        let Some(slot) = self.witness_slot(&state, cost) else {
            return false;
        };

        self.add(state, parent, (control, duration), cost, slot);

        true
    }

    /// Runs `iterations` iterations.
    pub fn run<R: Rng + ?Sized>(&mut self, iterations: usize, rng: &mut R) {
        for _ in 0..iterations {
            self.step(rng);
        }
    }

    /// Ends the current SST* epoch when it is over: shrinks both radii and schedules the next.
    fn advance_schedule(&mut self) {
        let Some(schedule) = self.config.star else {
            return;
        };

        if self.iteration < self.epoch.end {
            return;
        }

        self.config.selection_radius *= schedule.shrink;
        self.config.pruning_radius *= schedule.shrink;
        self.epoch.index += 1;
        self.epoch.end = self
            .epoch
            .end
            .saturating_add(self.epoch_length(schedule, self.epoch.index));
    }

    /// The length of epoch `j >= 1`: `(1 + ln j) * shrink^(-(d + l + 1) j) * first_epoch`.
    fn epoch_length(&self, schedule: SstStarSchedule, j: usize) -> usize {
        let degrees = self.problem.space.dimension() + self.problem.propagator.control_dimension();
        let exponent = -count_to_f64(degrees + 1) * count_to_f64(j);
        let length = (1.0 + libm::log(count_to_f64(j)))
            * libm::pow(schedule.shrink, exponent)
            * count_to_f64(schedule.first_epoch);

        saturating_count(length)
    }

    /// Draws the goal center with probability `goal_bias`, otherwise a sampler state.
    fn draw<R: Rng + ?Sized>(&mut self, rng: &mut R) -> Sp::State {
        if unit_f64(rng) < self.config.goal_bias {
            self.problem.goal.center().clone()
        } else {
            self.problem.sampler.sample(rng)
        }
    }

    /// Best-first selection: the cheapest active node within the selection radius of
    /// `target`, lowest id on ties, or the nearest active node if there is none.
    fn select(&mut self, target: &Sp::State) -> usize {
        let space = &self.problem.space;
        self.active
            .within_radius(space, target, self.config.selection_radius, &mut self.near);

        let mut best: Option<usize> = None;
        for &id in &self.near {
            if best.is_none_or(|b| self.nodes[id].cost < self.nodes[b].cost) {
                best = Some(id);
            }
        }

        best.unwrap_or_else(|| {
            self.active
                .nearest(space, target)
                .expect("every witness keeps one node active")
        })
    }

    /// Applies a random control for a random number of steps from node `from`, checking every
    /// step. Returns the reached state, the control and the duration, or `None` if a step was
    /// not valid.
    fn propagate<R: Rng + ?Sized>(
        &mut self,
        from: usize,
        rng: &mut R,
    ) -> Option<(Sp::State, P::Control, f64)> {
        let control = self.problem.propagator.sample_control(rng);
        let steps = random_step_count(self.config.max_steps, rng);
        let (space, propagator) = (&self.problem.space, &self.problem.propagator);
        let mut state = self.nodes[from].state.clone();

        for _ in 0..steps {
            let next = propagator.propagate(space, &state, &control, self.config.step_size);

            if !self
                .problem
                .checker
                .check_motion(space, &state, &next)
                .is_valid()
            {
                return None;
            }

            state = next;
        }

        let duration = count_to_f64(steps) * self.config.step_size;

        Some((state, control, duration))
    }

    /// The witness a new node at `state` with `cost` would represent, or `None` if that
    /// witness already has a node that is at least as cheap.
    fn witness_slot(&self, state: &Sp::State, cost: f64) -> Option<WitnessSlot> {
        let space = &self.problem.space;
        let witness = self
            .witness_nn
            .nearest(space, state)
            .expect("the start is always a witness");
        let Witness {
            state: point,
            representative,
        } = &self.witnesses[witness];

        if space.distance(state, point) > self.config.pruning_radius {
            return Some(WitnessSlot::New);
        }

        (cost < self.nodes[*representative].cost).then_some(WitnessSlot::Existing(witness))
    }

    /// Adds the node, makes it the representative of its witness and deactivates the previous
    /// representative.
    fn add(
        &mut self,
        state: Sp::State,
        parent: usize,
        control: (P::Control, f64),
        cost: f64,
        slot: WitnessSlot,
    ) {
        let id = self.nodes.len();

        match slot {
            WitnessSlot::Existing(witness) => {
                let previous = self.witnesses[witness].representative;
                self.active.remove(previous);
                self.witnesses[witness].representative = id;
            }
            WitnessSlot::New => {
                self.witness_nn.insert(self.witnesses.len(), state.clone());
                self.witnesses.push(Witness {
                    state: state.clone(),
                    representative: id,
                });
            }
        }

        if self.problem.goal.contains(&self.problem.space, &state)
            && self
                .best_goal
                .is_none_or(|best| cost < self.nodes[best].cost)
        {
            self.best_goal = Some(id);
        }

        self.active.insert(id, state.clone());
        self.nodes.push(Node {
            state,
            parent: Some(parent),
            control: Some(control),
            cost,
        });
    }

    /// The cheapest trajectory into the goal region found so far, or `None` if the goal was not
    /// reached yet. Its cost never increases as iterations run.
    #[must_use]
    pub fn best_trajectory(&self) -> Option<Trajectory<Sp::State, P::Control>> {
        let goal = self.best_goal?;
        let mut states = Vec::new();
        let mut controls = Vec::new();
        let mut current = Some(goal);

        while let Some(id) = current {
            let node = &self.nodes[id];
            states.push(node.state.clone());
            controls.extend(node.control.clone());
            current = node.parent;
        }

        states.reverse();
        controls.reverse();

        Some(Trajectory {
            states,
            controls,
            cost: self.nodes[goal].cost,
        })
    }

    /// The number of nodes ever added, including inactive ones and the root.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// The number of active nodes, which equals the number of witnesses.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active.len()
    }

    /// The current selection and pruning radii; they shrink over time under SST*.
    #[must_use]
    pub const fn radii(&self) -> (f64, f64) {
        (self.config.selection_radius, self.config.pruning_radius)
    }
}

/// A step count uniform in `1..=max_steps`.
fn random_step_count<R: Rng + ?Sized>(max_steps: usize, rng: &mut R) -> usize {
    let max = u64::try_from(max_steps).unwrap_or(u64::MAX);
    let offset = usize::try_from(rng.next_u64() % max).expect("below max_steps");

    // shortcut: modulo bias is below max_steps / 2^64, irrelevant for any realistic step count.
    offset + 1
}

/// Converts a non-negative float count to `usize`, saturating; NaN becomes zero.
fn saturating_count(value: f64) -> usize {
    if value.is_nan() || value < 1.0 {
        0
    } else if value >= f64::from(u32::MAX) {
        usize::try_from(u32::MAX).unwrap_or(usize::MAX)
    } else {
        #[expect(
            clippy::cast_possible_truncation,
            clippy::cast_sign_loss,
            reason = "value is in [1, u32::MAX) after the guards above"
        )]
        let count = value as u32;
        usize::try_from(count).unwrap_or(usize::MAX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand_core::SeedableRng;

    #[test]
    fn step_counts_cover_the_whole_range() {
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(1);
        let mut seen = [false; 5];

        for _ in 0..200 {
            let steps = random_step_count(5, &mut rng);
            assert!((1..=5).contains(&steps));
            seen[steps - 1] = true;
        }

        assert_eq!(seen, [true; 5]);
    }

    #[test]
    fn saturating_count_clamps() {
        assert_eq!(saturating_count(f64::NAN), 0);
        assert_eq!(saturating_count(0.5), 0);
        assert_eq!(saturating_count(7.9), 7);
        assert_eq!(
            saturating_count(1e30),
            usize::try_from(u32::MAX).expect("fits")
        );
    }
}
