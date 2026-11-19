//! A 2D test world with box obstacles and boxes the checker cannot decide.

use sampling_planning::{
    Euclidean, GoalRegion, Path, Problem, StateSpace, StraightLine, UniformBox, Validity,
    ValidityChecker, discretized_motion,
};

/// The planning problem type used by all tests.
pub type World2d = Problem<Euclidean<2>, UniformBox<2>, Boxes, StraightLine>;

/// An axis-aligned box given by its lower and upper corners.
pub type Rect = ([f64; 2], [f64; 2]);

/// A checker with obstacle boxes (invalid) and fog boxes (unknown).
#[derive(Debug, Clone, Default)]
pub struct Boxes {
    /// Boxes whose states are [`Validity::Invalid`].
    pub obstacles: Vec<Rect>,
    /// Boxes whose states are [`Validity::Unknown`].
    pub fog: Vec<Rect>,
}

/// Whether `state` lies in the closed box `rect`.
fn inside(rect: &Rect, state: &[f64; 2]) -> bool {
    (0..2).all(|i| (rect.0[i]..=rect.1[i]).contains(&state[i]))
}

impl ValidityChecker<Euclidean<2>> for Boxes {
    fn check_state(&self, state: &[f64; 2]) -> Validity {
        if self.obstacles.iter().any(|rect| inside(rect, state)) {
            Validity::Invalid
        } else if self.fog.iter().any(|rect| inside(rect, state)) {
            Validity::Unknown
        } else {
            Validity::Valid
        }
    }

    fn check_motion(&self, space: &Euclidean<2>, from: &[f64; 2], to: &[f64; 2]) -> Validity {
        discretized_motion(space, from, to, 0.02, |state| self.check_state(state))
    }
}

/// A 10 x 10 world with two walls that force a slalom from (1, 1) to (9, 9).
pub fn slalom() -> World2d {
    problem(Boxes {
        obstacles: vec![([4.0, 0.0], [5.0, 7.0]), ([6.0, 3.0], [7.0, 10.0])],
        fog: Vec::new(),
    })
}

/// A 10 x 10 world around `checker`, from (1, 1) to within 0.5 of (9, 9).
pub fn problem(checker: Boxes) -> World2d {
    Problem {
        space: Euclidean::<2>,
        sampler: UniformBox::new([0.0, 0.0], [10.0, 10.0]),
        checker,
        steering: StraightLine::new(1.0),
        start: [1.0, 1.0],
        goal: GoalRegion::new([9.0, 9.0], 0.5),
    }
}

/// Asserts that `path` solves `world`: it starts at the start, ends in the goal, stays in the
/// sampling box, every segment checks as valid and is at most one steering step long, and its
/// cost is its length.
pub fn assert_solves(world: &World2d, path: &Path<[f64; 2]>) {
    let states = &path.states;
    assert_eq!(states.first(), Some(&world.start));
    assert!(
        world
            .goal
            .contains(&world.space, states.last().expect("non-empty"))
    );

    for state in states {
        assert!(world.sampler.contains(state), "{state:?} left the box");
    }

    let mut length = 0.0;
    for pair in states.windows(2) {
        let validity = world.checker.check_motion(&world.space, &pair[0], &pair[1]);
        assert_eq!(validity, Validity::Valid, "segment {pair:?}");
        assert!(world.space.distance(&pair[0], &pair[1]) <= world.steering.max_step() + 1e-9);
        length += world.space.distance(&pair[0], &pair[1]);
    }

    assert!(
        (length - path.cost).abs() < 1e-9,
        "cost {} vs length {length}",
        path.cost
    );
}
