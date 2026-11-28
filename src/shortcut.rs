//! Path post-processing: shortcutting solution paths.
//!
//! Sampling-based planners return jagged paths. Both functions here replace pieces of a path
//! by direct motions that check as [`Validity::Valid`](crate::Validity::Valid), so the result
//! is still a solution, and never return a path that is more expensive than their input.
//!
//! Shortcuts produce edges longer than one steering step. Both functions take a `max_edge`
//! bound for callers whose checkers or controllers rely on short edges (reachable-set checks,
//! for example); pass [`f64::INFINITY`] to allow any length.

use alloc::vec::Vec;

use rand_core::Rng;

use crate::{Path, StateSpace, ValidityChecker, unit_f64};

/// Greedy vertex shortcutting: from each kept state, jumps to the furthest later state that a
/// direct, valid motion of length at most `max_edge` reaches.
///
/// Deterministic, keeps the first and last state, and only drops states. Each step tries the
/// candidates from the end of the path backward, so a path of `m` states costs `O(m^2)`
/// motion checks in the worst case and `O(m)` when long shortcuts exist.
///
/// Shortcutting prices paths by length under [`StateSpace::distance`] and compares with
/// [`Path::cost`]. A path from a planner with another
/// [`CostObjective`](crate::CostObjective) carries that objective's cost, so it must not be
/// shortcut as is: a shortcut can cut through high-risk regions.
///
/// # Examples
///
/// ```
/// use sampling_planning::{Euclidean, Path, Validity, ValidityChecker, shortcut};
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
/// let zigzag = Path {
///     states: vec![[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]],
///     cost: 2.0 * 2.0_f64.sqrt(),
/// };
/// let straight = shortcut(&Euclidean::<2>, &Free, &zigzag, f64::INFINITY);
///
/// assert_eq!(straight.states, [[0.0, 0.0], [2.0, 0.0]]);
/// assert!((straight.cost - 2.0).abs() < 1e-12);
/// ```
pub fn shortcut<Sp, C>(
    space: &Sp,
    checker: &C,
    path: &Path<Sp::State>,
    max_edge: f64,
) -> Path<Sp::State>
where
    Sp: StateSpace,
    C: ValidityChecker<Sp>,
{
    let states = &path.states;
    let mut kept = Vec::new();
    let mut current = 0;

    while current + 1 < states.len() {
        kept.push(states[current].clone());
        current = furthest_reachable(space, checker, states, current, max_edge);
    }

    kept.extend(states.last().cloned());

    cheaper(space, kept, path)
}

/// The index of the furthest state after `from` that a direct valid motion of length at most
/// `max_edge` reaches; `from + 1` (the existing edge) if no longer jump works.
fn furthest_reachable<Sp, C>(
    space: &Sp,
    checker: &C,
    states: &[Sp::State],
    from: usize,
    max_edge: f64,
) -> usize
where
    Sp: StateSpace,
    C: ValidityChecker<Sp>,
{
    (from + 2..states.len())
        .rev()
        .find(|&to| is_direct_motion_allowed(space, checker, &states[from], &states[to], max_edge))
        .unwrap_or(from + 1)
}

/// Whether the direct motion from `from` to `to` is at most `max_edge` long and valid.
fn is_direct_motion_allowed<Sp, C>(
    space: &Sp,
    checker: &C,
    from: &Sp::State,
    to: &Sp::State,
    max_edge: f64,
) -> bool
where
    Sp: StateSpace,
    C: ValidityChecker<Sp>,
{
    space.distance(from, to) <= max_edge && checker.check_motion(space, from, to).is_valid()
}

/// Randomized shortcutting: `attempts` times, picks two random points along the path and
/// replaces the piece between them by a direct motion if that is valid, at most `max_edge`
/// long and strictly cheaper.
///
/// Unlike [`shortcut`] the endpoints of a shortcut lie anywhere on the path, not only at its
/// states, so it also straightens corners. All three motions of a splice are checked: from
/// the state before the cut to its first point, across the cut, and from its second point to
/// the state after it. Run [`shortcut`] afterward to drop states that became redundant.
///
/// Shortcutting prices paths by length under [`StateSpace::distance`] and compares with
/// [`Path::cost`]. A path from a planner with another
/// [`CostObjective`](crate::CostObjective) carries that objective's cost, so it must not be
/// shortcut as is: a shortcut can cut through high-risk regions.
///
/// # Examples
///
/// ```
/// use rand_core::SeedableRng;
/// use sampling_planning::{Euclidean, Path, Validity, ValidityChecker, shortcut_random};
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
/// let corner = Path { states: vec![[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]], cost: 4.0 };
/// let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(1);
/// let smoothed = shortcut_random(&Euclidean::<2>, &Free, &corner, f64::INFINITY, 50, &mut rng);
///
/// assert!(smoothed.cost < 3.0);
/// ```
pub fn shortcut_random<Sp, C, R>(
    space: &Sp,
    checker: &C,
    path: &Path<Sp::State>,
    max_edge: f64,
    attempts: usize,
    rng: &mut R,
) -> Path<Sp::State>
where
    Sp: StateSpace,
    C: ValidityChecker<Sp>,
    R: Rng + ?Sized,
{
    let mut states = path.states.clone();

    for _ in 0..attempts {
        if states.len() < 3 {
            break;
        }

        let cut = RandomCut::draw(space, &states, rng);
        if let Some(spliced) = cut.splice(space, checker, &states, max_edge) {
            states = spliced;
        }
    }

    cheaper(space, states, path)
}

/// A candidate shortcut between two points on a path, given as segment index and fraction.
struct RandomCut {
    /// The segment `(states[i], states[i + 1])` holding the first point.
    first_segment: usize,
    /// Fraction in `[0, 1)` along the first segment.
    first_t: f64,
    /// The segment holding the second point; at least `first_segment`.
    second_segment: usize,
    /// Fraction in `[0, 1)` along the second segment.
    second_t: f64,
}

impl RandomCut {
    /// Draws two points uniformly by arc length along the path `states` (at least 2 states).
    fn draw<Sp: StateSpace, R: Rng + ?Sized>(
        space: &Sp,
        states: &[Sp::State],
        rng: &mut R,
    ) -> Self {
        let lengths: Vec<f64> = states
            .windows(2)
            .map(|pair| space.distance(&pair[0], &pair[1]))
            .collect();
        let total: f64 = lengths.iter().sum();
        let (a, b) = (unit_f64(rng) * total, unit_f64(rng) * total);
        let ((first_segment, first_t), (second_segment, second_t)) =
            (locate(&lengths, a.min(b)), locate(&lengths, a.max(b)));

        Self {
            first_segment,
            first_t,
            second_segment,
            second_t,
        }
    }

    /// The path with the piece between the two points replaced by a direct motion, or `None`
    /// if the cut is invalid, too long, within one segment or not strictly cheaper.
    fn splice<Sp, C>(
        &self,
        space: &Sp,
        checker: &C,
        states: &[Sp::State],
        max_edge: f64,
    ) -> Option<Vec<Sp::State>>
    where
        Sp: StateSpace,
        C: ValidityChecker<Sp>,
    {
        if self.first_segment >= self.second_segment {
            return None;
        }

        let point =
            |segment: usize, t: f64| space.interpolate(&states[segment], &states[segment + 1], t);
        let from = point(self.first_segment, self.first_t);
        let to = point(self.second_segment, self.second_t);

        let old = space.distance(&from, &states[self.first_segment + 1])
            + path_length(space, &states[self.first_segment + 1..=self.second_segment])
            + space.distance(&states[self.second_segment], &to);
        if space.distance(&from, &to) >= old {
            return None;
        }

        // The stubs are pieces of checked motions, but a sampled or reachable-set check of a
        // whole motion proves nothing about a piece of it, so they are checked on their own.
        let before = &states[self.first_segment];
        let after = &states[self.second_segment + 1];
        let allowed = checker.check_motion(space, before, &from).is_valid()
            && is_direct_motion_allowed(space, checker, &from, &to, max_edge)
            && checker.check_motion(space, &to, after).is_valid();

        if !allowed {
            return None;
        }

        let mut spliced = states[..=self.first_segment].to_vec();
        spliced.extend([from, to]);
        spliced.extend_from_slice(&states[self.second_segment + 1..]);

        Some(spliced)
    }
}

/// The segment index and the fraction along it at arc length `s` of a path with the given
/// segment lengths.
fn locate(lengths: &[f64], s: f64) -> (usize, f64) {
    let mut remaining = s;

    for (segment, &length) in lengths.iter().enumerate() {
        if remaining < length {
            return (segment, remaining / length);
        }

        remaining -= length;
    }

    (lengths.len() - 1, 1.0)
}

/// The sum of the distances between consecutive `states`.
fn path_length<Sp: StateSpace>(space: &Sp, states: &[Sp::State]) -> f64 {
    states
        .windows(2)
        .map(|pair| space.distance(&pair[0], &pair[1]))
        .sum()
}

/// `states` as a path, unless rounding made it more expensive than `original`.
fn cheaper<Sp: StateSpace>(
    space: &Sp,
    states: Vec<Sp::State>,
    original: &Path<Sp::State>,
) -> Path<Sp::State> {
    let cost = path_length(space, &states);

    if cost <= original.cost {
        Path { states, cost }
    } else {
        original.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Euclidean, Validity, discretized_motion};
    use rand_core::SeedableRng;

    /// A wall at `x` in `[1.4, 1.6]` for `y < 1.5`.
    struct Wall;

    impl ValidityChecker<Euclidean<2>> for Wall {
        fn check_state(&self, s: &[f64; 2]) -> Validity {
            if (1.4..=1.6).contains(&s[0]) && s[1] < 1.5 {
                Validity::Invalid
            } else {
                Validity::Valid
            }
        }

        fn check_motion(&self, space: &Euclidean<2>, from: &[f64; 2], to: &[f64; 2]) -> Validity {
            discretized_motion(space, from, to, 0.01, |s| self.check_state(s))
        }
    }

    /// A path over the wall: up, across, down, in unit-ish steps.
    fn detour() -> Path<[f64; 2]> {
        let states = alloc::vec![
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 2.0],
            [1.5, 2.0],
            [3.0, 2.0],
            [3.0, 0.0]
        ];
        let cost = path_length(&Euclidean::<2>, &states);

        Path { states, cost }
    }

    /// Asserts that every segment of `path` is valid and at most `max_edge` long.
    fn assert_valid(path: &Path<[f64; 2]>, max_edge: f64) {
        for pair in path.states.windows(2) {
            assert_eq!(
                Wall.check_motion(&Euclidean::<2>, &pair[0], &pair[1]),
                Validity::Valid
            );
            assert!(Euclidean::<2>.distance(&pair[0], &pair[1]) <= max_edge);
        }
    }

    #[test]
    fn greedy_shortcut_keeps_endpoints_and_validity() {
        let path = detour();
        let short = shortcut(&Euclidean::<2>, &Wall, &path, f64::INFINITY);

        assert_eq!(short.states.first(), path.states.first());
        assert_eq!(short.states.last(), path.states.last());
        assert!(short.cost < path.cost);
        assert_valid(&short, f64::INFINITY);
        assert_eq!(short.states, [[0.0, 0.0], [1.5, 2.0], [3.0, 0.0]]);
    }

    #[test]
    fn greedy_shortcut_respects_max_edge() {
        let path = detour();
        let short = shortcut(&Euclidean::<2>, &Wall, &path, 2.0);

        assert!(short.cost <= path.cost);
        assert_eq!(short.states.len(), path.states.len() - 1);
        assert_valid(&short, 2.0);
    }

    #[test]
    fn random_shortcut_never_increases_cost() {
        let path = detour();
        let mut rng = rand_pcg::Pcg64Mcg::seed_from_u64(9);
        let smoothed = shortcut_random(&Euclidean::<2>, &Wall, &path, f64::INFINITY, 200, &mut rng);

        assert_eq!(smoothed.states.first(), path.states.first());
        assert_eq!(smoothed.states.last(), path.states.last());
        assert!(smoothed.cost < path.cost - 0.5, "{}", smoothed.cost);
        assert!((smoothed.cost - path_length(&Euclidean::<2>, &smoothed.states)).abs() < 1e-12);
        assert_valid(&smoothed, f64::INFINITY);
    }
}
