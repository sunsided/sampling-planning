//! Parallel tree growth (feature `parallel`): a forest of RRT* trees grown on several
//! threads, then merged into one tree.

use alloc::vec::Vec;

use rand_core::Rng;

use super::{Error, RrtStar, RrtStarConfig};
use crate::{
    CostObjective, NearestNeighbors, Problem, Sampler, StateSpace, Steering, ValidityChecker,
};

/// Grows one RRT* tree per RNG in `rngs` on its own thread for `iterations` iterations, then
/// merges the forest into the first tree and returns it.
///
/// Every tree starts at the problem's start and grows from clones of `problem`, the empty
/// index `nn` and `objective` with its own RNG, so the trees explore independently. The merge adds the
/// states of the other trees to the first one in order (tree by tree, each in insertion
/// order), each connected like a new RRT* node to its cheapest neighbor with a valid motion
/// and then rewiring its neighbors; states without such a neighbor are dropped. The result
/// depends only on the RNGs, never on thread scheduling, and its best path costs at most as
/// much as the first tree's. With `n` RNGs the work of `n * iterations` iterations runs in
/// about the time of `iterations` plus the merge, which costs one RRT* insertion per merged
/// state.
///
/// This is the "forest merged later" form of parallel tree growth: threads never share a
/// tree, so no locking is needed and runs stay reproducible.
///
/// # Errors
///
/// The same as [`RrtStar::with_objective`]; checked once, before any thread starts.
///
/// # Panics
///
/// Panics if `rngs` is empty, or re-raises the panic of a thread.
///
/// # Examples
///
/// ```
/// use rand_core::SeedableRng;
/// use sampling_planning::{
///     Euclidean, GoalRegion, LinearScan, PathLength, Problem, StraightLine, UniformBox, Validity,
///     ValidityChecker,
///     trees::{RrtStarConfig, grow_forest},
/// };
///
/// /// Free space everywhere.
/// #[derive(Clone)]
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
/// let problem = Problem {
///     space: Euclidean::<2>,
///     sampler: UniformBox::new([0.0, 0.0], [10.0, 10.0]),
///     checker: Free,
///     steering: StraightLine::new(1.0),
///     start: [1.0, 1.0],
///     goal: GoalRegion::new([9.0, 9.0], 0.5),
/// };
/// let config = RrtStarConfig::new(1.5 * RrtStarConfig::gamma_lower_bound(2, 100.0), 1.0);
/// let mut rngs: Vec<_> = (0..4).map(rand_pcg::Pcg64Mcg::seed_from_u64).collect();
/// let tree = grow_forest(&problem, config, &LinearScan::default(), &PathLength, &mut rngs, 500)
///     .expect("valid problem");
///
/// assert!(tree.best_path().expect("free space").cost < 12.0);
/// ```
pub fn grow_forest<Sp, Sa, C, St, N, O, R>(
    problem: &Problem<Sp, Sa, C, St>,
    config: RrtStarConfig,
    nn: &N,
    objective: &O,
    rngs: &mut [R],
    iterations: usize,
) -> Result<RrtStar<Sp, Sa, C, St, N, O>, Error>
where
    Sp: StateSpace + Clone + Send,
    Sp::State: Send,
    Sa: Sampler<Sp::State> + Clone + Send,
    C: ValidityChecker<Sp> + Clone + Send,
    St: Steering<Sp> + Clone + Send,
    N: NearestNeighbors<Sp> + Clone + Send,
    O: CostObjective<Sp> + Clone + Send,
    R: Rng + Send,
{
    assert!(!rngs.is_empty(), "a forest needs at least one RNG");

    let planter =
        || RrtStar::with_objective(problem.clone(), config, nn.clone(), objective.clone());
    planter()?;

    let mut trees: Vec<RrtStar<Sp, Sa, C, St, N, O>> = std::thread::scope(|scope| {
        let handles: Vec<_> = rngs
            .iter_mut()
            .map(|rng| {
                let mut tree = planter().expect("validated above");
                scope.spawn(move || {
                    tree.run(iterations, rng);
                    tree
                })
            })
            .collect();

        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic))
            })
            .collect()
    });

    let mut merged = trees.remove(0);
    for tree in trees {
        // The root is the start, already in the merged tree.
        merged.absorb(tree.tree_states().into_iter().skip(1));
    }

    Ok(merged)
}
