//! Integration tests for batched validity checks, shared roadmap queries and (feature
//! `parallel`) their multi-threaded versions.

use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;
use sampling_planning::{
    Euclidean, GoalRegion, UniformBox,
    roadmaps::{Prm, PrmConfig, QueryWorkspace},
};

use crate::world::slalom;

/// A PRM* roadmap over the slalom's world with `checker`, grown from `seed`.
fn roadmap<C>(checker: C, lazy: bool, seed: u64) -> Prm<Euclidean<2>, UniformBox<2>, C>
where
    C: sampling_planning::ValidityChecker<Euclidean<2>>,
{
    let mut config = PrmConfig::star(1.5 * PrmConfig::gamma_lower_bound(2, 100.0));
    config.lazy = lazy;
    let world = slalom();
    let mut prm = Prm::new(world.space, world.sampler, checker, config).expect("valid config");
    prm.grow(250, &mut Pcg64Mcg::seed_from_u64(seed));

    prm
}

/// Query pairs over the slalom's free space.
fn queries() -> Vec<([f64; 2], GoalRegion<[f64; 2]>)> {
    let starts = [[1.0, 1.0], [2.0, 9.0], [9.5, 5.0]];
    let goals = [[9.0, 9.0], [5.5, 9.0]];

    starts
        .iter()
        .flat_map(|&start| {
            goals
                .iter()
                .map(move |&goal| (start, GoalRegion::new(goal, 0.5)))
        })
        .collect()
}

#[test]
fn shared_queries_equal_mutable_queries_on_eager_and_lazy_roadmaps() {
    for lazy in [false, true] {
        let mut prm = roadmap(slalom().checker, lazy, 3);
        let shared: Vec<_> = queries()
            .iter()
            .map(|(start, goal)| prm.query_shared(start, goal).expect("valid start"))
            .collect();
        let mutable: Vec<_> = queries()
            .iter()
            .map(|(start, goal)| prm.query(start, goal).expect("valid start"))
            .collect();

        assert_eq!(shared, mutable, "lazy = {lazy}");
        assert!(shared.iter().filter(|path| path.is_some()).count() >= 4);
    }
}

#[test]
fn one_workspace_answers_many_queries_like_fresh_ones() {
    for lazy in [false, true] {
        let prm = roadmap(slalom().checker, lazy, 4);
        let mut workspace = QueryWorkspace::default();

        for (start, goal) in queries() {
            assert_eq!(
                prm.query_with(&mut workspace, &start, &goal),
                prm.query_shared(&start, &goal),
                "lazy = {lazy}"
            );
        }
    }
}

#[cfg(feature = "parallel")]
mod parallel {
    use core::num::NonZeroUsize;

    use sampling_planning::ParallelChecker;

    use rand_core::SeedableRng;
    use rand_pcg::Pcg64Mcg;
    use sampling_planning::{
        KdTree, PathLength,
        trees::{RrtStar, RrtStarConfig, grow_forest},
    };

    use super::{queries, roadmap};
    use crate::world::{assert_solves, slalom};

    /// Four threads.
    fn four() -> NonZeroUsize {
        NonZeroUsize::new(4).expect("non-zero")
    }

    #[test]
    fn a_parallel_checker_builds_the_same_roadmap() {
        let sequential = roadmap(slalom().checker, false, 5);
        let parallel = roadmap(ParallelChecker::new(slalom().checker, four(), 1), false, 5);

        assert_eq!(sequential.node_count(), parallel.node_count());
        assert_eq!(sequential.edge_count(), parallel.edge_count());

        for (start, goal) in queries() {
            assert_eq!(
                sequential.query_shared(&start, &goal),
                parallel.query_shared(&start, &goal)
            );
        }
    }

    #[test]
    fn batched_queries_answer_in_input_order() {
        for lazy in [false, true] {
            let prm = roadmap(slalom().checker, lazy, 7);
            let expected: Vec<_> = queries()
                .iter()
                .map(|(start, goal)| prm.query_shared(start, goal))
                .collect();

            for threads in [1, 3, 64] {
                let threads = NonZeroUsize::new(threads).expect("non-zero");
                assert_eq!(prm.query_batch(&queries(), threads), expected);
            }

            assert_eq!(prm.query_batch(&[], four()), Vec::new());
        }
    }

    #[test]
    fn a_merged_forest_solves_and_beats_its_first_tree() {
        let config = RrtStarConfig::new(1.5 * RrtStarConfig::gamma_lower_bound(2, 100.0), 1.0);
        let rngs = || (0..4).map(Pcg64Mcg::seed_from_u64).collect::<Vec<_>>();
        let forest = |rngs: &mut [Pcg64Mcg]| {
            grow_forest(
                &slalom(),
                config,
                &KdTree::default(),
                &PathLength,
                rngs,
                1500,
            )
            .expect("valid problem")
        };

        let merged = forest(&mut rngs());
        let mut alone = RrtStar::with_nearest_neighbors(slalom(), config, KdTree::default())
            .expect("valid problem");
        alone.run(1500, &mut Pcg64Mcg::seed_from_u64(0));

        let path = merged.best_path().expect("the forest solves the slalom");
        assert_solves(&slalom(), &path);
        assert!(merged.node_count() > 3 * alone.node_count());
        assert!(path.cost <= alone.best_path().map_or(f64::INFINITY, |best| best.cost));
        assert_eq!(
            forest(&mut rngs()).best_path(),
            Some(path),
            "independent of scheduling"
        );
    }
}
