//! Unit tests for AIT*: the reverse search against a Dijkstra reference.

use alloc::{collections::BinaryHeap, vec, vec::Vec};

use rand_core::SeedableRng;
use rand_pcg::Pcg64Mcg;

use super::{AitStar, AitStarConfig, Kind};
use crate::graph::Entry;
use crate::{
    Euclidean, GoalRegion, LinearScan, StateSpace, UniformBox, Validity, ValidityChecker,
    discretized_motion,
};

/// A wall at `x` in `[4, 5]` with a gap above `y = 8`.
struct Wall;

impl ValidityChecker<Euclidean<2>> for Wall {
    fn check_state(&self, state: &[f64; 2]) -> Validity {
        if (4.0..=5.0).contains(&state[0]) && state[1] < 8.0 {
            Validity::Invalid
        } else {
            Validity::Valid
        }
    }

    fn check_motion(&self, space: &Euclidean<2>, from: &[f64; 2], to: &[f64; 2]) -> Validity {
        discretized_motion(space, from, to, 0.05, |state| self.check_state(state))
    }
}

/// The planner under test.
type Planner = AitStar<Euclidean<2>, UniformBox<2>, Wall, LinearScan<[f64; 2]>>;

/// A radius config with small batches.
fn radius_config() -> AitStarConfig {
    AitStarConfig {
        batch_size: 40,
        ..AitStarConfig::new(1.5 * AitStarConfig::gamma_lower_bound(2, 100.0))
    }
}

/// A k-nearest config with small batches.
fn k_config() -> AitStarConfig {
    AitStarConfig {
        k_nearest: Some(1.5 * AitStarConfig::k_constant_lower_bound(2)),
        ..radius_config()
    }
}

/// AIT* from (1, 1) to (9, 1) around [`Wall`].
fn planner(config: AitStarConfig) -> Planner {
    AitStar::new(
        Euclidean::<2>,
        UniformBox::new([0.0, 0.0], [10.0, 10.0]),
        Wall,
        [1.0, 1.0],
        GoalRegion::new([9.0, 1.0], 0.5),
        config,
    )
    .expect("valid problem")
}

/// The exact cost-to-go of every node over the batch graph with `c_hat` costs, skipping the
/// motions known to be invalid: Dijkstra from the goal states along reversed motions.
fn dijkstra_to_goal(planner: &Planner) -> Vec<f64> {
    let nodes = &planner.nodes;
    let mut cost = vec![f64::INFINITY; nodes.len()];
    let mut heap = BinaryHeap::new();

    for (id, node) in nodes.iter().enumerate() {
        if node.kind != Kind::Pruned && node.in_goal() {
            cost[id] = 0.0;
            heap.push(Entry {
                cost: 0.0,
                node: id,
            });
        }
    }

    while let Some(Entry { cost: c, node: y }) = heap.pop() {
        if c > cost[y] {
            continue;
        }

        for &x in &nodes[y].adjacent {
            let through = c + planner.space.distance(&nodes[x].state, &nodes[y].state);

            if !nodes[x].invalid.contains(&y) && through < cost[x] {
                cost[x] = through;
                heap.push(Entry {
                    cost: through,
                    node: x,
                });
            }
        }
    }

    cost
}

/// Asserts that `h_con` matches the reference on every live node `settled` accepts, and
/// returns how many nodes were compared.
fn assert_matches(planner: &Planner, reference: &[f64], settled: impl Fn(usize) -> bool) -> usize {
    let mut compared = 0;

    for (id, node) in planner.nodes.iter().enumerate() {
        if node.kind == Kind::Pruned || !settled(id) {
            continue;
        }

        let (actual, expected) = (node.h_con, reference[id]);
        let equal = actual.total_cmp(&expected).is_eq() || (actual - expected).abs() < 1e-9;
        assert!(equal, "node {id}: h_con {actual} vs Dijkstra {expected}");
        compared += 1;
    }

    compared
}

/// Asserts that the batch graph is symmetric and holds every forward-tree edge.
fn assert_graph_is_symmetric_with_tree_edges(planner: &Planner) {
    for (id, node) in planner.nodes.iter().enumerate() {
        for &other in &node.adjacent {
            assert!(
                planner.nodes[other].adjacent.contains(&id),
                "{id} - {other}"
            );
        }

        if let Some(parent) = node.parent {
            assert!(
                node.adjacent.contains(&parent),
                "tree edge {parent} -> {id}"
            );
        }
    }
}

/// Asserts that the lazily settled states and, after draining the reverse queue, all live
/// states have the exact cost-to-go.
fn assert_reverse_search_is_exact(planner: &mut Planner) {
    assert_graph_is_symmetric_with_tree_edges(planner);

    let reference = dijkstra_to_goal(planner);
    let top = planner
        .peek_state()
        .map_or(f64::INFINITY, |entry| entry.key);
    let lazy = assert_matches(planner, &reference, |x| {
        planner.is_consistent(x) && planner.nodes[x].h_con + planner.nodes[x].g_hat < top
    });
    assert!(lazy > 0, "the lazy search settled some states");

    while let Some(entry) = planner.peek_state() {
        planner.expand_state(entry.vertex);
    }

    let live = planner
        .nodes
        .iter()
        .filter(|node| node.kind != Kind::Pruned)
        .count();
    assert_eq!(assert_matches(planner, &reference, |_| true), live);
    assert!(reference.iter().filter(|cost| cost.is_finite()).count() > 10);
}

/// Steps the planner until it has recorded `wanted` invalid edges, and returns whether some
/// `h_con` rose within a batch on the way (an underconsistent repair).
fn step_until_invalid_edges(planner: &mut Planner, wanted: usize) -> bool {
    let mut rng = Pcg64Mcg::seed_from_u64(3);
    let mut rose = false;

    for _ in 0..5000 {
        let before: Vec<f64> = planner.nodes.iter().map(|node| node.h_con).collect();
        let batch = planner.batch_count();
        planner.step(&mut rng);

        rose |= planner.batch_count() == batch
            && before
                .iter()
                .zip(&planner.nodes)
                .any(|(&old, node)| old.is_finite() && node.h_con > old);

        if invalid_edge_count(planner) >= wanted {
            break;
        }
    }

    rose
}

/// The number of motions recorded as invalid.
fn invalid_edge_count(planner: &Planner) -> usize {
    planner.nodes.iter().map(|node| node.invalid.len()).sum()
}

#[test]
fn reverse_search_matches_dijkstra_after_invalid_edges() {
    for config in [radius_config(), k_config()] {
        let mut planner = planner(config);
        let rose = step_until_invalid_edges(&mut planner, 5);

        assert!(invalid_edge_count(&planner) >= 5);
        assert!(rose, "an invalid edge raised some cost-to-go");
        assert_reverse_search_is_exact(&mut planner);
    }
}

#[test]
fn reverse_search_matches_dijkstra_in_later_batches() {
    for config in [radius_config(), k_config()] {
        let mut planner = planner(config);
        let mut rng = Pcg64Mcg::seed_from_u64(4);

        while planner.batch_count() < 4 {
            planner.step(&mut rng);
        }

        planner.step(&mut rng);
        assert!(
            planner.best_path().is_some(),
            "solved before the fourth batch"
        );
        assert!(planner.nodes.iter().any(|node| node.kind == Kind::Pruned));
        assert!(invalid_edge_count(&planner) > 0);
        assert_reverse_search_is_exact(&mut planner);
    }
}
