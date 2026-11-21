//! The quality criterion of [`Spars`]: the stretch test over dense paths (OMPL's
//! `checkAddPath`) and adding dense paths to the sparse graph.

use alloc::{collections::BTreeSet, vec::Vec};

use super::Spars;
use crate::graph::Route;
use crate::roadmaps::GuardKind;
use crate::{NearestNeighbors, Path, StateSpace, ValidityChecker, shortcut};

impl<Sp, Sa, C, N> Spars<Sp, Sa, C, N>
where
    Sp: StateSpace,
    C: ValidityChecker<Sp>,
    N: NearestNeighbors<Sp>,
{
    /// Quality: tests the stretch bound around the representative `v` of dense node `q` toward
    /// each representative `v'` of `q`'s interface neighbors, ascending, and adds a dense path
    /// at the first violation that can be repaired. Returns whether it added one.
    pub(super) fn check_add_path(&mut self, q: usize, interface: &[usize]) -> bool {
        let Some(v) = self.representatives[q] else {
            return false;
        };

        let others: BTreeSet<usize> = interface
            .iter()
            .filter_map(|&n| self.representatives[n])
            .collect();

        others
            .into_iter()
            .any(|v_prime| self.check_add_path_toward(q, v, v_prime))
    }

    /// The stretch test for one `v'`, with OMPL's nesting: for each candidate `v''` (outer),
    /// `s_max` is compared against `d_min`, the shortest dense path from `q` to the interface
    /// of `v` with any of the candidates `VPPs[0..=j]` (inner, accumulated over `j` and reset
    /// per outer candidate). The dense paths per candidate do not depend on the outer one and
    /// are computed once.
    fn check_add_path_toward(&mut self, q: usize, v: usize, v_prime: usize) -> bool {
        let candidates = self.graph.non_adjacent_neighbors(v, v_prime);
        let routes: Vec<Option<Route>> = candidates
            .iter()
            .map(|&vpp| self.closest_interface_path(q, v, vpp))
            .collect();

        for &outer in &candidates {
            let s_max = self.max_spanner_estimate(v, v_prime, outer);
            let mut best: Option<&Route> = None;

            for (route, &vpp) in routes.iter().zip(&candidates) {
                if route
                    .as_ref()
                    .is_some_and(|route| best.is_none_or(|b| route.cost < b.cost))
                {
                    best = route.as_ref();
                }

                let Some(best) = best else {
                    continue;
                };

                if s_max > self.config.stretch * best.cost
                    && self.add_quality_path(q, v_prime, best, vpp)
                {
                    return true;
                }
            }
        }

        false
    }

    /// `max over x of (d(x, v) + d(v, v')) / 2`, where `x` is `v''` or a neighbor of `v''`
    /// adjacent to `v` but not to `v'` on whose interface `v''` has dense nodes (OMPL's
    /// `computeX` and the `s_max` loop).
    fn max_spanner_estimate(&self, v: usize, v_prime: usize, vpp: usize) -> f64 {
        let states = &self.graph.states;
        let to_v_prime = self.space.distance(&states[v], &states[v_prime]);

        self.graph.adjacency[vpp]
            .iter()
            .copied()
            .filter(|&x| {
                self.graph.has_edge(x, v)
                    && !self.graph.has_edge(x, v_prime)
                    && self.interface_lists[vpp]
                        .get(&x)
                        .is_some_and(|list| !list.is_empty())
            })
            .chain([vpp])
            .map(|x| f64::midpoint(self.space.distance(&states[x], &states[v]), to_v_prime))
            .fold(0.0, f64::max)
    }

    /// Extends the dense `route` from `q` by the interface neighbor of `q` represented by `v'`
    /// in front and the interface neighbor of its end represented by `v''` behind, and adds it
    /// to the sparse graph between `v'` and `v''`. Returns `false`, changing nothing, if an
    /// interface neighbor is missing or a sparse edge would be invalid.
    fn add_quality_path(&mut self, q: usize, v_prime: usize, route: &Route, vpp: usize) -> bool {
        let end = *route.nodes.last().unwrap_or(&q);
        let (Some(first), Some(last)) = (
            self.interface_neighbor(q, v_prime),
            self.interface_neighbor(end, vpp),
        ) else {
            return false;
        };

        let states = core::iter::once(first)
            .chain(route.nodes.iter().copied())
            .chain([last])
            .map(|id| self.dense_states[id].clone())
            .collect();

        self.add_path_to_spanner(states, v_prime, vpp)
    }

    /// Adds `states` as a chain of quality guards from guard `from` to guard `to`, after
    /// dropping states the greedy [`shortcut()`](crate::shortcut()) skips (in place of OMPL's `reduceVertices`).
    /// With at most one state, connects `from` and `to` directly and resets the failure
    /// counter. Every sparse edge is checked first; returns `false`, changing nothing, if one
    /// is invalid.
    fn add_path_to_spanner(&mut self, states: Vec<Sp::State>, from: usize, to: usize) -> bool {
        if states.len() <= 1 {
            if !self.is_guard_edge_valid(from, to) {
                return false;
            }

            self.graph.connect(from, to);
            self.failures = 0;
            return true;
        }

        let states = self.reduce_vertices(states);
        if !self.is_chain_valid(from, &states, to) {
            return false;
        }

        let mut prior = from;
        for state in states {
            let guard = self.add_guard(state, GuardKind::Quality);
            self.graph.connect(prior, guard);
            prior = guard;
        }

        self.graph.connect(prior, to);

        true
    }

    /// `states` with the states dropped that [`shortcut()`](crate::shortcut()) skips; keeps the first and last.
    fn reduce_vertices(&self, states: Vec<Sp::State>) -> Vec<Sp::State> {
        let cost = states
            .windows(2)
            .map(|pair| self.space.distance(&pair[0], &pair[1]))
            .sum();
        let path = Path { states, cost };

        shortcut(&self.space, &self.checker, &path, f64::INFINITY).states
    }

    /// Whether every motion along guard `from`, `states`, guard `to` checks as valid.
    fn is_chain_valid(&self, from: usize, states: &[Sp::State], to: usize) -> bool {
        let chain: Vec<&Sp::State> = core::iter::once(&self.graph.states[from])
            .chain(states)
            .chain([&self.graph.states[to]])
            .collect();

        chain
            .windows(2)
            .all(|pair| self.is_edge_valid(pair[0], pair[1]))
    }
}
