//! The dense graph of [`Spars`]: k-nearest PRM* nodes, their representatives and the
//! interface lists of every guard.

use alloc::{collections::BTreeSet, vec::Vec};

use super::Spars;
use crate::graph::{Arc, Route, shortest_path};
use crate::radius::k_nearest_count;
use crate::{NearestNeighbors, StateSpace, ValidityChecker};

impl<Sp, Sa, C, N> Spars<Sp, Sa, C, N>
where
    Sp: StateSpace,
    C: ValidityChecker<Sp>,
    N: NearestNeighbors<Sp>,
{
    /// Adds a valid state to the dense graph, connected to its `ceil(k_constant ln n)` nearest
    /// dense nodes by valid motions, and files it and its interface neighbors under their
    /// representatives (OMPL's `addMilestone`). Returns its dense id.
    pub(super) fn add_milestone(&mut self, state: Sp::State) -> usize {
        let m = self.dense_states.len();
        let k = k_nearest_count(self.k_constant, m + 1);
        let mut near = Vec::new();
        self.dense_nn.k_nearest(&self.space, &state, k, &mut near);
        near.retain(|&n| self.is_edge_valid(&state, &self.dense_states[n]));

        self.dense_states.push(state.clone());
        self.dense_adjacency.push(Vec::new());
        self.representatives.push(None);
        for n in near {
            self.dense_adjacency[m].push(n);
            self.dense_adjacency[n].push(m);
        }

        self.dense_nn.insert(m, state);
        self.calculate_representative(m);
        self.refile(m);

        for neighbor in self.interface_neighborhood(m) {
            self.refile(neighbor);
        }

        m
    }

    /// Sets the representative of dense node `q` to the closest guard (ties by id) visible
    /// within `sparse_delta`; leaves it unchanged if there is none (OMPL's
    /// `calculateRepresentative`).
    fn calculate_representative(&mut self, q: usize) {
        let state = &self.dense_states[q];
        let closest =
            self.graph
                .closest_visible(&self.space, state, self.config.sparse_delta, |guard| {
                    self.is_edge_valid(state, guard)
                });

        if let Some(guard) = closest {
            self.representatives[q] = Some(guard);
        }
    }

    /// Re-assigns the representatives of the dense nodes within `sparse_delta + dense_delta`
    /// of the new guard `guard` and files them again (OMPL's `updateRepresentatives`).
    pub(super) fn update_representatives(&mut self, guard: usize) {
        let radius = self.config.sparse_delta + self.config.dense_delta;
        let mut points = Vec::new();
        self.dense_nn
            .within_radius(&self.space, &self.graph.states[guard], radius, &mut points);

        for &q in &points {
            self.unfile(q);
            self.calculate_representative(q);
        }

        for q in points {
            self.refile(q);
        }
    }

    /// Removes dense node `q` from its representative's lists, then adds it to the interface
    /// list of each representative its interface neighbors have, or to the non-interface list
    /// if there are none (OMPL's `removeFromRepresentatives` then `addToRepresentatives`).
    fn refile(&mut self, q: usize) {
        self.unfile(q);
        let Some(rep) = self.representatives[q] else {
            return;
        };

        let others = self.interface_representatives(q);
        if others.is_empty() {
            self.non_interface_lists[rep].insert(q);
        }

        for other in others {
            self.interface_lists[rep]
                .entry(other)
                .or_default()
                .insert(q);
        }
    }

    /// Removes dense node `q` from every list of its current representative.
    fn unfile(&mut self, q: usize) {
        let Some(rep) = self.representatives[q] else {
            return;
        };

        self.non_interface_lists[rep].remove(&q);
        for list in self.interface_lists[rep].values_mut() {
            list.remove(&q);
        }
    }

    /// The representatives, other than `q`'s own, of the dense neighbors of `q` closer than
    /// `dense_delta` (OMPL's `getInterfaceNeighborRepresentatives`).
    fn interface_representatives(&self, q: usize) -> BTreeSet<usize> {
        self.interface_neighborhood(q)
            .into_iter()
            .filter_map(|n| self.representatives[n])
            .collect()
    }

    /// The dense neighbors of `q` closer than `dense_delta` whose representative differs from
    /// `q`'s, in edge order (OMPL's `getInterfaceNeighborhood`).
    pub(super) fn interface_neighborhood(&self, q: usize) -> Vec<usize> {
        let rep = self.representatives[q];

        self.dense_adjacency[q]
            .iter()
            .copied()
            .filter(|&n| {
                self.representatives[n] != rep
                    && self.dense_distance(q, n) < self.config.dense_delta
            })
            .collect()
    }

    /// The first dense neighbor of `q`, in edge order, within `dense_delta` and represented by
    /// `rep`, or `None` (where OMPL's `getInterfaceNeighbor` throws).
    pub(super) fn interface_neighbor(&self, q: usize, rep: usize) -> Option<usize> {
        self.dense_adjacency[q].iter().copied().find(|&n| {
            self.representatives[n] == Some(rep)
                && self.dense_distance(q, n) <= self.config.dense_delta
        })
    }

    /// The shortest dense path from `q` to a dense node on the interface of guard `v` with
    /// guard `other`, the lowest id among equally close ones, or `None` if there is no such
    /// node or none is reachable. A path of just `q` when `q` is on that interface.
    pub(super) fn closest_interface_path(&self, q: usize, v: usize, other: usize) -> Option<Route> {
        let targets = self.interface_lists[v].get(&other)?;

        if targets.is_empty() {
            return None;
        }

        let arcs = |node: usize, out: &mut Vec<Arc>| {
            out.clear();
            out.extend(self.dense_adjacency[node].iter().map(|&to| Arc {
                to,
                cost: self.dense_distance(node, to),
                edge: None,
            }));
        };

        shortest_path(self.dense_states.len(), q, |id| targets.contains(&id), arcs)
    }

    /// The distance between dense nodes `a` and `b`.
    fn dense_distance(&self, a: usize, b: usize) -> f64 {
        self.space
            .distance(&self.dense_states[a], &self.dense_states[b])
    }
}
