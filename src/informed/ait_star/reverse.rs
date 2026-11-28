//! AIT*'s reverse search: Lifelong Planning A* from the goal states over the batch's graph,
//! with `c_hat` edge costs and no collision checks (Algorithms 5 and 6).
//!
//! In LPA* terms `h_con` is `rhs` and `h_exp` is `g`. A state is consistent when both agree;
//! it is overconsistent when `h_exp > h_con` (it got cheaper) and underconsistent when
//! `h_exp < h_con` (an invalid edge made it pricier).

use super::{AitStar, Kind, VertexEntry};
use crate::{NearestNeighbors, Sampler, StateSpace, ValidityChecker};

impl<Sp, Sa, C, N> AitStar<Sp, Sa, C, N>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    N: NearestNeighbors<Sp> + Clone,
{
    /// Forgets every cost-to-go and queues the goal states with `h_con = 0` (Algorithm 5,
    /// lines 1 to 6).
    pub(super) fn restart_reverse_search(&mut self) {
        self.reverse_queue.clear();

        for node in &mut self.graph.nodes {
            node.ext.h_con = f64::INFINITY;
            node.ext.h_exp = f64::INFINITY;
            node.ext.reverse_key = None;
        }

        for x in 0..self.graph.nodes.len() {
            if self.is_reverse_root(x) {
                self.graph.nodes[x].ext.h_con = 0.0;
                self.queue_state(x);
            }
        }
    }

    /// Whether `x` is a goal of the reverse search: a live state in the goal region.
    fn is_reverse_root(&self, x: usize) -> bool {
        self.graph.nodes[x].kind != Kind::Pruned && self.graph.nodes[x].in_goal()
    }

    /// Whether `x` has `h_con == h_exp`.
    pub(super) fn is_consistent(&self, x: usize) -> bool {
        self.graph.nodes[x]
            .ext
            .h_con
            .total_cmp(&self.graph.nodes[x].ext.h_exp)
            .is_eq()
    }

    /// The LPA* key of `x`: `(min(h_con, h_exp) + g_hat, min(h_con, h_exp))`.
    fn reverse_key(&self, x: usize) -> [f64; 2] {
        let node = &self.graph.nodes[x];
        let h = node.ext.h_con.min(node.ext.h_exp);

        [h + node.g_hat, h]
    }

    /// Queues `x` with its current key, unless it is queued with that key already.
    fn queue_state(&mut self, x: usize) {
        let key = self.reverse_key(x);

        if self.graph.nodes[x].ext.reverse_key == Some(key) {
            return;
        }

        self.graph.nodes[x].ext.reverse_key = Some(key);
        self.reverse_queue.push(VertexEntry {
            key: key[0],
            cost: key[1],
            vertex: x,
        });
    }

    /// The best live reverse-queue entry, dropping stale entries on the way.
    pub(super) fn peek_state(&mut self) -> Option<VertexEntry> {
        while let Some(&entry) = self.reverse_queue.peek() {
            if self.graph.nodes[entry.vertex].ext.reverse_key == Some([entry.key, entry.cost]) {
                return Some(entry);
            }

            self.reverse_queue.pop();
        }

        None
    }

    /// Runs the reverse search until the best forward edge is settled: its child is
    /// consistent and the edge's first key is below the reverse queue's first key, or the
    /// reverse queue is empty (OMPL's rule, in place of Algorithm 5, lines 10 to 12).
    pub(super) fn advance_reverse_search(&mut self) {
        while let Some(edge) = self.peek_edge() {
            let Some(top) = self.peek_state() else {
                return;
            };

            if self.is_consistent(edge.to) && edge.key[0] < top.key {
                return;
            }

            self.expand_state(top.vertex);
        }
    }

    /// Pops `x` from the reverse queue and expands it (Algorithm 5, lines 13 to 20): an
    /// overconsistent state takes its new cost-to-go, an underconsistent one is reset and
    /// re-evaluated; then every neighbor that may route through `x` is updated.
    ///
    /// The neighbor updates are the optimized LPA* form of Algorithm 6 with identical results:
    /// a lowered `h_exp[x]` can only lower a neighbor's `h_con` to the route through `x`, and a
    /// raised one only affects neighbors whose `h_con` was that route.
    pub(super) fn expand_state(&mut self, x: usize) {
        self.reverse_queue.pop();
        self.graph.nodes[x].ext.reverse_key = None;

        self.graph.ensure_adjacent(x);
        let neighbors = core::mem::take(&mut self.graph.nodes[x].links.adjacent);
        let node = &mut self.graph.nodes[x];
        let overconsistent = node.ext.h_con < node.ext.h_exp;

        if overconsistent {
            node.ext.h_exp = node.ext.h_con;
            self.lower_neighbors(x, &neighbors);
        } else {
            let old = core::mem::replace(&mut node.ext.h_exp, f64::INFINITY);
            self.raise_neighbors(x, old, &neighbors);
        }

        self.graph.nodes[x].links.adjacent = neighbors;

        if !overconsistent {
            self.update_state(x);
        }
    }

    /// Offers every neighbor `y` of the just-lowered `x` the route through `x`.
    fn lower_neighbors(&mut self, x: usize, neighbors: &[usize]) {
        let h_exp = self.graph.nodes[x].ext.h_exp;

        for &y in neighbors {
            if self.is_invalid(y, x) || self.is_reverse_root(y) {
                continue;
            }

            let through = h_exp + self.graph.cost_estimate(y, x);

            if through < self.graph.nodes[y].ext.h_con {
                self.set_h_con(y, through);
            }
        }
    }

    /// Re-evaluates every neighbor `y` whose `h_con` was the route through `x` at its old
    /// cost-to-go `old`.
    fn raise_neighbors(&mut self, x: usize, old: f64, neighbors: &[usize]) {
        for &y in neighbors {
            if self.is_invalid(y, x) {
                continue;
            }

            let through = old + self.graph.cost_estimate(y, x);

            if through.total_cmp(&self.graph.nodes[y].ext.h_con).is_eq() {
                self.update_state(y);
            }
        }
    }

    /// Recomputes `h_con[x]` from its neighbors' `h_exp` and queues `x` if that makes it
    /// inconsistent (Algorithm 6, with line 1 read as `x ∉ X_goal`).
    fn update_state(&mut self, x: usize) {
        if self.is_reverse_root(x) {
            return;
        }

        self.graph.ensure_adjacent(x);
        let h_con = self.graph.nodes[x]
            .links
            .adjacent
            .iter()
            .filter(|&&y| !self.is_invalid(x, y))
            .map(|&y| self.graph.nodes[y].ext.h_exp + self.graph.cost_estimate(x, y))
            .fold(f64::INFINITY, f64::min);

        self.set_h_con(x, h_con);
    }

    /// Sets `h_con[x]`, re-keys the forward edges into `x` if it changed, and queues or
    /// dequeues `x` by its consistency.
    fn set_h_con(&mut self, x: usize, h_con: f64) {
        if h_con.total_cmp(&self.graph.nodes[x].ext.h_con).is_ne() {
            self.graph.nodes[x].ext.h_con = h_con;
            self.rekey_incoming(x);
        }

        if self.is_consistent(x) {
            self.graph.nodes[x].ext.reverse_key = None;
        } else {
            self.queue_state(x);
        }
    }

    /// Records the invalid motion from `v` to `x` and repairs the reverse search around it
    /// (Algorithm 5, lines 7 to 9); the repair itself runs lazily before the next edge.
    pub(super) fn update_heuristic(&mut self, v: usize, x: usize) {
        self.graph.nodes[v].ext.invalid.push(x);
        self.update_state(v);
    }
}
