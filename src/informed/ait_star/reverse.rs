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

        for node in &mut self.nodes {
            node.h_con = f64::INFINITY;
            node.h_exp = f64::INFINITY;
            node.reverse_key = None;
        }

        for x in 0..self.nodes.len() {
            if self.is_reverse_root(x) {
                self.nodes[x].h_con = 0.0;
                self.queue_state(x);
            }
        }
    }

    /// Whether `x` is a goal of the reverse search: a live state in the goal region.
    fn is_reverse_root(&self, x: usize) -> bool {
        self.nodes[x].kind != Kind::Pruned && self.nodes[x].in_goal()
    }

    /// Whether `x` has `h_con == h_exp`.
    pub(super) fn is_consistent(&self, x: usize) -> bool {
        self.nodes[x].h_con.total_cmp(&self.nodes[x].h_exp).is_eq()
    }

    /// The LPA* key of `x`: `(min(h_con, h_exp) + g_hat, min(h_con, h_exp))`.
    fn reverse_key(&self, x: usize) -> [f64; 2] {
        let node = &self.nodes[x];
        let h = node.h_con.min(node.h_exp);

        [h + node.g_hat, h]
    }

    /// Queues `x` with its current key, unless it is queued with that key already.
    fn queue_state(&mut self, x: usize) {
        let key = self.reverse_key(x);

        if self.nodes[x].reverse_key == Some(key) {
            return;
        }

        self.nodes[x].reverse_key = Some(key);
        self.reverse_queue.push(VertexEntry {
            key: key[0],
            cost: key[1],
            vertex: x,
        });
    }

    /// The best live reverse-queue entry, dropping stale entries on the way.
    pub(super) fn peek_state(&mut self) -> Option<VertexEntry> {
        while let Some(&entry) = self.reverse_queue.peek() {
            if self.nodes[entry.vertex].reverse_key == Some([entry.key, entry.cost]) {
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
        self.nodes[x].reverse_key = None;

        let neighbors = core::mem::take(&mut self.nodes[x].adjacent);
        let node = &mut self.nodes[x];
        let overconsistent = node.h_con < node.h_exp;

        if overconsistent {
            node.h_exp = node.h_con;
            self.lower_neighbors(x, &neighbors);
        } else {
            let old = core::mem::replace(&mut node.h_exp, f64::INFINITY);
            self.raise_neighbors(x, old, &neighbors);
        }

        self.nodes[x].adjacent = neighbors;

        if !overconsistent {
            self.update_state(x);
        }
    }

    /// Offers every neighbor `y` of the just-lowered `x` the route through `x`.
    fn lower_neighbors(&mut self, x: usize, neighbors: &[usize]) {
        let h_exp = self.nodes[x].h_exp;

        for &y in neighbors {
            if self.is_invalid(y, x) || self.is_reverse_root(y) {
                continue;
            }

            let through = h_exp + self.cost_estimate(y, x);

            if through < self.nodes[y].h_con {
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

            let through = old + self.cost_estimate(y, x);

            if through.total_cmp(&self.nodes[y].h_con).is_eq() {
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

        let h_con = self.nodes[x]
            .adjacent
            .iter()
            .filter(|&&y| !self.is_invalid(x, y))
            .map(|&y| self.nodes[y].h_exp + self.cost_estimate(x, y))
            .fold(f64::INFINITY, f64::min);

        self.set_h_con(x, h_con);
    }

    /// Sets `h_con[x]`, re-keys the forward edges into `x` if it changed, and queues or
    /// dequeues `x` by its consistency.
    fn set_h_con(&mut self, x: usize, h_con: f64) {
        if h_con.total_cmp(&self.nodes[x].h_con).is_ne() {
            self.nodes[x].h_con = h_con;
            self.rekey_incoming(x);
        }

        if self.is_consistent(x) {
            self.nodes[x].reverse_key = None;
        } else {
            self.queue_state(x);
        }
    }

    /// Records the invalid motion from `v` to `x` and repairs the reverse search around it
    /// (Algorithm 5, lines 7 to 9); the repair itself runs lazily before the next edge.
    pub(super) fn update_heuristic(&mut self, v: usize, x: usize) {
        self.nodes[v].invalid.push(x);
        self.update_state(v);
    }
}
