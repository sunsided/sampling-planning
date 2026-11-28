//! EIT*'s reverse search: an A* over sparsely checked edges from the goal states that computes
//! the cost-to-go `h` and the effort-to-go of every state it reaches.

use super::{EitStar, Kind};
use crate::informed::queues::EdgeEntry;
use crate::radius::count_to_f64;
use crate::{NearestNeighbors, Sampler, StateSpace, ValidityChecker};

impl<Sp, Sa, C, N> EitStar<Sp, Sa, C, N>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    N: NearestNeighbors<Sp> + Clone,
{
    /// Forgets the reverse tree and starts a new reverse search from the goal states (OMPL's
    /// `restartReverseSearch`).
    pub(super) fn restart_reverse_search(&mut self) {
        self.reverse_queue.clear();
        self.reverse_tag += 1;

        for node in &mut self.graph.nodes {
            node.ext.h = f64::INFINITY;
            node.ext.effort = usize::MAX;
            node.ext.reverse_parent = None;
        }

        for goal in 0..self.graph.nodes.len() {
            if self.is_goal(goal) {
                self.graph.nodes[goal].ext.h = 0.0;
                self.graph.nodes[goal].ext.effort = 0;
                self.expand_reverse(goal, None);
            }
        }

        // The new tag reopened every state, and every cost-to-go changed.
        self.open_targets = self.forward_queue.len();
        self.rekey_forward_queue();
    }

    /// Queues the reverse edges from `source` to its neighbors, except `skip` and neighbors
    /// whose admissible solution cost cannot beat the current solution.
    fn expand_reverse(&mut self, source: usize, skip: Option<usize>) {
        self.graph.ensure_adjacent(source);

        for index in 0..self.graph.nodes[source].adjacent.len() {
            let target = self.graph.nodes[source].adjacent[index];

            if Some(target) == skip || self.graph.nodes[target].kind == Kind::Pruned {
                continue;
            }

            let entry = self.reverse_entry(source, target);
            if self.graph.best_cost.is_finite() && entry.key[0] > self.graph.best_cost {
                continue;
            }

            self.reverse_queue.push(entry);
        }
    }

    /// The queue entry of the reverse edge from `source` (closer to the goal) to `target`:
    /// `(h(s) + c_hat + g_hat(t), h(s) + c_hat, effort)`.
    fn reverse_entry(&self, source: usize, target: usize) -> EdgeEntry {
        let to_go = self.graph.nodes[source].ext.h + self.graph.cost_estimate(target, source);
        let effort = self.graph.nodes[source]
            .ext
            .effort
            .saturating_add(self.remaining_checks(target, source));

        EdgeEntry {
            key: [
                to_go + self.graph.nodes[target].g_hat,
                to_go,
                count_to_f64(effort),
            ],
            from: source,
            to: target,
        }
    }

    /// Whether the forward search needs the reverse search to go on (OMPL's
    /// `continueReverseSearch` without multi-query planning).
    pub(super) fn should_continue_reverse(&mut self) -> bool {
        let Some(reverse_top) = self.reverse_queue.peek().map(|entry| entry.key[0]) else {
            return false;
        };

        let Some(edge) = self.peek_forward() else {
            return true;
        };

        let target = edge.1;
        let closed_and_better = self.is_closed(target) && self.forward_lower_bound() < reverse_top;
        let all_closed = self.open_targets == 0;
        let reached =
            self.config.suboptimality.is_infinite() && self.graph.nodes[target].ext.h.is_finite();

        !(closed_and_better || all_closed || reached)
    }

    /// Whether state `id` was expanded in the running reverse search.
    pub(super) fn is_closed(&self, id: usize) -> bool {
        self.graph.nodes[id].ext.closed_tag == self.reverse_tag
    }

    /// Processes the best reverse edge (OMPL's `iterateReverseSearch`): closes its source and
    /// adds its target to the reverse tree if the edge passes the sparse check and lowers the
    /// target's cost-to-go.
    pub(super) fn reverse_step(&mut self) {
        let Some(entry) = self.reverse_queue.pop() else {
            return;
        };

        let (source, target) = (entry.from, entry.to);
        self.close(source);

        if self.graph.nodes[target].ext.reverse_parent == Some(source) {
            self.expand_reverse(target, Some(source));
            return;
        }

        if self.is_closed(target) {
            return;
        }

        let valid = self.could_be_valid(target, source);
        // The sparse check changed the remaining effort of the edge in both directions.
        self.refresh_forward(target, source);
        self.refresh_forward(source, target);

        if !valid {
            return;
        }

        let to_go = self.graph.nodes[source].ext.h + self.graph.cost_estimate(target, source);
        if to_go >= self.graph.nodes[target].ext.h {
            return;
        }

        let effort = self.graph.nodes[source]
            .ext
            .effort
            .saturating_add(self.remaining_checks(target, source));
        let node = &mut self.graph.nodes[target];
        node.ext.reverse_parent = Some(source);
        node.ext.h = to_go;
        node.ext.effort = node.ext.effort.min(effort);
        self.refresh_forward_into(target);
        self.expand_reverse(target, Some(source));
    }

    /// Closes state `id` in the running reverse search.
    fn close(&mut self, id: usize) {
        if !self.is_closed(id) {
            self.graph.nodes[id].ext.closed_tag = self.reverse_tag;
            self.open_targets -= self.graph.nodes[id].ext.incoming.len();
        }
    }

    /// Whether the forward edge from `from` to `to` is an edge of the reverse tree.
    pub(super) fn is_in_reverse_tree(&self, from: usize, to: usize) -> bool {
        self.graph.nodes[from].ext.reverse_parent == Some(to)
            || self.graph.nodes[to].ext.reverse_parent == Some(from)
    }
}
