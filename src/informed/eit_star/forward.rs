//! EIT*'s forward search: grows the tree from the start along the edges with the least effort
//! that can still improve the solution.

use super::queue::ForwardKey;
use super::{EitStar, Kind};
use crate::{NearestNeighbors, Sampler, StateSpace, ValidityChecker};

impl<Sp, Sa, C, N> EitStar<Sp, Sa, C, N>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    N: NearestNeighbors<Sp> + Clone,
{
    /// Queues the forward edges from vertex `vertex` to its neighbors, except `skip` and
    /// pruned or known-invalid ones.
    pub(super) fn expand_forward(&mut self, vertex: usize, skip: Option<usize>) {
        self.graph.ensure_adjacent(vertex);

        for index in 0..self.graph.nodes[vertex].adjacent.len() {
            let target = self.graph.nodes[vertex].adjacent[index];

            if Some(target) != skip
                && self.graph.nodes[target].kind != Kind::Pruned
                && !self.checks.invalid.contains(&(vertex, target))
            {
                self.queue_forward(vertex, target);
            }
        }
    }

    /// The keys of the forward edge from `from` to `to`.
    pub(super) fn forward_key(&self, from: usize, to: usize) -> ForwardKey {
        let bound = self.graph.nodes[from].g
            + self.graph.cost_estimate(from, to)
            + self.graph.nodes[to].ext.h;
        let effort = self
            .remaining_checks(from, to)
            .saturating_add(self.graph.nodes[to].ext.effort);

        ForwardKey { bound, effort }
    }

    /// The smallest lower bound in the forward queue, or infinity when it is empty.
    pub(super) fn forward_lower_bound(&self) -> f64 {
        self.forward_queue
            .lowest_bound()
            .map_or(f64::INFINITY, |(_, bound)| bound)
    }

    /// The edge the forward search takes next; see [`ForwardQueue::front`].
    ///
    /// [`ForwardQueue::front`]: super::queue::ForwardQueue::front
    pub(super) fn peek_forward(&self) -> Option<(usize, usize)> {
        self.forward_queue.front(self.config.suboptimality)
    }

    /// Queues the forward edge from `from` to `to` with its current keys.
    fn queue_forward(&mut self, from: usize, to: usize) {
        let key = self.forward_key(from, to);

        if self.forward_queue.insert((from, to), key) {
            self.graph.nodes[to].ext.incoming.push(from);
            self.open_targets += usize::from(!self.is_closed(to));
        }
    }

    /// Removes the forward edge from `from` to `to` from the queue.
    fn dequeue_forward(&mut self, from: usize, to: usize) {
        if self.forward_queue.remove((from, to)).is_some() {
            self.graph.nodes[to]
                .ext
                .incoming
                .retain(|&source| source != from);
            self.open_targets -= usize::from(!self.is_closed(to));
        }
    }

    /// Empties the forward queue.
    pub(super) fn clear_forward_queue(&mut self) {
        self.forward_queue.clear();
        self.open_targets = 0;

        for node in &mut self.graph.nodes {
            node.ext.incoming.clear();
        }
    }

    /// Re-keys the forward edge from `from` to `to`, if it is queued.
    pub(super) fn refresh_forward(&mut self, from: usize, to: usize) {
        if self.forward_queue.contains((from, to)) {
            let key = self.forward_key(from, to);
            self.forward_queue.insert((from, to), key);
        }
    }

    /// Re-keys the queued forward edges into `to`, after its cost-to-go or effort changed.
    pub(super) fn refresh_forward_into(&mut self, to: usize) {
        for index in 0..self.graph.nodes[to].ext.incoming.len() {
            let from = self.graph.nodes[to].ext.incoming[index];
            self.refresh_forward(from, to);
        }
    }

    /// Re-keys the queued forward edges out of `from`, after its cost-to-come changed.
    fn refresh_forward_from(&mut self, from: usize) {
        self.graph.ensure_adjacent(from);

        for index in 0..self.graph.nodes[from].adjacent.len() {
            let to = self.graph.nodes[from].adjacent[index];
            self.refresh_forward(from, to);
        }
    }

    /// Re-keys every queued forward edge, after the reverse search restarted.
    pub(super) fn rekey_forward_queue(&mut self) {
        let edges: alloc::vec::Vec<(usize, usize)> = self.forward_queue.edges().collect();

        for (from, to) in edges {
            self.refresh_forward(from, to);
        }
    }

    /// Whether the forward queue holds an edge that could still beat the solution (OMPL's
    /// `continueForwardSearch`).
    pub(super) fn should_continue_forward(&self) -> bool {
        !self.forward_queue.is_empty() && self.forward_lower_bound() < self.graph.best_cost
    }

    /// Processes the next forward edge (OMPL's `iterateForwardSearch`). Returns whether the
    /// tree changed.
    pub(super) fn forward_step(&mut self) -> bool {
        let Some((from, to)) = self.peek_forward() else {
            return false;
        };

        self.dequeue_forward(from, to);

        if self.graph.nodes[to].parent == Some(from) {
            if !self.graph.nodes[to].in_goal() {
                self.expand_forward(to, None);
            }

            return false;
        }

        let estimate = self.graph.nodes[from].g + self.graph.cost_estimate(from, to);
        if estimate >= self.graph.nodes[to].g {
            return false;
        }

        if !self.is_valid(from, to) {
            self.on_invalid_edge(from, to);
            return false;
        }

        // With c = c_hat for a valid edge, the checked cost is the estimate.
        if estimate + self.graph.nodes[to].ext.h >= self.graph.best_cost {
            return false;
        }

        self.connect(from, to, estimate);

        true
    }

    /// Reacts to an invalid forward edge: when the reverse tree relies on it, checks edges more
    /// densely, `2k + 1` states instead of `k`, and restarts the reverse search.
    fn on_invalid_edge(&mut self, from: usize, to: usize) {
        if self.is_in_reverse_tree(from, to) {
            self.sparse_checks = self.sparse_checks.saturating_mul(2).saturating_add(1);
            self.restart_reverse_search();
        }
    }

    /// Makes `from` the forward parent of `to` with cost `g`, updates the costs below `to` and
    /// the solution, and expands `to` unless it is a goal.
    fn connect(&mut self, from: usize, to: usize, g: f64) {
        if let Some(old) = self.graph.nodes[to].parent {
            self.graph.nodes[old].children.retain(|&child| child != to);
        }

        let was_sample = self.graph.nodes[to].kind == Kind::Sample;
        let node = &mut self.graph.nodes[to];
        node.parent = Some(from);
        node.g = g;
        node.kind = Kind::Vertex;
        self.graph.nodes[from].children.push(to);
        self.propagate_costs(to);

        if self.graph.nodes[to].in_goal() {
            if was_sample {
                self.graph.solutions.push(to);
            }
        } else {
            self.expand_forward(to, Some(from));
        }

        self.graph.update_best_cost();
    }

    /// Recomputes the costs of all descendants of `x` from their parents' costs.
    fn propagate_costs(&mut self, x: usize) {
        let mut stack = self.graph.nodes[x].children.clone();

        self.refresh_forward_from(x);

        while let Some(child) = stack.pop() {
            let parent = self.graph.nodes[child]
                .parent
                .expect("a child has a parent");
            self.graph.nodes[child].g =
                self.graph.nodes[parent].g + self.graph.cost_estimate(parent, child);
            self.refresh_forward_from(child);
            stack.extend_from_slice(&self.graph.nodes[child].children);
        }
    }
}
