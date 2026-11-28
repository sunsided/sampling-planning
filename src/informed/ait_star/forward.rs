//! AIT*'s forward search: the edge queue and the tree it grows from the start (Algorithms 2
//! to 4).

use super::{AitStar, EdgeEntry, Kind};
use crate::{NearestNeighbors, Sampler, StateSpace, ValidityChecker};

impl<Sp, Sa, C, N> AitStar<Sp, Sa, C, N>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    N: NearestNeighbors<Sp> + Clone,
{
    /// The forward key of the edge from `v` to `x`: `(g_F(v) + c_hat + h_con(x), g_F(v) +
    /// c_hat, g_F(v))`.
    pub(super) fn edge_key(&self, v: usize, x: usize) -> [f64; 3] {
        let g = self.nodes[v].g;
        let through = g + self.cost_estimate(v, x);

        [through + self.nodes[x].h_con, through, g]
    }

    /// Queues the outgoing edges of vertex `v` to all its neighbors that are not known to be
    /// invalid (Algorithms 3 and 4).
    pub(super) fn expand(&mut self, v: usize) {
        self.ensure_adjacent(v);
        let neighbors = core::mem::take(&mut self.nodes[v].adjacent);

        for &x in &neighbors {
            if !self.is_invalid(v, x) {
                self.queue_edge(v, x);
            }
        }

        self.nodes[v].adjacent = neighbors;
    }

    /// Queues the edge from `v` to `x` with its current key, unless it is queued with that
    /// key already.
    fn queue_edge(&mut self, v: usize, x: usize) {
        let key = self.edge_key(v, x);
        let incoming = &mut self.nodes[x].incoming;

        match incoming.iter_mut().find(|(from, _)| *from == v) {
            Some((_, old)) if same_key(*old, key) => return,
            Some((_, old)) => *old = key,
            None => incoming.push((v, key)),
        }

        self.forward_queue.push(EdgeEntry {
            key,
            from: v,
            to: x,
        });
    }

    /// Re-queues the queued edges into `x` after its `h_con` changed.
    pub(super) fn rekey_incoming(&mut self, x: usize) {
        let mut incoming = core::mem::take(&mut self.nodes[x].incoming);

        for (v, key) in &mut incoming {
            let fresh = self.edge_key(*v, x);

            if !same_key(*key, fresh) {
                *key = fresh;
                self.forward_queue.push(EdgeEntry {
                    key: fresh,
                    from: *v,
                    to: x,
                });
            }
        }

        self.nodes[x].incoming = incoming;
    }

    /// Whether `edge` is the live queue entry of its edge.
    fn is_live(&self, edge: &EdgeEntry) -> bool {
        self.nodes[edge.to]
            .incoming
            .iter()
            .any(|&(from, key)| from == edge.from && same_key(key, edge.key))
    }

    /// The best live forward edge, dropping stale entries on the way.
    pub(super) fn peek_edge(&mut self) -> Option<EdgeEntry> {
        while let Some(&edge) = self.forward_queue.peek() {
            if self.is_live(&edge) {
                return Some(edge);
            }

            self.forward_queue.pop();
        }

        None
    }

    /// Removes and returns the best live forward edge.
    pub(super) fn pop_edge(&mut self) -> Option<EdgeEntry> {
        let edge = self.peek_edge()?;
        self.forward_queue.pop();
        self.nodes[edge.to]
            .incoming
            .retain(|&(from, _)| from != edge.from);

        Some(edge)
    }

    /// Empties the forward queue.
    pub(super) fn clear_forward_queue(&mut self) {
        self.forward_queue.clear();
        self.stale_keys = false;

        for node in &mut self.nodes {
            node.incoming.clear();
        }
    }

    /// Re-keys the forward queue after a rewiring lowered the costs of a subtree.
    pub(super) fn resort_if_stale(&mut self) {
        if !self.stale_keys {
            return;
        }

        self.stale_keys = false;
        self.forward_queue.clear();

        for x in 0..self.nodes.len() {
            let mut incoming = core::mem::take(&mut self.nodes[x].incoming);

            for (v, key) in &mut incoming {
                *key = self.edge_key(*v, x);
                self.forward_queue.push(EdgeEntry {
                    key: *key,
                    from: *v,
                    to: x,
                });
            }

            self.nodes[x].incoming = incoming;
        }
    }

    /// Processes the best edge (Algorithm 2, lines 5 to 25): expands along a tree edge, adds
    /// or rewires its target when the checked edge improves it, hands an invalid edge to the
    /// reverse search, or ends the batch when even the estimate cannot improve the solution.
    /// Returns whether the tree changed.
    pub(super) fn process_edge(&mut self, edge: EdgeEntry) -> bool {
        let (v, x) = (edge.from, edge.to);

        if self.nodes[v].kind != Kind::Vertex || self.nodes[x].kind == Kind::Pruned {
            return false;
        }

        let [total, through, _] = self.edge_key(v, x);

        if total >= self.best_cost {
            self.clear_forward_queue();
            return false;
        }

        if self.nodes[x].parent == Some(v) {
            self.expand(x);
            return false;
        }

        if through >= self.nodes[x].g {
            return false;
        }

        if !self.is_motion_valid(v, x) {
            self.update_heuristic(v, x);
            return false;
        }

        // With c = c_hat for a valid edge, lines 11 and 12 equal lines 6 and 9.
        self.connect(v, x, through);
        self.expand(x);

        true
    }

    /// Whether the motion from node `from` to node `to` checks as valid.
    fn is_motion_valid(&self, from: usize, to: usize) -> bool {
        self.checker
            .check_motion(&self.space, &self.nodes[from].state, &self.nodes[to].state)
            .is_valid()
    }

    /// Makes `v` the parent of `x` with cost `g`: adds the sample `x` to the tree, or rewires
    /// the vertex `x` and lowers the costs of its subtree. Updates the solution.
    fn connect(&mut self, v: usize, x: usize, g: f64) {
        if let Some(old) = self.nodes[x].parent {
            self.nodes[old].children.retain(|&child| child != x);
            self.stale_keys = true;
        }

        let was_sample = self.nodes[x].kind == Kind::Sample;
        let node = &mut self.nodes[x];
        node.parent = Some(v);
        node.g = g;
        node.kind = Kind::Vertex;
        self.nodes[v].children.push(x);

        if was_sample {
            if self.nodes[x].in_goal() {
                self.solutions.push(x);
            }
        } else {
            self.propagate_costs(x);
        }

        self.update_best_cost();
    }

    /// Recomputes the costs of all descendants of `x` from their parents' costs.
    fn propagate_costs(&mut self, x: usize) {
        let mut stack = self.nodes[x].children.clone();

        while let Some(child) = stack.pop() {
            let parent = self.nodes[child].parent.expect("a child has a parent");
            self.nodes[child].g = self.nodes[parent].g + self.cost_estimate(parent, child);
            stack.extend_from_slice(&self.nodes[child].children);
        }
    }

    /// Sets `c_current` to the cost of the cheapest solution vertex (Algorithm 2, line 19).
    fn update_best_cost(&mut self) {
        self.best_cost = self
            .solutions
            .iter()
            .map(|&id| self.nodes[id].g)
            .fold(f64::INFINITY, f64::min);
    }
}

/// Whether two keys are bit-for-bit identical: a key recomputed from unchanged costs matches
/// the stored one exactly, any change makes it differ.
fn same_key(a: [f64; 3], b: [f64; 3]) -> bool {
    a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits())
}
