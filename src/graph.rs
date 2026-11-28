//! Deterministic shortest paths and priority-queue entries, shared by the graph-based planners.

use alloc::{collections::BinaryHeap, vec::Vec};
use core::cmp::Ordering;

/// An outgoing arc found while expanding a node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Arc {
    /// The node the arc leads to.
    pub(crate) to: usize,
    /// The arc cost; non-negative.
    pub(crate) cost: f64,
    /// The roadmap edge behind the arc, or `None` for query-local arcs.
    pub(crate) edge: Option<usize>,
}

/// The result of [`shortest_path`].
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Route {
    /// The nodes from the source to the goal node, both included.
    pub(crate) nodes: Vec<usize>,
    /// The roadmap edges along the route, in travel order; `None` for query-local arcs.
    pub(crate) edges: Vec<Option<usize>>,
    /// The total cost.
    pub(crate) cost: f64,
}

/// A priority-queue entry ordered by cost, then by node id, both ascending.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Entry {
    /// The tentative cost of `node`.
    pub(crate) cost: f64,
    /// The node.
    pub(crate) node: usize,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Entry {}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    /// Reversed, so that [`BinaryHeap`] pops the cheapest entry, and the lowest id on ties.
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .cost
            .total_cmp(&self.cost)
            .then_with(|| other.node.cmp(&self.node))
    }
}

/// Dijkstra's algorithm from `source` over `node_count` nodes until the first goal node is
/// settled.
///
/// `arcs(node, out)` replaces `out` with the outgoing arcs of `node`. Ties between equally
/// cheap nodes are settled lowest id first, so the result is deterministic.
pub(crate) fn shortest_path<A>(
    node_count: usize,
    source: usize,
    is_goal: impl Fn(usize) -> bool,
    arcs: A,
) -> Option<Route>
where
    A: FnMut(usize, &mut Vec<Arc>),
{
    Dijkstra::default().run(node_count, source, is_goal, arcs)
}

/// The buffers of [`shortest_path`], kept between searches so that repeated queries do not
/// allocate once the buffers have grown to the graph's size.
#[derive(Debug, Clone, Default)]
pub(crate) struct Dijkstra {
    /// The tentative cost per node.
    cost: Vec<f64>,
    /// The parent node and the edge to it, per node.
    parent: Vec<Option<(usize, Option<usize>)>>,
    /// Whether each node is settled.
    settled: Vec<bool>,
    /// The open nodes.
    heap: BinaryHeap<Entry>,
    /// The arcs of the node being expanded.
    out: Vec<Arc>,
}

impl Dijkstra {
    /// [`shortest_path`] in these buffers.
    pub(crate) fn run<A>(
        &mut self,
        node_count: usize,
        source: usize,
        is_goal: impl Fn(usize) -> bool,
        mut arcs: A,
    ) -> Option<Route>
    where
        A: FnMut(usize, &mut Vec<Arc>),
    {
        self.reset(node_count);
        self.cost[source] = 0.0;
        self.heap.push(Entry {
            cost: 0.0,
            node: source,
        });

        while let Some(Entry { cost: c, node }) = self.heap.pop() {
            if self.settled[node] {
                continue;
            }

            self.settled[node] = true;
            if is_goal(node) {
                return Some(trace(&self.parent, node, c));
            }

            arcs(node, &mut self.out);
            self.relax(node, c);
        }

        None
    }

    /// Clears the buffers for a graph of `node_count` nodes.
    fn reset(&mut self, node_count: usize) {
        self.cost.clear();
        self.cost.resize(node_count, f64::INFINITY);
        self.parent.clear();
        self.parent.resize(node_count, None);
        self.settled.clear();
        self.settled.resize(node_count, false);
        self.heap.clear();
    }

    /// Lowers the cost of every target of the arcs in `out` that `node`, settled at cost `c`,
    /// reaches more cheaply.
    fn relax(&mut self, node: usize, c: f64) {
        for arc in &self.out {
            let candidate = c + arc.cost;

            if candidate < self.cost[arc.to] {
                self.cost[arc.to] = candidate;
                self.parent[arc.to] = Some((node, arc.edge));
                self.heap.push(Entry {
                    cost: candidate,
                    node: arc.to,
                });
            }
        }
    }
}

/// Follows `parent` links back from `goal` to the source.
fn trace(parent: &[Option<(usize, Option<usize>)>], goal: usize, cost: f64) -> Route {
    let mut nodes = alloc::vec![goal];
    let mut edges = Vec::new();
    let mut current = goal;

    while let Some((previous, edge)) = parent[current] {
        nodes.push(previous);
        edges.push(edge);
        current = previous;
    }

    nodes.reverse();
    edges.reverse();

    Route { nodes, edges, cost }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A diamond 0 -> {1, 2} -> 3 with equal costs on both sides and a pricier direct arc.
    fn diamond(node: usize, out: &mut Vec<Arc>) {
        let arc = |to, cost, edge| Arc {
            to,
            cost,
            edge: Some(edge),
        };

        out.clear();
        match node {
            0 => out.extend([arc(2, 1.0, 0), arc(1, 1.0, 1), arc(3, 2.5, 2)]),
            1 => out.push(arc(3, 1.0, 3)),
            2 => out.push(arc(3, 1.0, 4)),
            _ => {}
        }
    }

    #[test]
    fn finds_cheapest_route_and_breaks_ties_by_id() {
        let route = shortest_path(4, 0, |n| n == 3, diamond).expect("reachable");

        assert_eq!(route.nodes, [0, 1, 3]);
        assert_eq!(route.edges, [Some(1), Some(3)]);
        assert!((route.cost - 2.0).abs() < 1e-12);
    }

    #[test]
    fn unreachable_goal_is_none_and_source_goal_is_trivial() {
        assert_eq!(shortest_path(4, 3, |n| n == 0, diamond), None);

        let trivial = shortest_path(4, 0, |n| n == 0, diamond).expect("source is a goal");
        assert_eq!(trivial.nodes, [0]);
        assert_eq!(trivial.edges, []);
    }
}
