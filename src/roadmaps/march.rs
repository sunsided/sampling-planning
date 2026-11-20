//! The fast-marching machinery shared by FMT* and BFMT*: a sample batch with cached neighbor
//! lists, and wavefronts that march over it.

use alloc::{collections::BinaryHeap, vec::Vec};

use crate::graph::Entry;
use crate::radius::{dimension_f64, shrinking_radius};
use crate::{NearestNeighbors, StateSpace, ValidityChecker};

/// A fixed batch of valid states with an index and lazily computed neighbor lists.
pub(super) struct Batch<'a, Sp: StateSpace, C, N> {
    /// The configuration space.
    pub(super) space: &'a Sp,
    /// Decides which motions are allowed.
    pub(super) checker: &'a C,
    /// The batch states, indexed by node id.
    pub(super) states: Vec<Sp::State>,
    /// Index over all batch states.
    nn: N,
    /// The connection radius for this batch.
    radius: f64,
    /// Neighbors within `radius` per node, excluding the node; filled on first use.
    near: Vec<Option<Vec<usize>>>,
}

impl<'a, Sp, C, N> Batch<'a, Sp, C, N>
where
    Sp: StateSpace,
    C: ValidityChecker<Sp>,
    N: NearestNeighbors<Sp>,
{
    /// Indexes `states` in `nn` (which must be empty) and fixes the radius
    /// `gamma (ln n / n)^(1/d)`.
    pub(super) fn new(
        space: &'a Sp,
        checker: &'a C,
        gamma: f64,
        states: Vec<Sp::State>,
        mut nn: N,
    ) -> Self {
        let n = states.len();

        for (id, state) in states.iter().enumerate() {
            nn.insert(id, state.clone());
        }

        Self {
            space,
            checker,
            radius: shrinking_radius(gamma, n, dimension_f64(space.dimension())),
            states,
            nn,
            near: alloc::vec![None; n],
        }
    }

    /// The number of batch states.
    pub(super) fn len(&self) -> usize {
        self.states.len()
    }

    /// The neighbors of `node` within the radius, in id order, computed on first use.
    fn neighbors(&mut self, node: usize) -> &[usize] {
        if self.near[node].is_none() {
            let mut near = Vec::new();
            self.nn
                .within_radius(self.space, &self.states[node], self.radius, &mut near);
            near.retain(|&other| other != node);
            self.near[node] = Some(near);
        }

        self.near[node].as_deref().unwrap_or_default()
    }

    /// Whether the motion from node `from` to node `to` checks as valid.
    fn is_motion_valid(&self, from: usize, to: usize) -> bool {
        self.checker
            .check_motion(self.space, &self.states[from], &self.states[to])
            .is_valid()
    }
}

/// Which way a wavefront's edges are traveled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Direction {
    /// Away from the root: edges go from parent to child (the start tree).
    Outward,
    /// Toward the root: edges go from child to parent (the goal tree of BFMT*).
    Inward,
}

/// Where a node stands in one wavefront's march.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Status {
    /// Not connected yet (the set `W` or `V_unvisited`).
    Unvisited,
    /// Connected during the current expansion; joins the open set after it (`H_new`).
    Fresh,
    /// Connected and on the wavefront (the set `H` or `V_open`).
    Open,
    /// Expanded; never touched again.
    Closed,
}

/// One marching tree over a [`Batch`]: FMT* has one, BFMT* two.
pub(super) struct Wave {
    /// How edges are traveled, and therefore checked and costed.
    direction: Direction,
    /// Where each node stands.
    pub(super) status: Vec<Status>,
    /// Cost-to-root of connected nodes in travel direction; infinite for unconnected ones.
    pub(super) cost: Vec<f64>,
    /// Tree parent of connected nodes; `None` for the root and unconnected nodes.
    pub(super) parent: Vec<Option<usize>>,
    /// The open set ordered by cost, then id; closed entries are skipped when popped.
    open: BinaryHeap<Entry>,
}

impl Wave {
    /// A wave over `n` nodes with only `root` connected and open.
    pub(super) fn new(n: usize, root: usize, direction: Direction) -> Self {
        let mut wave = Self {
            direction,
            status: alloc::vec![Status::Unvisited; n],
            cost: alloc::vec![f64::INFINITY; n],
            parent: alloc::vec![None; n],
            open: BinaryHeap::new(),
        };
        wave.status[root] = Status::Open;
        wave.cost[root] = 0.0;
        wave.open.push(Entry {
            cost: 0.0,
            node: root,
        });

        wave
    }

    /// The cheapest open node, skipping closed entries, without removing it.
    pub(super) fn peek_open(&mut self) -> Option<usize> {
        while let Some(&Entry { node, .. }) = self.open.peek() {
            if self.status[node] == Status::Open {
                return Some(node);
            }

            self.open.pop();
        }

        None
    }

    /// Connects each unvisited neighbor of the open node `z` to its cheapest open neighbor if
    /// that motion is valid, moves the new nodes to the open set and closes `z`. Returns the
    /// newly connected nodes.
    ///
    /// Only the locally best parent is checked; a failure leaves the node for later waves
    /// (Janson et al. 2015, Algorithm 2).
    pub(super) fn expand<Sp, C, N>(
        &mut self,
        batch: &mut Batch<'_, Sp, C, N>,
        z: usize,
    ) -> Vec<usize>
    where
        Sp: StateSpace,
        C: ValidityChecker<Sp>,
        N: NearestNeighbors<Sp>,
    {
        let mut fresh = Vec::new();
        let near_z = batch.neighbors(z).to_vec();

        for x in near_z {
            if self.status[x] != Status::Unvisited {
                continue;
            }

            let Some((y, cost)) = self.cheapest_open_neighbor(batch, x) else {
                continue;
            };

            if self.is_edge_valid(batch, y, x) {
                self.status[x] = Status::Fresh;
                self.cost[x] = cost;
                self.parent[x] = Some(y);
                fresh.push(x);
            }
        }

        for &x in &fresh {
            self.status[x] = Status::Open;
            self.open.push(Entry {
                cost: self.cost[x],
                node: x,
            });
        }

        self.status[z] = Status::Closed;

        fresh
    }

    /// The open neighbor `y` of `x` minimizing `cost(y) + c(y, x)` (the edge cost in travel
    /// direction), lowest id on ties, with that cost.
    fn cheapest_open_neighbor<Sp, C, N>(
        &self,
        batch: &mut Batch<'_, Sp, C, N>,
        x: usize,
    ) -> Option<(usize, f64)>
    where
        Sp: StateSpace,
        C: ValidityChecker<Sp>,
        N: NearestNeighbors<Sp>,
    {
        let mut best: Option<(usize, f64)> = None;
        let near_x = batch.neighbors(x).to_vec();

        for y in near_x {
            if self.status[y] != Status::Open {
                continue;
            }

            let cost = self.cost[y] + self.edge_cost(batch, y, x);
            if best.is_none_or(|(_, best_cost)| cost < best_cost) {
                best = Some((y, cost));
            }
        }

        best
    }

    /// The cost of the tree edge between `parent` and `child`, in travel direction.
    fn edge_cost<Sp, C, N>(&self, batch: &Batch<'_, Sp, C, N>, parent: usize, child: usize) -> f64
    where
        Sp: StateSpace,
    {
        let (from, to) = self.travel(parent, child);

        batch.space.distance(&batch.states[from], &batch.states[to])
    }

    /// Whether the tree edge between `parent` and `child` checks as valid in travel direction.
    fn is_edge_valid<Sp, C, N>(
        &self,
        batch: &Batch<'_, Sp, C, N>,
        parent: usize,
        child: usize,
    ) -> bool
    where
        Sp: StateSpace,
        C: ValidityChecker<Sp>,
        N: NearestNeighbors<Sp>,
    {
        let (from, to) = self.travel(parent, child);

        batch.is_motion_valid(from, to)
    }

    /// The tree edge between `parent` and `child` as `(from, to)` in travel direction.
    const fn travel(&self, parent: usize, child: usize) -> (usize, usize) {
        match self.direction {
            Direction::Outward => (parent, child),
            Direction::Inward => (child, parent),
        }
    }

    /// The nodes from `id` up to the root, `id` first.
    pub(super) fn branch(&self, id: usize) -> Vec<usize> {
        let mut nodes = Vec::new();
        let mut current = Some(id);

        while let Some(node) = current {
            nodes.push(node);
            current = self.parent[node];
        }

        nodes
    }
}
