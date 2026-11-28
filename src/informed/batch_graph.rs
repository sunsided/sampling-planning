//! The forward tree and batch graph shared by AIT* and EIT*: stored states, informed batches,
//! pruning, the batch's random geometric graph and the best solution.

use alloc::vec::Vec;

use rand_core::Rng;

use super::batch::{Informed, Neighborhood};
use super::rgg::{NeighborCache, merge_sorted};
use super::{BitStarConfig, Error};
use crate::{GoalRegion, NearestNeighbors, Path, Sampler, StateSpace, ValidityChecker};

/// The id of the start, the forward tree's root.
pub(super) const START: usize = 0;

/// What a stored state currently is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    /// A forward-tree vertex.
    Vertex,
    /// A sample not connected to the forward tree.
    Sample,
    /// Pruned for good: it cannot improve the solution.
    Pruned,
}

/// One stored state: a forward-tree vertex or a sample, with its batch-graph neighbors and
/// the planner's reverse-search values in `ext`.
#[derive(Debug, Clone)]
pub(super) struct Node<S, X> {
    /// The state.
    pub(super) state: S,
    /// What the state currently is.
    pub(super) kind: Kind,
    /// The cost-to-come through the forward tree, `g_F`; infinite for samples.
    pub(super) g: f64,
    /// The admissible cost-to-come `g_hat = d(start, x)`, also the reverse search's heuristic.
    pub(super) g_hat: f64,
    /// The geometric cost-to-go `max(0, d(x, goal center) - tolerance)`, used for pruning and
    /// to recognize goal states (zero inside the goal region).
    pub(super) h_hat: f64,
    /// The forward-tree parent; `None` for the root and samples.
    pub(super) parent: Option<usize>,
    /// The forward-tree children.
    pub(super) children: Vec<usize>,
    /// The neighbors in this batch's graph: the RGG neighbors, symmetrized, plus the forward
    /// tree edges at the start of the batch. Sorted ascending, no duplicates. Valid while
    /// `adjacent_batch` is the current batch; see [`BatchGraph::ensure_adjacent`].
    pub(super) adjacent: Vec<usize>,
    /// The batch whose graph `adjacent` holds; `0` before the first batch.
    pub(super) adjacent_batch: usize,
    /// The forward-tree parent and children at the start of the batch, which join the
    /// lazily built neighbor list.
    pub(super) tree_links: Vec<usize>,
    /// The planner's own per-state values.
    pub(super) ext: X,
}

impl<S, X> Node<S, X> {
    /// The admissible solution cost through this state, `f_hat = g_hat + h_hat`.
    pub(super) fn f_hat(&self) -> f64 {
        self.g_hat + self.h_hat
    }

    /// Whether the state lies in the goal region: a root of the reverse search.
    pub(super) fn in_goal(&self) -> bool {
        self.h_hat <= 0.0
    }
}

/// The problem, the stored states with their forward tree, and this batch's graph.
#[derive(Debug, Clone)]
pub(super) struct BatchGraph<Sp: StateSpace, Sa, C, N, X> {
    /// The configuration space.
    pub(super) space: Sp,
    /// Draws the batch samples.
    pub(super) sampler: Sa,
    /// Decides which states and motions are allowed.
    pub(super) checker: C,
    /// The start state.
    pub(super) start: Sp::State,
    /// The goal region.
    pub(super) goal: GoalRegion<Sp::State>,
    /// The batch size, neighbor rule and edge cap.
    pub(super) config: BitStarConfig,
    /// Every stored state; ids index this list and are never reused.
    pub(super) nodes: Vec<Node<Sp::State, X>>,
    /// An empty index, cloned for every rebuild.
    prototype: N,
    /// Index over the live (not pruned) states of the current batch.
    pub(super) nn: N,
    /// Ids of vertices inside the goal region.
    pub(super) solutions: Vec<usize>,
    /// The cost of the best solution, `c_best`; infinite until one is found.
    pub(super) best_cost: f64,
    /// The neighbor rule of the current batch.
    pub(super) neighborhood: Neighborhood,
    /// The number of batches started so far.
    pub(super) batches: usize,
    /// Radius neighbor lists carried across batches.
    neighbor_cache: NeighborCache,
}

impl<Sp, Sa, C, N, X> BatchGraph<Sp, Sa, C, N, X>
where
    Sp: StateSpace,
    Sa: Sampler<Sp::State>,
    C: ValidityChecker<Sp>,
    N: NearestNeighbors<Sp> + Clone,
    X: Default,
{
    /// Checks `config` and the start, stores the start as vertex `0` and the goal center as
    /// the first sample, if it is valid.
    ///
    /// # Errors
    ///
    /// The errors of [`BitStarConfig`] validation, and [`Error::InvalidStart`] if `start` does
    /// not check as valid.
    pub(super) fn new(
        space: Sp,
        sampler: Sa,
        checker: C,
        start: Sp::State,
        goal: GoalRegion<Sp::State>,
        config: BitStarConfig,
        nn: N,
    ) -> Result<Self, Error> {
        config.validate()?;

        let validity = checker.check_state(&start);
        if !validity.is_valid() {
            return Err(Error::InvalidStart(validity));
        }

        let mut graph = Self {
            space,
            sampler,
            checker,
            start: start.clone(),
            goal,
            config,
            nodes: Vec::new(),
            nn: nn.clone(),
            prototype: nn,
            solutions: Vec::new(),
            best_cost: f64::INFINITY,
            neighborhood: Neighborhood::Radius(0.0),
            batches: 0,
            neighbor_cache: NeighborCache::default(),
        };
        graph.plant_root(start);

        Ok(graph)
    }

    /// Stores the start as vertex `0` and the goal center as the first sample, if it is valid.
    fn plant_root(&mut self, start: Sp::State) {
        let root = self.push_node(start, Kind::Vertex);
        self.nodes[root].g = 0.0;

        if self.nodes[root].in_goal() {
            self.solutions.push(root);
            self.best_cost = 0.0;
        }

        let center = self.goal.center().clone();
        if self.checker.check_state(&center).is_valid() {
            self.push_node(center, Kind::Sample);
        }
    }

    /// Stores a new state of the given kind with its heuristics and returns its id.
    fn push_node(&mut self, state: Sp::State, kind: Kind) -> usize {
        let id = self.nodes.len();
        let center_distance = self.space.distance(&state, self.goal.center());

        self.nodes.push(Node {
            g_hat: self.space.distance(&self.start, &state),
            h_hat: (center_distance - self.goal.tolerance()).max(0.0),
            g: f64::INFINITY,
            state,
            kind,
            parent: None,
            children: Vec::new(),
            adjacent: Vec::new(),
            adjacent_batch: 0,
            tree_links: Vec::new(),
            ext: X::default(),
        });

        id
    }

    /// Starts the next batch: prunes once a solution exists, samples, reindexes and fixes
    /// the batch graph.
    pub(super) fn next_batch<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        self.batches += 1;

        if self.best_cost.is_finite() {
            self.prune();
        }

        let first_new = self.nodes.len();
        self.sample_batch(rng);
        self.reindex();
        self.build_graph(first_new);
    }

    /// Removes the samples and vertices that cannot improve the solution, as BIT* does
    /// (Gammell et al. 2020, Algorithm 3); vertices that still could are kept as samples.
    fn prune(&mut self) {
        let c = self.best_cost;

        for node in &mut self.nodes {
            if node.kind == Kind::Sample && node.f_hat() >= c {
                node.kind = Kind::Pruned;
            }
        }

        let mut vertices: Vec<usize> = (0..self.nodes.len())
            .filter(|&id| self.nodes[id].kind == Kind::Vertex)
            .collect();
        vertices.sort_by(|&a, &b| self.nodes[a].g.total_cmp(&self.nodes[b].g).then(a.cmp(&b)));

        // Rounding can lift the solution's own branch above `c` (a goal vertex on the
        // region's rim, in line with the start, gets `h_hat(start) > c`); never cut it.
        let mut on_solution = alloc::vec![false; self.nodes.len()];
        let mut current = self.best_solution();
        while let Some(id) = current {
            on_solution[id] = true;
            current = self.nodes[id].parent;
        }

        for v in vertices {
            let node = &self.nodes[v];
            if node.kind == Kind::Vertex
                && !on_solution[v]
                && (node.f_hat() > c || node.g + node.h_hat > c)
            {
                self.disconnect(v);
            }
        }

        // Children of a pruned vertex cost more than the solution too, so the sweep above
        // removed them as well; this keeps the tree whole should rounding disagree.
        for v in 1..self.nodes.len() {
            if self.nodes[v].kind == Kind::Vertex && self.nodes[v].parent.is_none() {
                self.disconnect(v);
            }
        }

        self.solutions
            .retain(|&id| self.nodes[id].kind == Kind::Vertex);
    }

    /// Removes vertex `v` from the tree; it becomes a sample if it could still improve the
    /// solution and is pruned otherwise. Its children are handled by the caller's loop.
    fn disconnect(&mut self, v: usize) {
        if let Some(parent) = self.nodes[v].parent.take() {
            self.nodes[parent].children.retain(|&child| child != v);
        }

        let recycle = self.nodes[v].f_hat() < self.best_cost;
        let node = &mut self.nodes[v];
        node.kind = if recycle { Kind::Sample } else { Kind::Pruned };
        node.g = f64::INFINITY;

        for child in core::mem::take(&mut node.children) {
            self.nodes[child].parent = None;
        }
    }

    /// Draws up to `batch_size` valid samples that can improve the solution.
    fn sample_batch<R: Rng + ?Sized>(&mut self, rng: &mut R) {
        for _ in 0..self.config.batch_size {
            let informed = Informed {
                space: &self.space,
                checker: &self.checker,
                start: &self.start,
                goal: &self.goal,
                best_cost: self.best_cost,
            };

            if let Some(state) = informed.draw(&mut self.sampler, rng) {
                self.push_node(state, Kind::Sample);
            }
        }
    }

    /// Rebuilds the index over all vertices and samples and fixes this batch's neighbor rule
    /// from `q = |V| + |X_unconn|`, the new batch included, and the informed set's measure.
    fn reindex(&mut self) {
        self.nn = self.prototype.clone();
        let mut live: usize = 0;

        for (id, node) in self.nodes.iter().enumerate() {
            if node.kind != Kind::Pruned {
                self.nn.insert(id, node.state.clone());
                live += 1;
            }
        }

        // As OMPL: every informed state counts, the new batch included, and the radius
        // shrinks with the informed set's measure.
        let q = live.max(2);
        let bound = self.best_cost + self.goal.tolerance();
        let fraction = self
            .sampler
            .informed_fraction(&self.space, &self.start, self.goal.center(), bound)
            .unwrap_or(1.0);
        self.neighborhood =
            Neighborhood::for_batch(&self.config, q, self.space.dimension(), fraction);
    }

    /// Fixes this batch's graph: every state's RGG neighbors, symmetrized, plus the forward
    /// tree edges, which may be longer than this batch's radius.
    ///
    /// With a connection radius the neighbor lists are built lazily, on a state's first
    /// expansion in the batch ([`BatchGraph::ensure_adjacent`]), as OMPL does: a radius query
    /// from `x` finds exactly the states whose own query would find `x` (the distance is
    /// symmetric), so a lazy list equals the symmetrized one, and states the searches never
    /// touch cost nothing. `k` nearest neighbors are not symmetric, and neither are radius
    /// neighborhoods in a space that is not [symmetric](StateSpace::is_symmetric), such as
    /// [`Dubins`](crate::Dubins); those build the whole symmetrized graph here.
    fn build_graph(&mut self, first_new: usize) {
        for node in &mut self.nodes {
            node.adjacent.clear();
            node.tree_links.clear();
        }

        for id in 0..self.nodes.len() {
            if self.nodes[id].kind == Kind::Pruned {
                continue;
            }

            if let Some(parent) = self.nodes[id].parent {
                self.nodes[id].tree_links.push(parent);
                self.nodes[parent].tree_links.push(id);
            }
        }

        match self.neighborhood {
            Neighborhood::Radius(radius) if self.space.is_symmetric() => {
                self.carry_neighbor_lists(radius, first_new);
            }
            _ => self.build_symmetrized_graph(),
        }
    }

    /// Builds every live state's neighbor list for the `k`-nearest rule: its own neighbors,
    /// those that have it as a neighbor, and its tree links.
    fn build_symmetrized_graph(&mut self) {
        let mut near = Vec::new();

        for id in 0..self.nodes.len() {
            if self.nodes[id].kind == Kind::Pruned {
                continue;
            }

            self.query_neighbors(id, &mut near);

            for &other in &near {
                self.nodes[id].adjacent.push(other);
                self.nodes[other].adjacent.push(id);
            }
        }

        for id in 0..self.nodes.len() {
            let links = core::mem::take(&mut self.nodes[id].tree_links);
            let node = &mut self.nodes[id];
            node.adjacent.extend_from_slice(&links);
            node.adjacent.sort_unstable();
            node.adjacent.dedup();
            node.adjacent_batch = self.batches;
            node.tree_links = links;
        }
    }

    /// Starts this batch's radius lists: when the radius did not grow, the new samples are
    /// queried now and joined to the lists of their older neighbors, so earlier lists stay
    /// usable ([`NeighborCache`]); otherwise every list is rebuilt on first use.
    fn carry_neighbor_lists(&mut self, radius: f64, first_new: usize) {
        let states = self.nodes.len();
        if !self
            .neighbor_cache
            .begin_batch(self.batches, states, radius)
        {
            return;
        }

        let mut near = Vec::new();
        for sample in first_new..states {
            if self.nodes[sample].kind == Kind::Pruned {
                continue;
            }

            self.query_neighbors(sample, &mut near);
            near.sort_unstable();
            self.neighbor_cache.store(sample, &near);

            let older = near.partition_point(|&id| id < first_new);
            self.neighbor_cache.add_sample(sample, &near[..older]);
        }
    }

    /// Builds the neighbor list of `x` for this batch unless it is built already: the live
    /// states within the connection radius, from the carried list or a fresh query, and the
    /// tree links, sorted. Pruned states have none.
    pub(super) fn ensure_adjacent(&mut self, x: usize) {
        if self.nodes[x].adjacent_batch == self.batches {
            return;
        }

        let mut near = core::mem::take(&mut self.nodes[x].adjacent);
        near.clear();

        if self.nodes[x].kind != Kind::Pruned {
            self.radius_neighbors(x, &mut near);
            merge_sorted(&mut near, &self.nodes[x].tree_links);
        }

        let node = &mut self.nodes[x];
        node.adjacent = near;
        node.adjacent_batch = self.batches;
    }

    /// Replaces `out` with the radius neighbors of `x` in this batch: its carried list,
    /// narrowed to the live states within the radius and the edge cap exactly as a fresh query
    /// would select them, or a fresh query that becomes its list.
    fn radius_neighbors(&mut self, x: usize, out: &mut Vec<usize>) {
        let Neighborhood::Radius(radius) = self.neighborhood else {
            unreachable!("only radius neighborhoods are built lazily");
        };

        let (nodes, space, cap) = (&self.nodes, &self.space, self.config.max_radius);
        let keep = |y: usize| {
            let (from, to) = (&nodes[x].state, &nodes[y].state);

            y != x
                && nodes[y].kind != Kind::Pruned
                && space.distance(to, from) <= radius
                && (cap.is_infinite() || space.distance(from, to) <= cap)
        };

        if let Some(list) = self.neighbor_cache.filter(x, keep) {
            out.extend_from_slice(list);
        } else {
            self.query_neighbors(x, out);
            out.sort_unstable();
            self.neighbor_cache.store(x, out);
        }
    }

    /// Replaces `out` with the batch neighbors of `id` under the batch rule, without `id`
    /// itself and without edges longer than `max_radius`.
    fn query_neighbors(&self, id: usize, out: &mut Vec<usize>) {
        let cap = self.config.max_radius;
        let keep = |x| x != id && (cap.is_infinite() || self.cost_estimate(id, x) <= cap);

        self.neighborhood
            .query(&self.space, &self.nn, &self.nodes[id].state, keep, out);
    }
}

impl<Sp: StateSpace, Sa, C, N, X> BatchGraph<Sp, Sa, C, N, X> {
    /// The admissible edge cost estimate `c_hat(from, to) = d(from, to)`, in travel direction.
    pub(super) fn cost_estimate(&self, from: usize, to: usize) -> f64 {
        self.space
            .distance(&self.nodes[from].state, &self.nodes[to].state)
    }

    /// Sets `best_cost` to the cheapest solution's cost.
    pub(super) fn update_best_cost(&mut self) {
        self.best_cost = self
            .solutions
            .iter()
            .map(|&id| self.nodes[id].g)
            .fold(f64::INFINITY, f64::min);
    }

    /// The cheapest goal vertex, the lowest id among equally cheap ones, or `None`.
    fn best_solution(&self) -> Option<usize> {
        self.solutions
            .iter()
            .copied()
            .min_by(|&a, &b| self.nodes[a].g.total_cmp(&self.nodes[b].g).then(a.cmp(&b)))
    }

    /// The cheapest path found so far, or `None` if no vertex reached the goal yet.
    pub(super) fn best_path(&self) -> Option<Path<Sp::State>> {
        let best = self.best_solution()?;
        let mut states = Vec::new();
        let mut current = Some(best);

        while let Some(id) = current {
            states.push(self.nodes[id].state.clone());
            current = self.nodes[id].parent;
        }

        states.reverse();

        Some(Path {
            states,
            cost: self.nodes[best].g,
        })
    }

    /// The number of stored states of `kind`.
    pub(super) fn count(&self, kind: Kind) -> usize {
        self.nodes.iter().filter(|node| node.kind == kind).count()
    }
}
