//! The sparse roadmap graph shared by SPARS and SPARS2: guards, undirected edges, connected
//! components and multi-query shortest paths.

use alloc::{collections::BTreeSet, vec::Vec};

use super::Error;
use crate::graph::{Arc, shortest_path};
use crate::{GoalRegion, NearestNeighbors, Path, StateSpace, ValidityChecker};

/// Checks the ranges SPARS and SPARS2 share: positive, finite `dense_delta < sparse_delta` and
/// a finite `stretch > 1`.
pub(super) fn validate_ranges(
    sparse_delta: f64,
    dense_delta: f64,
    stretch: f64,
) -> Result<(), Error> {
    let positive = |value: f64| value.is_finite() && value > 0.0;
    let valid = positive(sparse_delta)
        && positive(dense_delta)
        && dense_delta < sparse_delta
        && stretch.is_finite()
        && stretch > 1.0;

    if valid {
        Ok(())
    } else {
        Err(Error::InvalidRadius)
    }
}

/// Whether the motion between `a` and `b` checks as valid, also from `b` to `a` if `both`.
pub(super) fn is_edge_valid<Sp, C>(
    space: &Sp,
    checker: &C,
    a: &Sp::State,
    b: &Sp::State,
    both: bool,
) -> bool
where
    Sp: StateSpace,
    C: ValidityChecker<Sp>,
{
    checker.check_motion(space, a, b).is_valid()
        && (!both || checker.check_motion(space, b, a).is_valid())
}

/// Why a guard was added to a sparse roadmap.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GuardKind {
    /// The sample saw no guard within the sparse radius (coverage).
    Coverage,
    /// The sample saw guards from several connected components (connectivity).
    Connectivity,
    /// The sample bridged two neighboring guards that could not see each other (interface).
    Interface,
    /// The guard lies on a path added to restore the stretch bound (quality).
    Quality,
}

/// Disjoint sets with path halving and union by size; components only ever merge.
#[derive(Debug, Clone, Default)]
struct UnionFind {
    /// The parent of each element; roots point to themselves.
    parent: Vec<usize>,
    /// The size of the set rooted at each root.
    size: Vec<usize>,
}

impl UnionFind {
    /// Adds a singleton set and returns its element.
    fn push(&mut self) -> usize {
        let id = self.parent.len();
        self.parent.push(id);
        self.size.push(1);

        id
    }

    /// The root of the set containing `x`, halving the path on the way.
    fn find(&mut self, mut x: usize) -> usize {
        while self.parent[x] != x {
            self.parent[x] = self.parent[self.parent[x]];
            x = self.parent[x];
        }

        x
    }

    /// Merges the sets containing `a` and `b`.
    fn union(&mut self, a: usize, b: usize) {
        let (mut a, mut b) = (self.find(a), self.find(b));

        if a == b {
            return;
        }

        if self.size[a] < self.size[b] {
            core::mem::swap(&mut a, &mut b);
        }

        self.parent[b] = a;
        self.size[a] += self.size[b];
    }
}

/// The sparse graph: guard states, undirected edges and connected components.
#[derive(Debug, Clone)]
pub(super) struct SparseGraph<S, N> {
    /// The guard states, by guard id.
    pub(super) states: Vec<S>,
    /// Why each guard was added.
    pub(super) kinds: Vec<GuardKind>,
    /// The neighbors of each guard, in the order the edges were added.
    pub(super) adjacency: Vec<Vec<usize>>,
    /// Every edge once, as `(smaller id, larger id)`.
    edges: BTreeSet<(usize, usize)>,
    /// Connected components.
    components: UnionFind,
    /// Index over the guard states.
    pub(super) nn: N,
}

impl<S: Clone, N> SparseGraph<S, N> {
    /// An empty graph over the given empty index.
    pub(super) const fn new(nn: N) -> Self {
        Self {
            states: Vec::new(),
            kinds: Vec::new(),
            adjacency: Vec::new(),
            edges: BTreeSet::new(),
            components: UnionFind {
                parent: Vec::new(),
                size: Vec::new(),
            },
            nn,
        }
    }

    /// The number of guards.
    pub(super) fn len(&self) -> usize {
        self.states.len()
    }

    /// The number of edges.
    pub(super) fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// The number of guards of each kind: coverage, connectivity, interface, quality.
    pub(super) fn guard_counts(&self) -> [usize; 4] {
        let count = |kind| self.kinds.iter().filter(|&&k| k == kind).count();

        [
            count(GuardKind::Coverage),
            count(GuardKind::Connectivity),
            count(GuardKind::Interface),
            count(GuardKind::Quality),
        ]
    }

    /// Whether `a` and `b` share an edge.
    pub(super) fn has_edge(&self, a: usize, b: usize) -> bool {
        self.edges.contains(&(a.min(b), a.max(b)))
    }

    /// Adds the edge between `a` and `b` unless it exists or `a == b`.
    pub(super) fn connect(&mut self, a: usize, b: usize) {
        if a == b || !self.edges.insert((a.min(b), a.max(b))) {
            return;
        }

        self.adjacency[a].push(b);
        self.adjacency[b].push(a);
        self.components.union(a, b);
    }

    /// Whether `a` and `b` lie in the same connected component.
    pub(super) fn same_component(&mut self, a: usize, b: usize) -> bool {
        self.components.find(a) == self.components.find(b)
    }

    /// The guards of `visible` that lie in a different component from another one of them,
    /// in pair order: each pair `(a, b)` with `a` before `b` in different components adds both.
    /// Empty if all visible guards are connected (OMPL's connectivity links).
    pub(super) fn connectivity_links(&mut self, visible: &[usize]) -> Vec<usize> {
        let mut links = Vec::new();

        for (i, &a) in visible.iter().enumerate() {
            for &b in &visible[i + 1..] {
                if !self.same_component(a, b) {
                    links.extend([a, b]);
                }
            }
        }

        links
    }

    /// Connects the new connectivity guard `guard` to each of `links` that is not yet adjacent
    /// to it or in its component, so that each component is joined once.
    pub(super) fn connect_links(&mut self, guard: usize, links: &[usize]) {
        for &link in links {
            if !self.has_edge(guard, link) && !self.same_component(link, guard) {
                self.connect(guard, link);
            }
        }
    }

    /// The interface pair of a sample: its two closest guards `(v1, v2)` when both are visible
    /// and not adjacent, else `None`. `near` and `visible` come from [`Self::neighborhood`].
    pub(super) fn interface_pair(
        &self,
        near: &[usize],
        visible: &[usize],
    ) -> Option<(usize, usize)> {
        let [v1, v2, ..] = *visible else {
            return None;
        };

        (near[..2] == [v1, v2] && !self.has_edge(v1, v2)).then_some((v1, v2))
    }

    /// The neighbors `r'` of guard `v` other than `r` that are not adjacent to `r` (OMPL's
    /// `computeVPP`), in edge order.
    pub(super) fn non_adjacent_neighbors(&self, v: usize, r: usize) -> Vec<usize> {
        self.adjacency[v]
            .iter()
            .copied()
            .filter(|&other| other != r && !self.has_edge(other, r))
            .collect()
    }

    /// Adds a guard and returns its id.
    pub(super) fn add_guard<Sp>(&mut self, state: S, kind: GuardKind) -> usize
    where
        Sp: StateSpace<State = S>,
        N: NearestNeighbors<Sp>,
    {
        let id = self.states.len();
        self.nn.insert(id, state.clone());
        self.states.push(state);
        self.kinds.push(kind);
        self.adjacency.push(Vec::new());
        self.components.push();

        id
    }

    /// The guards within `radius` of `state`, closest first (ties by id).
    fn near_sorted<Sp>(&self, space: &Sp, state: &S, radius: f64) -> Vec<usize>
    where
        Sp: StateSpace<State = S>,
        N: NearestNeighbors<Sp>,
    {
        let mut near = Vec::new();
        self.nn.within_radius(space, state, radius, &mut near);
        let distance = |id: usize| space.distance(state, &self.states[id]);
        near.sort_by(|&a, &b| distance(a).total_cmp(&distance(b)).then(a.cmp(&b)));

        near
    }

    /// The guards within `radius` of `state`, closest first (ties by id), and the subset that
    /// `visible` accepts, in the same order.
    pub(super) fn neighborhood<Sp>(
        &self,
        space: &Sp,
        state: &S,
        radius: f64,
        visible: impl Fn(&S) -> bool,
    ) -> (Vec<usize>, Vec<usize>)
    where
        Sp: StateSpace<State = S>,
        N: NearestNeighbors<Sp>,
    {
        let near = self.near_sorted(space, state, radius);

        let visible = near
            .iter()
            .copied()
            .filter(|&id| visible(&self.states[id]))
            .collect();

        (near, visible)
    }

    /// The closest guard within `radius` of `state` (ties by id) that `visible` accepts, or
    /// `None`. Tests visibility closest first and stops at the first accepted guard.
    pub(super) fn closest_visible<Sp>(
        &self,
        space: &Sp,
        state: &S,
        radius: f64,
        visible: impl Fn(&S) -> bool,
    ) -> Option<usize>
    where
        Sp: StateSpace<State = S>,
        N: NearestNeighbors<Sp>,
    {
        let near = self.near_sorted(space, state, radius);

        near.into_iter().find(|&id| visible(&self.states[id]))
    }

    /// The cheapest path from `start` to `goal` over the graph, connecting both to every
    /// visible guard within `radius` (and to each other when they are that close), or `None`
    /// if they are not connected. Motions are checked in travel direction; graph edges were
    /// checked when they were added.
    ///
    /// # Errors
    ///
    /// [`Error::InvalidStart`] if `start` does not check as valid.
    pub(super) fn query<Sp, C>(
        &self,
        space: &Sp,
        checker: &C,
        start: &S,
        goal: &GoalRegion<S>,
        radius: f64,
    ) -> Result<Option<Path<S>>, Error>
    where
        Sp: StateSpace<State = S>,
        C: ValidityChecker<Sp>,
        N: NearestNeighbors<Sp>,
    {
        let validity = checker.check_state(start);
        if !validity.is_valid() {
            return Err(Error::InvalidStart(validity));
        }

        let n = self.len();
        let (start_id, center_id) = (n, n + 1);
        let center = goal.center();
        let (_, from_start) = self.neighborhood(space, start, radius, |guard| {
            checker.check_motion(space, start, guard).is_valid()
        });
        let mut to_center = alloc::vec![false; n];

        if checker.check_state(center).is_valid() {
            let mut near = Vec::new();
            self.nn.within_radius(space, center, radius, &mut near);

            for guard in near {
                to_center[guard] = checker
                    .check_motion(space, &self.states[guard], center)
                    .is_valid();
            }
        }

        let direct = space.distance(start, center) <= radius
            && checker.check_state(center).is_valid()
            && checker.check_motion(space, start, center).is_valid();
        let state_of = |id: usize| match id {
            id if id < n => &self.states[id],
            id if id == start_id => start,
            _ => center,
        };

        let arcs = |node: usize, out: &mut Vec<Arc>| {
            out.clear();
            let targets: Vec<usize> = if node == start_id {
                from_start
                    .iter()
                    .copied()
                    .chain(direct.then_some(center_id))
                    .collect()
            } else if node < n {
                self.adjacency[node]
                    .iter()
                    .copied()
                    .chain(to_center[node].then_some(center_id))
                    .collect()
            } else {
                Vec::new()
            };

            out.extend(targets.into_iter().map(|to| Arc {
                to,
                cost: space.distance(state_of(node), state_of(to)),
                edge: None,
            }));
        };
        let is_goal = |id: usize| id == center_id || goal.contains(space, state_of(id));
        let Some(route) = shortest_path(n + 2, start_id, is_goal, arcs) else {
            return Ok(None);
        };

        Ok(Some(Path {
            states: route.nodes.iter().map(|&id| state_of(id).clone()).collect(),
            cost: route.cost,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn union_find_merges_components() {
        let mut sets = UnionFind::default();
        let ids: Vec<usize> = (0..5).map(|_| sets.push()).collect();

        sets.union(ids[0], ids[1]);
        sets.union(ids[3], ids[4]);
        sets.union(ids[1], ids[4]);

        assert_eq!(sets.find(ids[0]), sets.find(ids[3]));
        assert_ne!(sets.find(ids[2]), sets.find(ids[0]));
    }
}
