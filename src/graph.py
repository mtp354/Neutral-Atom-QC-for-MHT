"""Encode local association choices as an immutable conflict graph.

The tracking front end produces association hypotheses.  Each hypothesis is a
vertex whose weight is calculated before a solver is selected.  Two vertices
conflict when they claim the same existing track or the same observation.  This
module turns those records into a canonical representation, splits the graph
into connected components, and provides stable logical coordinates. It
deliberately contains no solver, figure-writing, or neutral-atom logic; solvers
decide internally whether and how to use its component decomposition.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import itertools
from math import ceil, cos, isfinite, pi, sin, sqrt
from typing import Protocol


@dataclass(frozen=True, slots=True)
class GraphNode:
    """One weighted association hypothesis in a conflict graph.

    ``node_id`` is the hypothesis identifier exposed to solvers. Missed
    detections are updated outside the graph and therefore have no graph node.
    """

    node_id: int
    weight: float
    track_id: int
    observation_id: int

    def __post_init__(self) -> None:
        if self.node_id < 0:
            raise ValueError("node_id must be a non-negative integer")
        if not isfinite(float(self.weight)):
            raise ValueError("weight must be a finite real number")
        if self.track_id < 0:
            raise ValueError("track_id must be a non-negative integer")
        if self.observation_id < 0:
            raise ValueError("observation_id must be a non-negative integer")
        object.__setattr__(self, "node_id", int(self.node_id))
        object.__setattr__(self, "weight", float(self.weight))
        object.__setattr__(self, "track_id", int(self.track_id))
        object.__setattr__(self, "observation_id", int(self.observation_id))

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-safe representation with stable field names."""

        return {
            "node_id": self.node_id,
            "weight": self.weight,
            "track_id": self.track_id,
            "observation_id": self.observation_id,
        }


@dataclass(frozen=True, slots=True)
class ConflictGraph:
    """Canonical undirected graph of mutually exclusive hypotheses."""

    nodes: tuple[GraphNode, ...]
    edges: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        nodes = tuple(self.nodes)
        node_ids = [node.node_id for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("node_id values must be unique")
        known_ids = set(node_ids)

        normalized_edges: list[tuple[int, int]] = []
        for edge in tuple(self.edges):
            if len(edge) != 2:
                raise ValueError("each edge must contain exactly two node IDs")
            left, right = edge
            left, right = int(left), int(right)
            if left == right:
                raise ValueError("self-loop edges are not permitted")
            if left not in known_ids or right not in known_ids:
                raise ValueError("edge endpoint does not identify a graph node")
            normalized_edges.append((min(left, right), max(left, right)))

        if len(normalized_edges) != len(set(normalized_edges)):
            raise ValueError("duplicate undirected edges are not permitted")

        object.__setattr__(self, "nodes", tuple(sorted(nodes, key=lambda node: node.node_id)))
        object.__setattr__(self, "edges", tuple(sorted(normalized_edges)))

    @property
    def node_ids(self) -> tuple[int, ...]:
        """Node IDs in canonical ascending order."""

        return tuple(node.node_id for node in self.nodes)

    def node(self, node_id: int) -> GraphNode:
        """Return one node, raising ``KeyError`` for an unknown identifier."""

        wanted = int(node_id)
        for node in self.nodes:
            if node.node_id == wanted:
                return node
        raise KeyError(node_id)

    def neighbors(self, node_id: int) -> tuple[int, ...]:
        """Return the adjacent node IDs in canonical order."""

        self.node(node_id)
        wanted = int(node_id)
        adjacent: list[int] = []
        for left, right in self.edges:
            if left == wanted:
                adjacent.append(right)
            elif right == wanted:
                adjacent.append(left)
        return tuple(sorted(adjacent))


@dataclass(frozen=True, slots=True)
class GraphCluster:
    """One connected component of a :class:`ConflictGraph`."""

    cluster_id: int
    node_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.cluster_id < 0:
            raise ValueError("cluster_id must be a non-negative integer")
        node_ids = tuple(self.node_ids)
        if not node_ids:
            raise ValueError("a graph cluster must contain at least one node")
        if any(node_id < 0 for node_id in node_ids):
            raise ValueError("cluster node IDs must be non-negative integers")
        normalized = tuple(sorted(int(node_id) for node_id in node_ids))
        if len(normalized) != len(set(normalized)):
            raise ValueError("cluster node IDs must be unique")
        object.__setattr__(self, "cluster_id", int(self.cluster_id))
        object.__setattr__(self, "node_ids", normalized)


class AssociationLike(Protocol):
    """Minimum association interface accepted by :func:`encode_conflict_graph`."""

    hypothesis_id: int
    track_id: int
    observation_id: int
    weight: float


def encode_conflict_graph(associations: Iterable[AssociationLike]) -> ConflictGraph:
    """Encode association-like records as a weighted conflict graph.

    Records are intentionally accepted by attributes instead of by a concrete
    class, keeping the graph boundary independent of the tracking implementation.
    Vertices conflict when they use the same input track or the same
    observation.
    """

    nodes: list[GraphNode] = []
    for association in associations:
        try:
            nodes.append(
                GraphNode(
                    node_id=association.hypothesis_id,
                    weight=association.weight,
                    track_id=association.track_id,
                    observation_id=association.observation_id,
                )
            )
        except AttributeError as exc:
            raise TypeError(
                "association records must expose hypothesis_id, track_id, "
                "observation_id, and weight"
            ) from exc

    by_track: dict[int, list[int]] = {}
    by_observation: dict[int, list[int]] = {}
    for node in nodes:
        by_track.setdefault(node.track_id, []).append(node.node_id)
        by_observation.setdefault(node.observation_id, []).append(node.node_id)

    edges: set[tuple[int, int]] = set()
    for members in itertools.chain(by_track.values(), by_observation.values()):
        for left, right in itertools.combinations(sorted(members), 2):
            edges.add((left, right))
    return ConflictGraph(nodes=tuple(nodes), edges=tuple(edges))


def cluster_graph(graph: ConflictGraph) -> tuple[GraphCluster, ...]:
    """Return deterministic connected components of ``graph``.

    Components are ordered by their smallest node ID, and their consecutive
    ``cluster_id`` values therefore remain stable across equivalent input order.
    """

    remaining = set(graph.node_ids)
    components: list[tuple[int, ...]] = []
    while remaining:
        root = min(remaining)
        stack = [root]
        component: set[int] = set()
        while stack:
            node_id = stack.pop()
            if node_id in component:
                continue
            component.add(node_id)
            remaining.discard(node_id)
            stack.extend(
                neighbor
                for neighbor in reversed(graph.neighbors(node_id))
                if neighbor not in component
            )
        components.append(tuple(sorted(component)))

    components.sort(key=lambda component: component[0])
    return tuple(
        GraphCluster(cluster_id=index, node_ids=component)
        for index, component in enumerate(components)
    )


def logical_layout(graph: ConflictGraph) -> dict[int, tuple[float, float]]:
    """Create a deterministic two-dimensional layout without NetworkX.

    Each connected component occupies one grid cell.  Nodes in a non-trivial
    component are placed on a circle in node-ID order; isolated nodes sit at the
    cell center.  The coordinates are logical plotting units, not atom positions.
    """

    clusters = cluster_graph(graph)
    if not clusters:
        return {}

    radii = [max(0.8, len(cluster.node_ids) / (2.0 * pi)) for cluster in clusters]
    cell_spacing = 2.0 * max(radii) + 2.0
    columns = ceil(sqrt(len(clusters)))
    positions: dict[int, tuple[float, float]] = {}

    for index, (cluster, radius) in enumerate(zip(clusters, radii, strict=True)):
        row, column = divmod(index, columns)
        center_x = column * cell_spacing
        center_y = -row * cell_spacing
        if len(cluster.node_ids) == 1:
            positions[cluster.node_ids[0]] = (float(center_x), float(center_y))
            continue
        for offset, node_id in enumerate(cluster.node_ids):
            angle = pi / 2.0 - 2.0 * pi * offset / len(cluster.node_ids)
            positions[node_id] = (
                float(center_x + radius * cos(angle)),
                float(center_y + radius * sin(angle)),
            )
    return positions
