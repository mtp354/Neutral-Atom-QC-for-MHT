"""Check canonical graph encoding, clustering, validation, and layout."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pytest

from graph import (
    ConflictGraph,
    GraphCluster,
    GraphNode,
    cluster_graph,
    encode_conflict_graph,
    logical_layout,
)


@dataclass(frozen=True)
class _Association:
    hypothesis_id: int
    track_id: int
    observation_id: int
    weight: float


def _node(
    node_id: int,
    *,
    weight: float | None = None,
    track_id: int | None = None,
    observation_id: int | None = None,
) -> GraphNode:
    return GraphNode(
        node_id=node_id,
        weight=float(node_id + 1) if weight is None else weight,
        track_id=node_id if track_id is None else track_id,
        observation_id=node_id if observation_id is None else observation_id,
    )


def test_encode_adds_edges_for_shared_track_or_observation_only() -> None:
    associations = [
        _Association(3, track_id=10, observation_id=100, weight=2.0),
        _Association(1, track_id=10, observation_id=101, weight=1.0),
        _Association(2, track_id=11, observation_id=100, weight=1.5),
        _Association(4, track_id=12, observation_id=102, weight=-0.5),
        _Association(5, track_id=13, observation_id=103, weight=0.25),
    ]

    graph = encode_conflict_graph(associations)

    assert graph.node_ids == (1, 2, 3, 4, 5)
    assert graph.edges == ((1, 3), (2, 3))
    assert graph.neighbors(3) == (1, 2)
    assert graph.neighbors(4) == ()


def test_encode_accepts_duck_types_but_reports_missing_attributes() -> None:
    class Duck:
        hypothesis_id = 0
        track_id = 1
        observation_id = 2
        weight = 3.0

    assert encode_conflict_graph([Duck()]).node_ids == (0,)

    class MissingWeight:
        hypothesis_id = 0
        track_id = 1
        observation_id = 2

    with pytest.raises(TypeError, match="must expose"):
        encode_conflict_graph([MissingWeight()])


def test_connected_components_include_isolated_nodes_in_stable_order() -> None:
    graph = ConflictGraph(
        nodes=tuple(_node(node_id) for node_id in (4, 3, 2, 1, 0)),
        edges=((3, 2), (1, 0)),
    )

    clusters = cluster_graph(graph)

    assert clusters == (
        GraphCluster(0, (0, 1)),
        GraphCluster(1, (2, 3)),
        GraphCluster(2, (4,)),
    )
    assert cluster_graph(ConflictGraph(())) == ()


def test_graph_canonicalization_is_independent_of_input_order() -> None:
    nodes = (
        _node(0, weight=1.25, track_id=9, observation_id=5),
        _node(1, weight=-0.5, track_id=8, observation_id=7),
        _node(2, weight=3.0, track_id=7, observation_id=6),
    )
    first = ConflictGraph(nodes=nodes, edges=((0, 2), (0, 1)))
    second = ConflictGraph(nodes=tuple(reversed(nodes)), edges=((1, 0), (2, 0)))

    assert first == second


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("node_id", -1, "node_id"),
        ("weight", math.inf, "weight"),
        ("track_id", -1, "track_id"),
        ("observation_id", -1, "observation_id"),
    ],
)
def test_graph_node_rejects_invalid_values(field: str, value: object, message: str) -> None:
    values: dict[str, object] = {
        "node_id": 0,
        "weight": 1.0,
        "track_id": 0,
        "observation_id": 0,
    }
    values[field] = value
    with pytest.raises(ValueError, match=message):
        GraphNode(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("nodes", "edges", "message"),
    [
        ((_node(0), _node(0)), (), "unique"),
        ((_node(0),), ((0, 0),), "self-loop"),
        ((_node(0),), ((0, 1),), "endpoint"),
        ((_node(0), _node(1)), ((0, 1), (1, 0)), "duplicate"),
        ((_node(0), _node(1)), ((0, 1, 2),), "exactly two"),
    ],
)
def test_conflict_graph_rejects_invalid_topology(
    nodes: tuple[GraphNode, ...], edges: tuple[tuple[int, ...], ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ConflictGraph(nodes=nodes, edges=edges)  # type: ignore[arg-type]


def test_node_and_neighbor_lookup_are_strict() -> None:
    graph = ConflictGraph(
        nodes=tuple(_node(node_id) for node_id in range(4)),
        edges=((0, 1), (0, 2), (2, 3)),
    )

    assert graph.node(2).node_id == 2
    assert graph.neighbors(0) == (1, 2)
    with pytest.raises(KeyError):
        graph.node(99)
    with pytest.raises(KeyError):
        graph.neighbors(99)


def test_graph_cluster_validation_and_canonical_order() -> None:
    assert GraphCluster(2, (4, 1, 3)).node_ids == (1, 3, 4)
    with pytest.raises(ValueError, match="cluster_id"):
        GraphCluster(-1, (0,))
    with pytest.raises(ValueError, match="at least one"):
        GraphCluster(0, ())
    with pytest.raises(ValueError, match="unique"):
        GraphCluster(0, (1, 1))


def test_logical_layout_is_deterministic_finite_and_complete() -> None:
    graph = ConflictGraph(
        nodes=tuple(_node(node_id) for node_id in range(6)),
        edges=((0, 1), (1, 2), (3, 4)),
    )

    first = logical_layout(graph)
    second = logical_layout(ConflictGraph(tuple(reversed(graph.nodes)), tuple(reversed(graph.edges))))

    assert first == second
    assert set(first) == set(graph.node_ids)
    assert len(set(first.values())) == len(graph.nodes)
    assert all(len(position) == 2 for position in first.values())
    assert all(math.isfinite(coordinate) for position in first.values() for coordinate in position)
    assert logical_layout(ConflictGraph(())) == {}
