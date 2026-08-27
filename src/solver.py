"""Define the one input/output language spoken by every solver.

The tracking controller produces one weighted conflict graph for a frame and
does not need to know how that graph is solved. This module provides compact,
immutable records for that hand-off and a template :class:`Solver` shared by
all implementations. Connected-component splitting is a solver concern: the
public input and result always describe the complete frame problem.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha256
import json
from math import fsum, isfinite
from types import MappingProxyType
from typing import Any, Mapping, final

from graph import ConflictGraph, GraphCluster, GraphNode, cluster_graph


SCHEMA_VERSION = "3.0"
SUCCESS_STATUSES = frozenset({"optimal", "completed"})


def _non_empty_string(value: str, name: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _freeze_json(value: Any, path: str) -> Any:
    """Recursively freeze a JSON-compatible diagnostics value."""

    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValueError(f"{path} mapping keys must be strings")
        return MappingProxyType(
            {
                key: _freeze_json(item, f"{path}.{key}")
                for key, item in sorted(value.items())
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    raise ValueError(f"{path} must contain only finite JSON-compatible values")


def _freeze_diagnostics(
    diagnostics: Mapping[str, object],
) -> Mapping[str, object]:
    """Freeze a diagnostics mapping without rechecking annotated object types."""

    return MappingProxyType(
        {
            key: _freeze_json(value, f"diagnostics.{key}")
            for key, value in sorted(diagnostics.items())
        }
    )


def _thaw_json(value: Any) -> Any:
    """Copy frozen JSON data into ordinary containers for serialization."""

    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _normalized_ids(values: Iterable[int], name: str) -> tuple[int, ...]:
    ids = tuple(sorted(values))
    if any(item < 0 for item in ids):
        raise ValueError(f"{name} must contain non-negative integers")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{name} must be unique")
    return ids


@dataclass(frozen=True, slots=True)
class SolverInput:
    """One complete frame graph handed unchanged to a solver.

    The graph may contain any number of disconnected components. A concrete
    solver may factor those components internally, but callers pass exactly one
    input and receive exactly one result for the frame.
    """

    problem_id: str
    frame: int
    graph: ConflictGraph
    fingerprint: str = ""

    def __post_init__(self) -> None:
        _non_empty_string(self.problem_id, "problem_id")
        if self.frame < 0:
            raise ValueError("frame must be non-negative")

        canonical = self.to_dict(include_fingerprint=False)
        expected = sha256(
            json.dumps(
                canonical,
                allow_nan=False,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if self.fingerprint and self.fingerprint != expected:
            raise ValueError("solver-input fingerprint does not match its contents")
        object.__setattr__(self, "fingerprint", expected)

    @property
    def nodes(self) -> tuple[GraphNode, ...]:
        """Return every graph node in canonical node-ID order."""

        return self.graph.nodes

    @property
    def edges(self) -> tuple[tuple[int, int], ...]:
        """Return every conflict edge in canonical order."""

        return self.graph.edges

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, object]:
        """Return the canonical JSON-safe representation used for hashing."""

        payload: dict[str, object] = {
            "schema_version": SCHEMA_VERSION,
            "problem_id": self.problem_id,
            "frame": self.frame,
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [list(edge) for edge in self.edges],
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class SolverSelection:
    """The small decision a concrete solver returns to the template."""

    selected_ids: tuple[int, ...] = ()
    status: str = "completed"
    diagnostics: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "selected_ids",
            _normalized_ids(self.selected_ids, "selected_ids"),
        )
        _non_empty_string(self.status, "status")
        object.__setattr__(
            self,
            "diagnostics",
            _freeze_diagnostics(self.diagnostics),
        )


@dataclass(frozen=True, slots=True)
class SolverResult:
    """The common immutable result returned by every solver for one frame."""

    problem_id: str
    input_fingerprint: str
    solver_name: str
    selected_ids: tuple[int, ...]
    objective: float
    feasible: bool
    status: str
    diagnostics: Mapping[str, object]
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        _non_empty_string(self.problem_id, "problem_id")
        _non_empty_string(self.input_fingerprint, "input_fingerprint")
        _non_empty_string(self.solver_name, "solver_name")
        _non_empty_string(self.status, "status")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
        object.__setattr__(
            self,
            "selected_ids",
            _normalized_ids(self.selected_ids, "selected_ids"),
        )
        objective = float(self.objective)
        if not isfinite(objective):
            raise ValueError("objective must be finite")
        object.__setattr__(self, "objective", objective)
        object.__setattr__(
            self,
            "diagnostics",
            _freeze_diagnostics(self.diagnostics),
        )

    @property
    def successful(self) -> bool:
        """Whether the status permits the tracking controller to advance."""

        return self.status in SUCCESS_STATUSES

    def to_dict(self) -> dict[str, object]:
        """Return a detached JSON-safe representation for tables or storage."""

        return {
            "schema_version": self.schema_version,
            "problem_id": self.problem_id,
            "input_fingerprint": self.input_fingerprint,
            "solver_name": self.solver_name,
            "selected_ids": list(self.selected_ids),
            "objective": self.objective,
            "feasible": self.feasible,
            "status": self.status,
            "diagnostics": _thaw_json(self.diagnostics),
        }


def validate_result(solver_input: SolverInput, result: SolverResult) -> None:
    """Check a result against the exact immutable problem originally supplied."""

    if result.problem_id != solver_input.problem_id:
        raise ValueError("result problem_id does not match solver input")
    if result.input_fingerprint != solver_input.fingerprint:
        raise ValueError("result input_fingerprint does not match solver input")

    selected = set(result.selected_ids)
    if not selected <= set(solver_input.graph.node_ids):
        raise ValueError("result selected unknown nodes")
    if any(left in selected and right in selected for left, right in solver_input.edges):
        raise ValueError("result is not an independent set")
    if not result.feasible:
        raise ValueError("a returned selection must be marked feasible")

    objective = fsum(
        node.weight for node in solver_input.nodes if node.node_id in selected
    )
    if result.objective != objective:
        raise ValueError("result objective was not computed from original input weights")
    if not result.successful and result.selected_ids:
        raise ValueError("an unsuccessful result cannot select graph nodes")


class Solver(ABC):
    """Template shared by every maximum-weight-independent-set solver."""

    def __init__(self, solver_name: str) -> None:
        self._solver_name = _non_empty_string(solver_name, "solver_name")

    @property
    def solver_name(self) -> str:
        """Return the stable name written into every result."""

        return self._solver_name

    @final
    def solve(self, solver_input: SolverInput) -> SolverResult:
        """Solve one complete frame graph and construct the validated result."""

        solver_name = _non_empty_string(self.solver_name, "solver_name")
        selection = self._select(solver_input)

        selected = set(selection.selected_ids)
        feasible = not any(
            left in selected and right in selected for left, right in solver_input.edges
        )
        objective = fsum(
            node.weight for node in solver_input.nodes if node.node_id in selected
        )
        result = SolverResult(
            problem_id=solver_input.problem_id,
            input_fingerprint=solver_input.fingerprint,
            solver_name=solver_name,
            selected_ids=selection.selected_ids,
            objective=objective,
            feasible=feasible,
            status=selection.status,
            diagnostics=selection.diagnostics,
        )
        validate_result(solver_input, result)
        return result

    @abstractmethod
    def _select(self, solver_input: SolverInput) -> SolverSelection:
        """Choose an independent set without constructing the public result."""


class ComponentSolver(Solver):
    """Shared foundation for solvers that factor a graph into components."""

    def __init__(
        self,
        solver_name: str,
        maximum_component_nodes: int,
    ) -> None:
        super().__init__(solver_name)
        if maximum_component_nodes < 1:
            raise ValueError("maximum_component_nodes must be positive")
        self.maximum_component_nodes = maximum_component_nodes

    def _components(self, solver_input: SolverInput) -> tuple[GraphCluster, ...]:
        """Split one complete frame graph in deterministic component order."""

        return cluster_graph(solver_input.graph)

    def _problem_diagnostics(
        self,
        solver_input: SolverInput,
        components: tuple[GraphCluster, ...],
    ) -> dict[str, object]:
        """Return the component-level fields common to concrete solvers."""

        return {
            "node_count": len(solver_input.nodes),
            "edge_count": len(solver_input.edges),
            "component_count": len(components),
            "component_sizes": tuple(
                len(component.node_ids) for component in components
            ),
            "maximum_component_nodes": self.maximum_component_nodes,
        }

    def _component_edges(
        self,
        solver_input: SolverInput,
        component: GraphCluster,
    ) -> tuple[tuple[int, int], ...]:
        """Return full-graph edges induced by one component."""

        node_ids = set(component.node_ids)
        return tuple(
            edge
            for edge in solver_input.edges
            if edge[0] in node_ids and edge[1] in node_ids
        )

    def _oversized_component_ids(
        self,
        components: Iterable[GraphCluster],
    ) -> tuple[int, ...]:
        """Return components that exceed this solver's declared capacity."""

        return tuple(
            component.cluster_id
            for component in components
            if len(component.node_ids) > self.maximum_component_nodes
        )


__all__ = [
    "SCHEMA_VERSION",
    "SUCCESS_STATUSES",
    "ComponentSolver",
    "Solver",
    "SolverInput",
    "SolverResult",
    "SolverSelection",
    "validate_result",
]
