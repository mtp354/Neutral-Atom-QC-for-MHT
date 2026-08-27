"""Check the shared solver contract and exact classical implementation.

These tests treat solver inputs as an immutable boundary: concrete solvers may
choose graph nodes, but the base class alone constructs objectives and result
records.  The examples also protect exact floating-point comparisons from
accidental rounding or tolerance-based tie handling.
"""

from __future__ import annotations

from dataclasses import replace
import json
from math import fsum
from types import MappingProxyType

import pytest

from classical_solver import ClassicalSolver
from graph import ConflictGraph, GraphNode
from solver import (
    ComponentSolver,
    Solver,
    SolverInput,
    SolverResult,
    SolverSelection,
    validate_result,
)


def worked_problem() -> SolverInput:
    """Return a small MWIS problem with the unique optimum ``(3, 6)``."""

    graph = ConflictGraph(
        nodes=tuple(
            GraphNode(node_id, weight, node_id, node_id)
            for node_id, weight in {
                2: 3.4,
                3: 9.1,
                4: 7.5,
                5: 4.8,
                6: 10.1,
            }.items()
        ),
        edges=((2, 3), (2, 4), (3, 4), (3, 5), (4, 5), (4, 6), (5, 6)),
    )
    return SolverInput(
        problem_id="worked-example",
        frame=7,
        graph=graph,
    )


class ConflictingSolver(Solver):
    """Deliberately violate independence to exercise template validation."""

    def __init__(self) -> None:
        super().__init__("invalid_conflicting_solver")

    def _select(self, solver_input: SolverInput) -> SolverSelection:
        return SolverSelection(selected_ids=(2, 3), status="completed")


def test_solver_is_an_abstract_template_and_classical_solver_inherits_it() -> None:
    with pytest.raises(TypeError):
        Solver("abstract")  # type: ignore[abstract]
    assert issubclass(ClassicalSolver, ComponentSolver)
    assert issubclass(ComponentSolver, Solver)


def test_classical_solver_inherits_identity_lifecycle_and_component_helpers() -> None:
    solver = ClassicalSolver(maximum_component_nodes=4)
    solver_input = worked_problem()
    components = solver._components(solver_input)

    assert ClassicalSolver.solve is Solver.solve
    assert ClassicalSolver.solver_name is Solver.solver_name
    assert ClassicalSolver._components is ComponentSolver._components
    assert ClassicalSolver._problem_diagnostics is ComponentSolver._problem_diagnostics
    assert ClassicalSolver._component_edges is ComponentSolver._component_edges
    assert (
        ClassicalSolver._oversized_component_ids
        is ComponentSolver._oversized_component_ids
    )
    assert solver.solver_name == "classical_exact"
    assert solver.maximum_component_nodes == 4
    assert solver._oversized_component_ids(components) == (0,)
    assert solver._problem_diagnostics(solver_input, components) == {
        "node_count": 5,
        "edge_count": 7,
        "component_count": 1,
        "component_sizes": (5,),
        "maximum_component_nodes": 4,
    }


def test_solver_input_fingerprint_is_canonical_and_json_safe() -> None:
    first = worked_problem()
    second = worked_problem()

    assert first.fingerprint == second.fingerprint
    serialized = json.loads(json.dumps(first.to_dict()))
    assert serialized["fingerprint"] == first.fingerprint
    assert [node["node_id"] for node in serialized["nodes"]] == [2, 3, 4, 5, 6]
    assert "cluster_id" not in serialized
    with pytest.raises(ValueError, match="fingerprint"):
        replace(first, fingerprint="0" * 64)


def test_classical_solver_finds_the_exact_worked_optimum() -> None:
    solver_input = worked_problem()

    result = ClassicalSolver().solve(solver_input)

    assert result.solver_name == "classical_exact"
    assert result.selected_ids == (3, 6)
    assert result.objective == pytest.approx(19.2)
    assert result.status == "optimal"
    assert result.successful
    assert result.feasible
    validate_result(solver_input, result)


def test_classical_solver_does_not_round_or_apply_a_tie_tolerance() -> None:
    graph = ConflictGraph(
        nodes=(
            GraphNode(0, 1.0, 0, 0),
            GraphNode(1, 1.0 + 5e-13, 1, 1),
        ),
        edges=((0, 1),),
    )
    solver_input = SolverInput("sub-picoweight", 1, graph)

    result = ClassicalSolver().solve(solver_input)

    assert result.selected_ids == (1,)
    assert result.objective == 1.0 + 5e-13


def test_base_template_computes_the_objective_with_fsum() -> None:
    graph = ConflictGraph(
        nodes=tuple(
            GraphNode(index, weight, index, index)
            for index, weight in enumerate((1e16, 1.0, -1e16, 2.0))
        )
    )
    solver_input = SolverInput("sum", 1, graph)

    result = ClassicalSolver().solve(solver_input)

    selected_weights = [graph.node(node_id).weight for node_id in result.selected_ids]
    assert result.objective == fsum(selected_weights)


def test_size_limit_is_an_explicit_unsuccessful_result() -> None:
    result = ClassicalSolver(maximum_component_nodes=4).solve(worked_problem())

    assert result.status == "unsupported_size"
    assert result.selected_ids == ()
    assert result.objective == 0.0
    assert result.feasible
    assert not result.successful
    assert result.diagnostics["optimal"] is False


def test_classical_solver_clusters_one_full_frame_input_internally() -> None:
    graph = ConflictGraph(
        nodes=(
            GraphNode(0, 1.0, 0, 0),
            GraphNode(1, 3.0, 0, 1),
            GraphNode(2, 5.0, 2, 2),
            GraphNode(3, 2.0, 3, 2),
            GraphNode(4, 4.0, 4, 4),
        ),
        edges=((0, 1), (2, 3)),
    )
    solver_input = SolverInput("disconnected-frame", 3, graph)

    result = ClassicalSolver(maximum_component_nodes=2).solve(solver_input)

    assert result.successful
    assert result.selected_ids == (1, 2, 4)
    assert result.objective == 12.0
    assert result.diagnostics["component_count"] == 3
    assert len(solver_input.nodes) == 5 > 2


def test_result_diagnostics_are_deeply_immutable_and_export_detached() -> None:
    solver_input = worked_problem()
    result = SolverResult(
        problem_id=solver_input.problem_id,
        input_fingerprint=solver_input.fingerprint,
        solver_name="test",
        selected_ids=(),
        objective=0.0,
        feasible=True,
        status="completed",
        diagnostics={"nested": {"items": [1, 2]}},
    )

    assert isinstance(result.diagnostics["nested"], MappingProxyType)
    with pytest.raises(TypeError):
        result.diagnostics["nested"]["changed"] = True  # type: ignore[index]
    exported = result.to_dict()
    exported["diagnostics"]["nested"]["items"].append(3)  # type: ignore[index,union-attr]
    assert result.diagnostics["nested"]["items"] == (1, 2)
    assert "runtime_seconds" not in exported


def test_template_rejects_a_concrete_solver_that_selects_conflicting_nodes() -> None:
    with pytest.raises(ValueError, match="independent set"):
        ConflictingSolver().solve(worked_problem())
