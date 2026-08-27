"""Opt-in smoke test for the real Pulser/QuTiP simulation stack.

The original 40,000 ns pulse and coordinate search are too expensive for the
ordinary unit suite. After installing ``.[quantum]``, run this test explicitly
from PowerShell with::

    $env:NEUTRAL_ATOM_INTEGRATION="1"
    python -m pytest tests/test_neutral_atom_integration.py -q
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from graph import ConflictGraph, GraphNode
from hpc import HPC, HPCConfig
from neutral_atom import QuantumSolver
from synthetic_data import QUANTUM_DEMO_DATA_CONFIG, SyntheticDataGenerator
from solver import SolverInput, validate_result


@pytest.mark.skipif(
    os.environ.get("NEUTRAL_ATOM_INTEGRATION") != "1",
    reason=(
        "set NEUTRAL_ATOM_INTEGRATION=1 to run the expensive Pulser/QuTiP smoke test"
    ),
)
def test_real_pulser_qutip_three_node_path_satisfies_the_solver_contract() -> None:
    pytest.importorskip("pulser")
    pytest.importorskip("pulser_simulation")
    solver_input = SolverInput(
        "real-pulser-path",
        0,
        ConflictGraph(
            nodes=(
                GraphNode(1, 2.0, 1, 1),
                GraphNode(2, 4.0, 2, 2),
                GraphNode(3, 3.0, 3, 3),
            ),
            edges=((1, 2), (2, 3)),
        ),
    )

    result = QuantumSolver().solve(solver_input)

    assert result.status == "completed"
    assert result.successful and result.feasible
    assert set(result.selected_ids) <= set(solver_input.graph.node_ids)
    validate_result(solver_input, result)


@pytest.mark.skipif(
    os.environ.get("NEUTRAL_ATOM_INTEGRATION") != "1",
    reason=(
        "set NEUTRAL_ATOM_INTEGRATION=1 to run the expensive Pulser/QuTiP smoke test"
    ),
)
def test_quantum_demo_data_completes_the_real_eight_frame_sequence(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pulser")
    pytest.importorskip("pulser_simulation")
    config = QUANTUM_DEMO_DATA_CONFIG
    dataset = SyntheticDataGenerator(config).generate(tmp_path)
    tracker = HPC(HPCConfig(), sequence=config.sequence)
    solver = QuantumSolver(maximum_component_nodes=8)
    simulated_runs = 0
    maximum_component_size = 0

    for frame in range(config.frame_count):
        prepared = tracker.prepare_frame(dataset.load_frame(frame), frame=frame)
        components = solver.prepare(prepared.solver_input())
        maximum_component_size = max(
            (
                maximum_component_size,
                *(len(component.node_ids) for component in components),
            )
        )
        result = tracker.solve(prepared, solver)

        assert result.status == "completed"
        assert result.successful and result.feasible
        simulated_runs += sum(
            component["execution_mode"] == "pulser_qutip"
            for component in result.diagnostics["components"]
        )
        tracker.advance(prepared, result)

    assert tracker.last_frame == config.frame_count - 1
    assert maximum_component_size <= solver.maximum_component_nodes
    assert simulated_runs >= 0
