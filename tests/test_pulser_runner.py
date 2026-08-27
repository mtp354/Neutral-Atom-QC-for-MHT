"""Protect the concrete Pulser runner without installing quantum packages."""

from __future__ import annotations

import builtins
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

import neutral_atom as neutral_atom_module
from graph import ConflictGraph, GraphNode
from neutral_atom import (
    NeutralAtomComponent,
    NeutralAtomConfig,
    PulserQutipRunner,
    QuantumSolver,
)
from solver import SolverInput


def path_problem() -> SolverInput:
    return SolverInput(
        "missing-quantum-dependency",
        2,
        ConflictGraph(
            nodes=(
                GraphNode(1, 2.0, 1, 1),
                GraphNode(2, 4.0, 2, 2),
                GraphNode(3, 3.0, 3, 3),
            ),
            edges=((1, 2), (2, 3)),
        ),
    )


def test_root_package_import_does_not_require_quantum_dependencies() -> None:
    root = Path(__file__).parents[1]
    source = root / "src"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(source), environment.get("PYTHONPATH", "")))
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    script = """
import builtins

ordinary_import = builtins.__import__

def without_quantum_dependencies(name, *args, **kwargs):
    if name.split('.')[0] in {'pulser', 'pulser_simulation', 'qutip'}:
        raise ModuleNotFoundError(f'blocked optional dependency: {name}', name=name)
    return ordinary_import(name, *args, **kwargs)

builtins.__import__ = without_quantum_dependencies
import neutral_atom_mht
assert neutral_atom_mht.QuantumSolver().solver_name == 'neutral_atom'
"""

    completed = subprocess.run(
        (sys.executable, "-c", script),
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_default_solver_reports_the_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ordinary_import = builtins.__import__

    def without_pulser(name, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "pulser":
            raise ModuleNotFoundError("No module named 'pulser'", name="pulser")
        return ordinary_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pulser)

    result = QuantumSolver().solve(path_problem())

    assert result.status == "dependency_missing"
    assert not result.successful
    assert result.selected_ids == ()
    assert result.objective == 0.0
    assert result.feasible
    assert result.diagnostics["failed_component_id"] == 0
    assert 'install ".[quantum]"' in result.diagnostics["message"]


def test_qutip_cache_is_redirected_below_local_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    fallback = tmp_path / "qutip_coeffs_1.3"
    fallback.mkdir()

    settings = SimpleNamespace(tmproot=".", coeffroot=fallback.name)
    qutip = ModuleType("qutip")
    qutip.settings = settings  # type: ignore[attr-defined]
    pulser = ModuleType("pulser")
    pulser.MockDevice = object()  # type: ignore[attr-defined]
    simulation = ModuleType("pulser_simulation")
    simulation.QutipBackendV2 = object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "qutip", qutip)
    monkeypatch.setitem(sys.modules, "pulser", pulser)
    monkeypatch.setitem(sys.modules, "pulser_simulation", simulation)
    monkeypatch.syspath_prepend(fallback.name)

    runner = PulserQutipRunner()
    _, backend_factory, _ = runner._runtime()

    expected_root = (tmp_path / ".cache" / "qutip").resolve()
    expected_coefficients = expected_root / fallback.name
    assert backend_factory is simulation.QutipBackendV2  # type: ignore[attr-defined]
    assert Path(settings.tmproot) == expected_root
    assert Path(settings.coeffroot) == expected_coefficients
    assert expected_coefficients.is_dir()
    assert not fallback.exists()
    assert Path.cwd() == tmp_path


class Drawable:
    def __init__(self) -> None:
        self.draw_calls = 0

    def draw(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        self.draw_calls += 1
        raise AssertionError("solving must not draw Pulser objects")


class FakeDetuningMap(Drawable):
    def __init__(self, weights: dict[str, float]) -> None:
        super().__init__()
        self.weights = weights


class FakeRegister(Drawable):
    def __init__(self, qubits: dict[str, tuple[float, float]]) -> None:
        super().__init__()
        self.qubits = qubits
        self.detuning_maps: list[FakeDetuningMap] = []

    def define_detuning_map(self, weights: dict[str, float]) -> FakeDetuningMap:
        detuning_map = FakeDetuningMap(weights)
        self.detuning_maps.append(detuning_map)
        return detuning_map


@dataclass(frozen=True, slots=True)
class FakeWaveform:
    duration: int
    values: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FakePulse:
    amplitude: FakeWaveform
    detuning: FakeWaveform
    phase: float


class FakeSequence(Drawable):
    def __init__(self, register: FakeRegister, device: object) -> None:
        super().__init__()
        self.register = register
        self.device = device
        self.declared_channels: list[tuple[str, str]] = []
        self.configured_maps: list[tuple[FakeDetuningMap, str]] = []
        self.pulses: list[tuple[FakePulse, str]] = []
        self.dmm_detunings: list[tuple[FakeWaveform, str]] = []

    def declare_channel(self, name: str, channel: str) -> None:
        self.declared_channels.append((name, channel))

    def config_detuning_map(self, detuning_map: FakeDetuningMap, name: str) -> None:
        self.configured_maps.append((detuning_map, name))

    def add(self, pulse: FakePulse, channel: str) -> None:
        self.pulses.append((pulse, channel))

    def add_dmm_detuning(self, waveform: FakeWaveform, name: str) -> None:
        self.dmm_detunings.append((waveform, name))


class FakePulser:
    def __init__(self) -> None:
        self.registers: list[FakeRegister] = []
        self.sequences: list[FakeSequence] = []
        self.interpolated_waveforms: list[FakeWaveform] = []
        self.constant_waveforms: list[FakeWaveform] = []
        self.created_pulses: list[FakePulse] = []

    def Register(self, qubits):  # type: ignore[no-untyped-def,override]
        register = FakeRegister(dict(qubits))
        self.registers.append(register)
        return register

    def Sequence(self, register, device):  # type: ignore[no-untyped-def,override]
        sequence = FakeSequence(register, device)
        self.sequences.append(sequence)
        return sequence

    def InterpolatedWaveform(self, duration, values):  # type: ignore[no-untyped-def]
        waveform = FakeWaveform(int(duration), tuple(float(value) for value in values))
        self.interpolated_waveforms.append(waveform)
        return waveform

    def ConstantWaveform(self, duration, value):  # type: ignore[no-untyped-def]
        waveform = FakeWaveform(int(duration), (float(value),))
        self.constant_waveforms.append(waveform)
        return waveform

    def Pulse(self, amplitude, detuning, phase):  # type: ignore[no-untyped-def]
        pulse = FakePulse(amplitude, detuning, float(phase))
        self.created_pulses.append(pulse)
        return pulse


class FakeDevice:
    interaction_coeff = 64.0


class FakeBackend:
    def __init__(self) -> None:
        self.run_calls = 0

    def run(self):  # type: ignore[no-untyped-def]
        self.run_calls += 1
        return SimpleNamespace(
            atom_order=("q2", "q0", "q1"),
            final_bitstrings={"110": np.int64(3), "001": np.int64(7)},
        )


class FakeBackendFactory:
    def __init__(self) -> None:
        self.sequences: list[FakeSequence] = []
        self.backends: list[FakeBackend] = []

    def __call__(self, sequence: FakeSequence) -> FakeBackend:
        backend = FakeBackend()
        self.sequences.append(sequence)
        self.backends.append(backend)
        return backend


def test_pulser_runner_builds_the_original_program_without_drawing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = NeutralAtomComponent(
        component_id=3,
        node_ids=(17, 42, 99),
        weights=(2.0, 5.0, 3.0),
        edges=((17, 42), (42, 99)),
        matrix=(
            (2.0, 1.0, 0.0),
            (1.0, 5.0, 1.0),
            (0.0, 1.0, 3.0),
        ),
    )
    pulser = FakePulser()
    device = FakeDevice()
    backend_factory = FakeBackendFactory()
    runner = PulserQutipRunner()
    minimize_call: dict[str, object] = {}

    def deterministic_minimize(
        function,  # type: ignore[no-untyped-def]
        initial_coordinates,
        *,
        args,
        method,
        tol,
        options,
    ):
        minimize_call.update(
            {
                "function": function,
                "initial_coordinates": np.asarray(initial_coordinates),
                "args": args,
                "method": method,
                "tol": tol,
                "options": options,
            }
        )
        return SimpleNamespace(
            x=np.array((0.0, 0.0, 1.0, 0.0, 2.0, 0.0)),
            fun=0.125,
            success=True,
        )

    monkeypatch.setattr(
        runner,
        "_runtime",
        lambda: (pulser, backend_factory, device),
    )
    monkeypatch.setattr(neutral_atom_module, "minimize", deterministic_minimize)

    random_state_before = np.random.get_state()
    run = runner.execute(component)
    random_state_after = np.random.get_state()

    assert minimize_call["function"] == runner.evaluate_mapping
    np.testing.assert_array_equal(
        minimize_call["initial_coordinates"],
        np.random.RandomState(0).random(6),
    )
    masked_matrix, supplied_device = minimize_call["args"]
    np.testing.assert_array_equal(
        masked_matrix,
        ((0.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.0, 1.0, 0.0)),
    )
    assert supplied_device is device
    assert minimize_call["method"] == "Nelder-Mead"
    assert minimize_call["tol"] == 1e-6
    assert minimize_call["options"] == {"maxiter": 200_000, "maxfev": None}

    assert len(pulser.registers) == len(pulser.sequences) == 1
    register = pulser.registers[0]
    sequence = pulser.sequences[0]
    detuning_map = register.detuning_maps[0]
    assert register.qubits == {
        "q0": (0.0, 0.0),
        "q1": (1.0, 0.0),
        "q2": (2.0, 0.0),
    }
    assert detuning_map.weights == pytest.approx(
        {"q0": 0.6, "q1": 0.0, "q2": 0.4}
    )
    assert sequence.register is register
    assert sequence.device is device
    assert sequence.declared_channels == [("rydberg_global", "rydberg_global")]
    assert sequence.configured_maps == [(detuning_map, "dmm_0")]

    assert len(pulser.interpolated_waveforms) == 2
    amplitude, detuning = pulser.interpolated_waveforms
    assert amplitude.duration == detuning.duration == 40_000
    assert amplitude.values == pytest.approx((1e-9, 10.0, 1e-9))
    assert detuning.values == pytest.approx((-10.0, 0.0, 10.0))
    pulse = pulser.created_pulses[0]
    assert pulse == FakePulse(amplitude, detuning, 0.0)
    assert sequence.pulses == [(pulse, "rydberg_global")]

    constant = pulser.constant_waveforms[0]
    assert constant == FakeWaveform(40_000, (-10.0,))
    assert sequence.dmm_detunings == [(constant, "dmm_0")]

    assert backend_factory.sequences == [sequence]
    assert backend_factory.backends[0].run_calls == 1
    assert run.component_id == 3
    assert run.node_ids == (17, 42, 99)
    assert run.atom_order == ("q2", "q0", "q1")
    assert run.bitstring_counts == (("001", 7), ("110", 3))
    assert run.coordinates == ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0))
    assert run.mapping_cost == pytest.approx(
        runner.evaluate_mapping(
            np.asarray(run.coordinates).ravel(),
            np.asarray(masked_matrix),
            device,
        )
    )
    assert run.mapping_success is True
    assert run.program is not None
    assert run.program.omega == 10.0
    assert run.program.register is register
    assert run.program.detuning_map is detuning_map
    assert run.program.sequence is sequence
    assert register.draw_calls == detuning_map.draw_calls == sequence.draw_calls == 0
    assert random_state_after[0] == random_state_before[0]
    np.testing.assert_array_equal(random_state_after[1], random_state_before[1])
    assert random_state_after[2:] == random_state_before[2:]


def test_pulser_runner_repairs_a_missing_blockade_edge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = NeutralAtomComponent(
        component_id=3,
        node_ids=(17, 42, 99),
        weights=(2.0, 5.0, 3.0),
        edges=((17, 42), (42, 99)),
        matrix=(
            (2.0, 1.0, 0.0),
            (1.0, 5.0, 1.0),
            (0.0, 1.0, 3.0),
        ),
    )
    pulser = FakePulser()
    device = FakeDevice()
    backend_factory = FakeBackendFactory()
    runner = PulserQutipRunner(
        NeutralAtomConfig(topology_restarts=1),
    )
    calls: list[object] = []
    results = iter(
        (
            SimpleNamespace(
                x=np.array((0.0, 0.0, 10.0, 0.0, 11.0, 0.0)),
                fun=0.5,
                success=True,
            ),
            SimpleNamespace(
                x=np.array((0.0, 0.0, 1.0, 0.0, 2.5, 0.0)),
                fun=0.0,
                success=True,
            ),
        )
    )

    def scripted_minimize(function, *args, **kwargs):  # type: ignore[no-untyped-def]
        calls.append(function)
        return next(results)

    monkeypatch.setattr(
        runner,
        "_runtime",
        lambda: (pulser, backend_factory, device),
    )
    monkeypatch.setattr(neutral_atom_module, "minimize", scripted_minimize)

    run = runner.execute(component)

    assert calls == [runner.evaluate_mapping, runner.evaluate_topology]
    coordinates = np.asarray(run.coordinates)
    omega, _, missing, unwanted = runner._embedding_geometry(
        coordinates,
        np.asarray(component.matrix) * ~np.eye(3, dtype=bool),
        device,
    )
    assert missing == unwanted == ()
    assert run.program is not None
    assert run.program.omega == pytest.approx(omega)
    assert backend_factory.backends[0].run_calls == 1


def test_topology_optimizer_repairs_the_fig4_leaf_embedding() -> None:
    component = NeutralAtomComponent(
        component_id=0,
        node_ids=(1, 2, 3, 4, 6),
        weights=(1.0, 1.0, 1.0, 1.0, 1.0),
        edges=((1, 2), (2, 3), (2, 4), (3, 4), (4, 6)),
        matrix=(
            (1.0, 1.0, 0.0, 0.0, 0.0),
            (1.0, 1.0, 1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0, 1.0, 0.0),
            (0.0, 1.0, 1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0, 1.0, 1.0),
        ),
    )
    initial = np.asarray(
        (
            (2.6733713718527543, 19.900852145445548),
            (-9.90134829874097, -3.636174960773828),
            (-9.844852080898136, 6.881306805648807),
            (-0.7622340383116302, 1.5712926996735987),
            (9.547403664897406, -0.5231195148669161),
        )
    )
    runner = PulserQutipRunner(NeutralAtomConfig(topology_restarts=3))
    matrix = np.asarray(component.matrix)
    masked_matrix = matrix * ~np.eye(len(matrix), dtype=bool)
    initial_cost = runner.evaluate_mapping(initial.ravel(), masked_matrix, FakeDevice())
    _, _, initial_missing, initial_unwanted = runner._embedding_geometry(
        initial,
        masked_matrix,
        FakeDevice(),
    )

    coordinates, cost, _ = runner._refine_topology(
        component,
        initial,
        matrix,
        FakeDevice(),
    )
    _, _, missing, unwanted = runner._embedding_geometry(
        coordinates,
        masked_matrix,
        FakeDevice(),
    )

    assert initial_missing == ((0, 1),)
    assert initial_unwanted == ()
    assert missing == unwanted == ()
    assert cost < initial_cost


def test_pulser_runner_rejects_an_unrepairable_blockade_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pulser = FakePulser()
    device = FakeDevice()
    backend_factory = FakeBackendFactory()
    runner = PulserQutipRunner(
        NeutralAtomConfig(topology_restarts=1),
    )
    invalid = np.array((0.0, 0.0, 10.0, 0.0, 11.0, 0.0))
    calls = 0

    def unrepairable_minimize(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal calls
        calls += 1
        return SimpleNamespace(x=invalid, fun=0.5, success=True)

    monkeypatch.setattr(
        runner,
        "_runtime",
        lambda: (pulser, backend_factory, device),
    )
    monkeypatch.setattr(neutral_atom_module, "minimize", unrepairable_minimize)

    result = QuantumSolver(runner=runner).solve(path_problem())

    assert calls == 2
    assert result.status == "embedding_failed"
    assert not result.successful
    assert "initial missing edges=((1, 2),)" in result.diagnostics["message"]
    assert pulser.registers == []
    assert backend_factory.backends == []


def test_nonconverged_finite_embedding_becomes_an_unsuccessful_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pulser = FakePulser()
    device = FakeDevice()
    backend_factory = FakeBackendFactory()
    runner = PulserQutipRunner()

    monkeypatch.setattr(
        runner,
        "_runtime",
        lambda: (pulser, backend_factory, device),
    )
    monkeypatch.setattr(
        neutral_atom_module,
        "minimize",
        lambda *args, **kwargs: SimpleNamespace(
            x=np.array((0.0, 0.0, 1.0, 0.0, 2.0, 0.0)),
            fun=0.25,
            success=False,
            message="iteration limit reached",
        ),
    )

    result = QuantumSolver(runner=runner).solve(path_problem())

    assert result.status == "embedding_failed"
    assert not result.successful
    assert result.selected_ids == ()
    assert result.objective == 0.0
    assert result.feasible
    assert result.diagnostics["failed_component_id"] == 0
    assert "iteration limit reached" in result.diagnostics["message"]
    assert "cost=0.25" in result.diagnostics["message"]
    assert pulser.registers == []
    assert backend_factory.backends == []


def test_vendor_failures_become_atomic_execution_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = PulserQutipRunner()
    monkeypatch.setattr(runner, "_runtime", lambda: (object(), object(), object()))

    def fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        raise ValueError("device rejected the sequence")

    monkeypatch.setattr(runner, "_execute_seeded", fail)

    result = QuantumSolver(runner=runner).solve(path_problem())

    assert result.status == "execution_error"
    assert not result.successful
    assert result.selected_ids == ()
    assert result.objective == 0.0
    assert result.feasible
    assert result.diagnostics["failed_component_id"] == 0
    assert "ValueError: device rejected the sequence" in result.diagnostics["message"]


class UnexpectedRunner:
    def __init__(self) -> None:
        self.components: list[NeutralAtomComponent] = []

    def execute(self, component: NeutralAtomComponent):  # type: ignore[no-untyped-def]
        self.components.append(component)
        raise AssertionError("a complete component has an exact structural solution")


def test_complete_component_uses_the_exact_clique_shortcut() -> None:
    runner = UnexpectedRunner()
    solver_input = SolverInput(
        "complete-component",
        6,
        ConflictGraph(
            nodes=(
                GraphNode(7, 5.0, 1, 1),
                GraphNode(12, 5.0, 2, 2),
                GraphNode(20, 4.0, 3, 3),
            ),
            edges=((7, 12), (7, 20), (12, 20)),
        ),
    )

    result = QuantumSolver(
        maximum_component_nodes=2,
        runner=runner,
    ).solve(solver_input)

    assert result.status == "completed"
    assert result.successful and result.feasible
    assert result.selected_ids == (7,)
    assert result.objective == 5.0
    assert runner.components == []
    assert result.diagnostics["simulated_component_count"] == 0
    assert result.diagnostics["analytical_clique_component_ids"] == (0,)
