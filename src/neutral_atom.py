"""Run and visualize the neutral-atom MWIS attempt in one focused module.

``QuantumSolver`` accepts one complete frame graph, factors disconnected
components internally, and maps every sampled bit back to the graph's original
node identifiers. The numerical embedding and pulse construction remain close
to the original Pulser experiment. Pulser itself is loaded only by the concrete
runner, so the package and its classical solver have no quantum requirement.

Solving remains headless and does not create figures or campaign records.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from math import fsum, isfinite
import os
from pathlib import Path
import sys
from threading import Lock
from typing import Protocol

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import euclidean, pdist, squareform

from graph import GraphCluster
from solver import (
    SUCCESS_STATUSES,
    ComponentSolver,
    SolverInput,
    SolverSelection,
)


@dataclass(frozen=True, slots=True)
class NeutralAtomConfig:
    """Numerical settings inherited from the original quantum attempt."""

    random_seed: int = 0
    mapping_tolerance: float = 1e-6
    mapping_max_iterations: int = 200_000
    topology_restarts: int = 8
    topology_safety_factor: float = 1.05
    pulse_duration_ns: int = 40_000
    interaction_scale: float = 10.0
    qutip_cache_dir: Path = Path(".cache") / "qutip"

    def __post_init__(self) -> None:
        if not 0 <= self.random_seed <= 2**32 - 1:
            raise ValueError("random_seed must be between 0 and 2**32 - 1")
        if not isfinite(self.mapping_tolerance) or self.mapping_tolerance <= 0.0:
            raise ValueError("mapping_tolerance must be finite and positive")
        if self.mapping_max_iterations < 1:
            raise ValueError("mapping_max_iterations must be positive")
        if self.topology_restarts < 1:
            raise ValueError("topology_restarts must be positive")
        if (
            not isfinite(self.topology_safety_factor)
            or self.topology_safety_factor <= 1.0
        ):
            raise ValueError("topology_safety_factor must be finite and greater than 1")
        if self.pulse_duration_ns < 1:
            raise ValueError("pulse_duration_ns must be positive")
        if not isfinite(self.interaction_scale) or self.interaction_scale <= 1.0:
            raise ValueError("interaction_scale must be finite and greater than 1")


@dataclass(frozen=True, slots=True)
class NeutralAtomComponent:
    """One internally factored graph component in stable qubit order."""

    component_id: int
    node_ids: tuple[int, ...]
    weights: tuple[float, ...]
    edges: tuple[tuple[int, int], ...]
    matrix: tuple[tuple[float, ...], ...]

    @property
    def qubit_ids(self) -> tuple[str, ...]:
        """Pulser labels aligned position-for-position with ``node_ids``."""

        return tuple(f"q{index}" for index in range(len(self.node_ids)))


@dataclass(frozen=True, slots=True)
class NeutralAtomProgram:
    """Built Pulser objects and embedding metadata for optional inspection."""

    component: NeutralAtomComponent
    coordinates: tuple[tuple[float, float], ...]
    mapping_cost: float
    mapping_success: bool
    omega: float
    register: object = field(repr=False, compare=False)
    detuning_map: object = field(repr=False, compare=False)
    sequence: object = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class NeutralAtomRun:
    """Raw samples plus the graph selection decoded from one component."""

    component_id: int
    node_ids: tuple[int, ...]
    atom_order: tuple[str, ...]
    bitstring_counts: tuple[tuple[str, int], ...]
    coordinates: tuple[tuple[float, float], ...]
    mapping_cost: float
    mapping_success: bool
    program: NeutralAtomProgram | None = field(default=None, repr=False, compare=False)
    execution_mode: str = "runner"
    selected_bitstring: str = ""
    selected_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        counts = tuple(
            sorted(
                (str(bitstring), int(count))
                for bitstring, count in self.bitstring_counts
            )
        )
        if len({bitstring for bitstring, _ in counts}) != len(counts):
            raise ValueError("bitstring_counts must contain unique bitstrings")
        if any(count < 1 for _, count in counts):
            raise ValueError("bitstring counts must be positive")
        coordinates = tuple(
            (float(coordinate[0]), float(coordinate[1]))
            for coordinate in self.coordinates
        )
        if len(coordinates) != len(self.node_ids):
            raise ValueError("coordinates must align with component node IDs")
        mapping_cost = float(self.mapping_cost)
        if not isfinite(mapping_cost) or mapping_cost < 0.0:
            raise ValueError("mapping_cost must be finite and non-negative")
        object.__setattr__(self, "node_ids", tuple(self.node_ids))
        object.__setattr__(
            self,
            "atom_order",
            tuple(str(item) for item in self.atom_order),
        )
        object.__setattr__(self, "bitstring_counts", counts)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "mapping_cost", mapping_cost)
        object.__setattr__(self, "mapping_success", bool(self.mapping_success))
        object.__setattr__(self, "selected_ids", tuple(sorted(self.selected_ids)))


@dataclass(frozen=True, slots=True)
class NeutralAtomExecution:
    """One atomic full-frame execution, including inspectable component runs."""

    problem_id: str
    input_fingerprint: str
    selected_ids: tuple[int, ...]
    status: str
    runs: tuple[NeutralAtomRun, ...]
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "selected_ids", tuple(sorted(self.selected_ids)))
        object.__setattr__(self, "runs", tuple(self.runs))
        frozen = SolverSelection(diagnostics=self.diagnostics).diagnostics
        object.__setattr__(self, "diagnostics", frozen)

    def to_selection(self) -> SolverSelection:
        """Discard vendor artifacts and enter the common solver output format."""

        return SolverSelection(
            selected_ids=self.selected_ids,
            status=self.status,
            diagnostics=self.diagnostics,
        )

    @property
    def successful(self) -> bool:
        """Whether this execution can safely enter the tracking update."""

        return self.status in SUCCESS_STATUSES


class NeutralAtomRunner(Protocol):
    """Structural interface for one component-level quantum executor."""

    def execute(self, component: NeutralAtomComponent) -> NeutralAtomRun:
        """Build, run, and return samples for one graph component."""


class NeutralAtomError(RuntimeError):
    """One status-bearing failure used across neutral-atom execution."""

    def __init__(self, status: str, message: str) -> None:
        self.status = status
        super().__init__(message)


class PulserQutipRunner:
    """Build and emulate the original weighted neutral-atom QAA program."""

    backend_name = "qutip_simulation"
    _random_lock = Lock()

    def __init__(
        self,
        config: NeutralAtomConfig | None = None,
        *,
        device: object | None = None,
        backend_factory: object | None = None,
    ) -> None:
        self.config = config if config is not None else NeutralAtomConfig()
        self.device = device
        self.backend_factory = backend_factory

    @staticmethod
    def evaluate_mapping(
        new_coordinates: np.ndarray,
        matrix: np.ndarray,
        device: object,
    ) -> float:
        """Measure how closely Rydberg interactions reproduce graph edges."""

        coordinates = np.reshape(new_coordinates, (len(matrix), 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            mapped = squareform(
                device.interaction_coeff / pdist(coordinates) ** 6
            ) / 4
        return float(np.linalg.norm(mapped - matrix))

    @staticmethod
    def evaluate_topology(
        new_coordinates: np.ndarray,
        matrix: np.ndarray,
        target_edge_distance: float,
        required_nonedge_distance: float,
    ) -> float:
        """Penalize edge distortion and nonedges that are too close."""

        coordinates = np.reshape(new_coordinates, (len(matrix), 2))
        distances = squareform(pdist(coordinates))
        upper_triangle = np.triu(np.ones(matrix.shape, dtype=bool), k=1)
        edge_mask = upper_triangle & (matrix > 0.0)
        nonedge_mask = upper_triangle & (matrix == 0.0)
        edge_errors = distances[edge_mask] / target_edge_distance - 1.0
        nonedge_violations = np.maximum(
            required_nonedge_distance / target_edge_distance
            - distances[nonedge_mask] / target_edge_distance,
            0.0,
        )
        return float(
            np.dot(edge_errors, edge_errors)
            + 10.0 * np.dot(nonedge_violations, nonedge_violations)
        )

    @staticmethod
    def _center_coordinates(coordinates: np.ndarray) -> np.ndarray:
        """Remove irrelevant translation and keep the register near the origin."""

        return coordinates - np.mean(coordinates, axis=0, keepdims=True)

    @classmethod
    def _scale_edge_median(
        cls,
        coordinates: np.ndarray,
        matrix: np.ndarray,
        target_edge_distance: float,
    ) -> np.ndarray:
        """Center a candidate and scale its median logical edge to the target."""

        centered = cls._center_coordinates(coordinates)
        distances = squareform(pdist(centered))
        edge_mask = np.triu(matrix > 0.0, k=1)
        edge_distances = distances[edge_mask]
        positive = edge_distances[edge_distances > np.finfo(float).eps]
        if positive.size:
            centered *= target_edge_distance / float(np.median(positive))
        return centered

    def _topology_starts(
        self,
        coordinates: np.ndarray,
        matrix: np.ndarray,
        target_edge_distance: float,
    ) -> tuple[np.ndarray, ...]:
        """Build deterministic graph-aware and seeded starts for refinement."""

        node_count = len(matrix)
        starts = [
            self._scale_edge_median(
                coordinates,
                matrix,
                target_edge_distance,
            )
        ]

        adjacency = matrix > 0.0
        np.fill_diagonal(adjacency, False)
        laplacian = np.diag(np.sum(adjacency, axis=1)) - adjacency.astype(float)
        _, eigenvectors = np.linalg.eigh(laplacian)
        spectral = eigenvectors[:, 1:3]
        if spectral.shape[1] == 1:
            spectral = np.column_stack((spectral[:, 0], np.zeros(node_count)))
        starts.append(
            self._scale_edge_median(spectral, matrix, target_edge_distance)
        )

        angles = 2.0 * np.pi * np.arange(node_count) / node_count
        circle = np.column_stack((np.cos(angles), np.sin(angles)))
        starts.append(self._scale_edge_median(circle, matrix, target_edge_distance))

        while len(starts) < self.config.topology_restarts:
            random_start = np.random.normal(size=(node_count, 2))
            starts.append(
                self._scale_edge_median(
                    random_start,
                    matrix,
                    target_edge_distance,
                )
            )
        return tuple(starts[: self.config.topology_restarts])

    def _embedding_geometry(
        self,
        coordinates: np.ndarray,
        matrix: np.ndarray,
        device: object,
    ) -> tuple[float, float, tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
        """Return the pulse scale, blockade radius, and topology mismatches."""

        nonedge_distances: list[float] = []
        pair_distances: dict[tuple[int, int], float] = {}
        for right in range(1, matrix.shape[0]):
            for left in range(right):
                distance = float(euclidean(coordinates[right], coordinates[left]))
                pair_distances[(left, right)] = distance
                if matrix[right, left] == 0.0:
                    nonedge_distances.append(distance)
        if not nonedge_distances or min(nonedge_distances) <= 0.0:
            raise NeutralAtomError(
                "execution_error",
                "embedding has no usable nonedge separation",
            )

        interaction_coefficient = float(device.interaction_coeff)
        omega = float(
            interaction_coefficient
            / min(nonedge_distances) ** 6
            * self.config.interaction_scale
        )
        if not isfinite(omega) or omega <= 0.0:
            raise NeutralAtomError(
                "execution_error",
                "embedding produced an invalid Rabi frequency",
            )
        radius_function = getattr(device, "rydberg_blockade_radius", None)
        blockade_radius = float(
            radius_function(omega)
            if callable(radius_function)
            else (interaction_coefficient / omega) ** (1.0 / 6.0)
        )
        if not isfinite(blockade_radius) or blockade_radius <= 0.0:
            raise NeutralAtomError(
                "execution_error",
                "device produced an invalid blockade radius",
            )

        missing_edges: list[tuple[int, int]] = []
        unwanted_edges: list[tuple[int, int]] = []
        for pair, distance in pair_distances.items():
            intended = matrix[pair[0], pair[1]] > 0.0
            blocked = distance <= blockade_radius
            if intended and not blocked:
                missing_edges.append(pair)
            elif blocked and not intended:
                unwanted_edges.append(pair)
        return (
            omega,
            blockade_radius,
            tuple(missing_edges),
            tuple(unwanted_edges),
        )

    def _refine_topology(
        self,
        component: NeutralAtomComponent,
        coordinates: np.ndarray,
        matrix: np.ndarray,
        device: object,
    ) -> tuple[np.ndarray, float, float]:
        """Repair a locally optimal interaction fit and require exact topology."""

        masked_matrix = ~np.eye(matrix.shape[0], dtype=bool) * matrix
        mapping_cost = self.evaluate_mapping(coordinates.ravel(), masked_matrix, device)
        omega, _, missing, unwanted = self._embedding_geometry(
            coordinates,
            masked_matrix,
            device,
        )
        if not missing and not unwanted:
            return coordinates, mapping_cost, omega

        target_edge_distance = float(
            (float(device.interaction_coeff) / 4.0) ** (1.0 / 6.0)
        )
        required_nonedge_distance = float(
            target_edge_distance
            * self.config.interaction_scale ** (1.0 / 6.0)
            * self.config.topology_safety_factor
        )
        feasible: list[tuple[float, np.ndarray, float]] = []
        for start in self._topology_starts(
            coordinates,
            masked_matrix,
            target_edge_distance,
        ):
            refinement = minimize(
                self.evaluate_topology,
                start.ravel(),
                args=(
                    masked_matrix,
                    target_edge_distance,
                    required_nonedge_distance,
                ),
                method="L-BFGS-B",
                tol=self.config.mapping_tolerance,
                options={"maxiter": self.config.mapping_max_iterations},
            )
            candidate = self._center_coordinates(
                np.reshape(refinement.x, (len(matrix), 2))
            )
            if not np.all(np.isfinite(candidate)):
                continue
            try:
                candidate_omega, _, candidate_missing, candidate_unwanted = (
                    self._embedding_geometry(candidate, masked_matrix, device)
                )
            except NeutralAtomError:
                continue
            if candidate_missing or candidate_unwanted:
                continue
            candidate_cost = self.evaluate_mapping(
                candidate.ravel(),
                masked_matrix,
                device,
            )
            if isfinite(candidate_cost):
                feasible.append((candidate_cost, candidate, candidate_omega))

        if not feasible:
            missing_node_edges = tuple(
                (component.node_ids[left], component.node_ids[right])
                for left, right in missing
            )
            unwanted_node_edges = tuple(
                (component.node_ids[left], component.node_ids[right])
                for left, right in unwanted
            )
            raise NeutralAtomError(
                "embedding_failed",
                f"component {component.component_id} could not realize its exact "
                f"blockade graph after {self.config.topology_restarts} refinements; "
                f"initial missing edges={missing_node_edges}, "
                f"initial unwanted edges={unwanted_node_edges}",
            )
        best_cost, best_coordinates, best_omega = min(
            feasible,
            key=lambda candidate: candidate[0],
        )
        return best_coordinates, best_cost, best_omega

    def execute(self, component: NeutralAtomComponent) -> NeutralAtomRun:
        """Embed, program, and emulate one non-trivial connected component."""

        with self._random_lock:
            # QuTiP chooses its coefficient cache while importing. Keep the
            # short working-directory change in the same process-wide critical
            # section as the backend execution.
            runtime = self._runtime()
            random_state = np.random.get_state()
            try:
                np.random.seed(self.config.random_seed)
                try:
                    return self._execute_seeded(component, runtime)
                except NeutralAtomError:
                    raise
                except Exception as exc:
                    raise NeutralAtomError(
                        "execution_error",
                        f"component {component.component_id} vendor execution failed: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
            finally:
                np.random.set_state(random_state)

    def _execute_seeded(
        self,
        component: NeutralAtomComponent,
        runtime: tuple[object, object, object],
    ) -> NeutralAtomRun:
        """Execute while NumPy's process RNG is scoped to the configured seed."""

        pulser, backend_factory, device = runtime
        matrix = np.asarray(component.matrix, dtype=float)
        initial_coordinates = np.random.random(len(matrix) * 2)

        mapping = minimize(
            self.evaluate_mapping,
            initial_coordinates,
            args=(~np.eye(matrix.shape[0], dtype=bool) * matrix, device),
            method="Nelder-Mead",
            tol=self.config.mapping_tolerance,
            options={
                "maxiter": self.config.mapping_max_iterations,
                "maxfev": None,
            },
        )
        coordinates_array = np.reshape(mapping.x, (len(matrix), 2))
        mapping_cost = float(mapping.fun)
        if not np.all(np.isfinite(coordinates_array)) or not isfinite(mapping_cost):
            raise NeutralAtomError(
                "execution_error",
                f"component {component.component_id} produced a non-finite embedding"
            )
        if not mapping.success:
            raise NeutralAtomError(
                "embedding_failed",
                f"component {component.component_id} embedding did not converge: "
                f"{mapping.message} (cost={mapping_cost:.6g})"
            )
        coordinates_array, mapping_cost, omega = self._refine_topology(
            component,
            coordinates_array,
            matrix,
            device,
        )
        coordinates = tuple(
            (float(coordinate[0]), float(coordinate[1]))
            for coordinate in coordinates_array
        )

        qubits = dict(zip(component.qubit_ids, coordinates, strict=True))
        register = pulser.Register(qubits)
        sequence = pulser.Sequence(register, device)
        sequence.declare_channel("rydberg_global", "rydberg_global")

        node_weights = np.diag(matrix)
        maximum_weight = float(np.max(node_weights))
        if maximum_weight > 0.0:
            normalized_weights = np.clip(node_weights / maximum_weight, 0.0, 1.0)
        else:
            normalized_weights = np.zeros_like(node_weights)
        detuning_weights = 1.0 - normalized_weights
        detuning_map = register.define_detuning_map(
            {
                qubit_id: float(detuning_weights[index])
                for index, qubit_id in enumerate(component.qubit_ids)
            }
        )
        sequence.config_detuning_map(detuning_map, "dmm_0")

        delta_initial = -omega
        delta_final = -delta_initial
        duration = self.config.pulse_duration_ns

        adiabatic_pulse = pulser.Pulse(
            pulser.InterpolatedWaveform(duration, [1e-9, omega, 1e-9]),
            pulser.InterpolatedWaveform(
                duration,
                [delta_initial, 0.0, delta_final],
            ),
            0,
        )
        sequence.add(adiabatic_pulse, "rydberg_global")
        sequence.add_dmm_detuning(
            pulser.ConstantWaveform(duration, -delta_final),
            "dmm_0",
        )

        backend = backend_factory(sequence)
        results = backend.run()
        counts = tuple(
            sorted(
                (str(bitstring), int(count))
                for bitstring, count in results.final_bitstrings.items()
            )
        )
        program = NeutralAtomProgram(
            component=component,
            coordinates=coordinates,
            mapping_cost=mapping_cost,
            mapping_success=bool(mapping.success),
            omega=omega,
            register=register,
            detuning_map=detuning_map,
            sequence=sequence,
        )
        return NeutralAtomRun(
            component_id=component.component_id,
            node_ids=component.node_ids,
            atom_order=tuple(str(atom_id) for atom_id in results.atom_order),
            bitstring_counts=counts,
            coordinates=coordinates,
            mapping_cost=mapping_cost,
            mapping_success=bool(mapping.success),
            program=program,
            execution_mode="pulser_qutip",
        )

    def _runtime(self) -> tuple[object, object, object]:
        """Resolve optional vendor objects only when quantum execution starts."""

        try:
            import pulser
        except ModuleNotFoundError as exc:
            if exc.name != "pulser":
                raise
            raise NeutralAtomError(
                "dependency_missing",
                'Pulser is required for neutral-atom simulation; install ".[quantum]"'
            ) from exc

        if self.backend_factory is None:
            try:
                backend_factory = self._load_qutip_backend()
            except ModuleNotFoundError as exc:
                if exc.name != "pulser_simulation":
                    raise
                raise NeutralAtomError(
                    "dependency_missing",
                    "pulser-simulation is required for neutral-atom simulation; "
                    'install ".[quantum]"'
                ) from exc
        else:
            backend_factory = self.backend_factory
        device = self.device if self.device is not None else pulser.MockDevice
        return pulser, backend_factory, device

    def _load_qutip_backend(self) -> object:
        """Import QuTiP through Pulser with its cache below ``.cache/qutip``."""

        original_directory = Path.cwd().resolve()
        cache_root = Path(self.config.qutip_cache_dir)
        if not cache_root.is_absolute():
            cache_root = original_directory / cache_root
        cache_root = cache_root.resolve()
        cache_root.mkdir(parents=True, exist_ok=True)

        previous_coefficient_root = self._active_coefficient_root(original_directory)
        os.chdir(cache_root)
        try:
            import pulser_simulation

            self._redirect_qutip_cache(cache_root)
        finally:
            os.chdir(original_directory)

        self._remove_empty_root_cache(previous_coefficient_root, original_directory)
        return pulser_simulation.QutipBackendV2

    @staticmethod
    def _active_coefficient_root(base_directory: Path) -> Path | None:
        """Resolve an already-imported QuTiP cache before redirecting it."""

        qutip = sys.modules.get("qutip")
        if qutip is None:
            return None
        coefficient_root = Path(str(qutip.settings.coeffroot))
        if not coefficient_root.is_absolute():
            coefficient_root = base_directory / coefficient_root
        return coefficient_root.resolve()

    @staticmethod
    def _redirect_qutip_cache(cache_root: Path) -> None:
        """Replace QuTiP's relative fallback with one stable absolute path."""

        qutip = sys.modules.get("qutip")
        if qutip is None:
            return
        previous_entry = str(qutip.settings.coeffroot)
        coefficient_name = Path(previous_entry).name
        if not coefficient_name.startswith("qutip_coeffs_"):
            coefficient_name = "qutip_coeffs"
        coefficient_root = cache_root / coefficient_name
        coefficient_root.mkdir(exist_ok=True)
        qutip.settings.tmproot = str(cache_root)
        qutip.settings.coeffroot = str(coefficient_root)
        if previous_entry != str(coefficient_root) and previous_entry in sys.path:
            sys.path.remove(previous_entry)

    @staticmethod
    def _remove_empty_root_cache(
        coefficient_root: Path | None,
        repository_root: Path,
    ) -> None:
        """Remove only QuTiP's empty, generated working-directory fallback."""

        if (
            coefficient_root is not None
            and coefficient_root.parent == repository_root
            and coefficient_root.name.startswith("qutip_coeffs_")
        ):
            try:
                coefficient_root.rmdir()
            except OSError:
                pass


class QuantumSolver(ComponentSolver):
    """Orchestrate component experiments and return one full-frame selection."""

    def __init__(
        self,
        config: NeutralAtomConfig | None = None,
        *,
        maximum_component_nodes: int = 16,
        runner: NeutralAtomRunner | None = None,
    ) -> None:
        super().__init__(
            solver_name="neutral_atom",
            maximum_component_nodes=maximum_component_nodes,
        )
        self.config = config if config is not None else NeutralAtomConfig()
        self.runner = runner if runner is not None else PulserQutipRunner(self.config)

    def prepare(self, solver_input: SolverInput) -> tuple[NeutralAtomComponent, ...]:
        """Create stable component matrices without loading Pulser."""

        return self._prepare_components(solver_input, self._components(solver_input))

    def _prepare_components(
        self,
        solver_input: SolverInput,
        clusters: Sequence[GraphCluster],
    ) -> tuple[NeutralAtomComponent, ...]:
        """Translate inherited graph components into quantum program inputs."""

        return tuple(
            self._component(solver_input, cluster)
            for cluster in clusters
        )

    def execute(self, solver_input: SolverInput) -> NeutralAtomExecution:
        """Execute every component atomically and retain optional artifacts."""

        clusters = self._components(solver_input)
        components = self._prepare_components(solver_input, clusters)
        simulated_components = tuple(
            component for component in components if not self._is_clique(component)
        )
        simulated_clusters = tuple(
            cluster
            for cluster, component in zip(clusters, components, strict=True)
            if not self._is_clique(component)
        )
        oversized = self._oversized_component_ids(simulated_clusters)
        negative_weight_components = tuple(
            component.component_id
            for component in simulated_components
            if any(weight < 0.0 for weight in component.weights)
        )
        base_diagnostics: dict[str, object] = {
            **self._problem_diagnostics(solver_input, clusters),
            "backend": getattr(self.runner, "backend_name", "injected_runner"),
            "simulated_component_count": len(simulated_components),
            "analytical_clique_component_ids": tuple(
                component.component_id
                for component in components
                if self._is_clique(component)
            ),
            "optimal": False,
        }
        if oversized:
            return NeutralAtomExecution(
                problem_id=solver_input.problem_id,
                input_fingerprint=solver_input.fingerprint,
                selected_ids=(),
                status="unsupported_size",
                runs=(),
                diagnostics={
                    **base_diagnostics,
                    "components": (),
                    "oversized_component_ids": oversized,
                },
            )
        if negative_weight_components:
            return NeutralAtomExecution(
                problem_id=solver_input.problem_id,
                input_fingerprint=solver_input.fingerprint,
                selected_ids=(),
                status="unsupported_weights",
                runs=(),
                diagnostics={
                    **base_diagnostics,
                    "components": (),
                    "negative_weight_component_ids": negative_weight_components,
                    "message": (
                        "the Pulser detuning map requires non-negative graph weights"
                    ),
                },
            )

        selected_ids: list[int] = []
        runs: list[NeutralAtomRun] = []
        component_diagnostics: list[dict[str, object]] = []
        for component in components:
            try:
                raw_run = (
                    self._clique_run(component)
                    if self._is_clique(component)
                    else self.runner.execute(component)
                )
                run, diagnostics = self._decode(component, raw_run)
            except NeutralAtomError as exc:
                return NeutralAtomExecution(
                    problem_id=solver_input.problem_id,
                    input_fingerprint=solver_input.fingerprint,
                    selected_ids=(),
                    status=exc.status,
                    runs=tuple(runs),
                    diagnostics={
                        **base_diagnostics,
                        "components": tuple(component_diagnostics),
                        "failed_component_id": component.component_id,
                        "message": str(exc),
                    },
                )
            selected_ids.extend(run.selected_ids)
            runs.append(run)
            component_diagnostics.append(diagnostics)

        return NeutralAtomExecution(
            problem_id=solver_input.problem_id,
            input_fingerprint=solver_input.fingerprint,
            selected_ids=tuple(sorted(selected_ids)),
            status="completed",
            runs=tuple(runs),
            diagnostics={
                **base_diagnostics,
                "components": tuple(component_diagnostics),
            },
        )

    def _select(self, solver_input: SolverInput) -> SolverSelection:
        return self.execute(solver_input).to_selection()

    def _component(
        self,
        solver_input: SolverInput,
        cluster: GraphCluster,
    ) -> NeutralAtomComponent:
        node_ids = cluster.node_ids
        weights = tuple(solver_input.graph.node(node_id).weight for node_id in node_ids)
        edges = self._component_edges(solver_input, cluster)
        positions = {node_id: index for index, node_id in enumerate(node_ids)}
        matrix = [[0.0 for _ in node_ids] for _ in node_ids]
        for index, weight in enumerate(weights):
            matrix[index][index] = weight
        for left, right in edges:
            left_index = positions[left]
            right_index = positions[right]
            matrix[left_index][right_index] = 1.0
            matrix[right_index][left_index] = 1.0
        return NeutralAtomComponent(
            component_id=cluster.cluster_id,
            node_ids=node_ids,
            weights=weights,
            edges=edges,
            matrix=tuple(tuple(row) for row in matrix),
        )

    @staticmethod
    def _is_clique(component: NeutralAtomComponent) -> bool:
        node_count = len(component.node_ids)
        return len(component.edges) == node_count * (node_count - 1) // 2

    @staticmethod
    def _clique_run(component: NeutralAtomComponent) -> NeutralAtomRun:
        """Resolve a clique exactly because the QAA nonedge scale is undefined."""

        best_index = max(
            range(len(component.node_ids)),
            key=lambda index: component.weights[index],
        )
        selected = component.weights[best_index] > 0.0
        bits = ["0"] * len(component.node_ids)
        if selected:
            bits[best_index] = "1"
        bitstring = "".join(bits)
        return NeutralAtomRun(
            component_id=component.component_id,
            node_ids=component.node_ids,
            atom_order=component.qubit_ids,
            bitstring_counts=((bitstring, 1),),
            coordinates=tuple(
                (float(index), 0.0) for index in range(len(component.node_ids))
            ),
            mapping_cost=0.0,
            mapping_success=True,
            execution_mode="analytical_clique",
        )

    @staticmethod
    def _decode(
        component: NeutralAtomComponent,
        run: NeutralAtomRun,
    ) -> tuple[NeutralAtomRun, dict[str, object]]:
        if run.component_id != component.component_id:
            raise ValueError("neutral-atom run component_id does not match request")
        if run.node_ids != component.node_ids:
            raise ValueError("neutral-atom run node IDs do not match request")
        if len(run.atom_order) != len(component.qubit_ids) or set(run.atom_order) != set(
            component.qubit_ids
        ):
            raise ValueError("neutral-atom run atom order does not match component qubits")

        node_for_qubit = dict(
            zip(component.qubit_ids, component.node_ids, strict=True)
        )
        atom_nodes = tuple(node_for_qubit[qubit_id] for qubit_id in run.atom_order)
        edge_set = {frozenset(edge) for edge in component.edges}
        weights = dict(zip(component.node_ids, component.weights, strict=True))
        candidates: list[tuple[float, int, tuple[int, ...], str]] = []
        invalid_samples = 0
        infeasible_samples = 0
        for bitstring, count in run.bitstring_counts:
            if len(bitstring) != len(atom_nodes) or set(bitstring) - {"0", "1"}:
                invalid_samples += count
                continue
            selected = tuple(
                sorted(
                    node_id
                    for node_id, bit in zip(atom_nodes, bitstring, strict=True)
                    if bit == "1"
                )
            )
            selected_set = set(selected)
            if any(edge <= selected_set for edge in edge_set):
                infeasible_samples += count
                continue
            objective = fsum(weights[node_id] for node_id in selected)
            candidates.append((objective, count, selected, bitstring))

        if not candidates:
            raise NeutralAtomError(
                "no_feasible_sample",
                f"component {component.component_id} returned no valid feasible sample "
                f"({invalid_samples} malformed, {infeasible_samples} conflicting)"
            )
        empty_bitstring = "0" * len(atom_nodes)
        if not any(candidate[2] == () for candidate in candidates):
            candidates.append((0.0, 0, (), empty_bitstring))
        objective, chosen_count, selected_ids, selected_bitstring = min(
            candidates,
            key=lambda candidate: (
                -candidate[0],
                -candidate[1],
                candidate[2],
                candidate[3],
            ),
        )
        decoded = replace(
            run,
            selected_bitstring=selected_bitstring,
            selected_ids=selected_ids,
        )
        diagnostics: dict[str, object] = {
            "component_id": component.component_id,
            "node_ids": component.node_ids,
            "qubit_ids": component.qubit_ids,
            "atom_order": run.atom_order,
            "node_count": len(component.node_ids),
            "edge_count": len(component.edges),
            "coordinates": run.coordinates,
            "mapping_cost": run.mapping_cost,
            "mapping_success": run.mapping_success,
            "execution_mode": run.execution_mode,
            "bitstring_counts": run.bitstring_counts,
            "sample_count": sum(count for _, count in run.bitstring_counts),
            "invalid_sample_count": invalid_samples,
            "infeasible_sample_count": infeasible_samples,
            "selected_bitstring": selected_bitstring,
            "selected_sample_count": chosen_count,
            "selected_ids": selected_ids,
            "objective": objective,
        }
        return decoded, diagnostics

__all__ = [
    "NeutralAtomComponent",
    "NeutralAtomConfig",
    "NeutralAtomError",
    "NeutralAtomExecution",
    "NeutralAtomProgram",
    "NeutralAtomRun",
    "NeutralAtomRunner",
    "PulserQutipRunner",
    "QuantumSolver",
]
