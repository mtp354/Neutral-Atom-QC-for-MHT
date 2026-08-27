"""Human-readable orchestration for detection, tracking, and solver handoffs.

``HPC`` is the one stateful object a user needs for the tracking workflow.  It
turns an image into observations, exposes every preprocessing stage as a named
method, hands an immutable weighted graph to a :class:`~solver.Solver`,
and applies the returned choices through the same Bayesian update regardless of
which solver produced them.  A full image sequence repeats that visible
``HPC -> Solver -> HPC`` exchange one frame at a time.

Only current track states are retained between frames.  Association hypotheses
belong to one :class:`PreparedFrame` and are never accumulated into a family of
global hypotheses.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from math import isfinite
from operator import index

import numpy as np

from detection import DetectionConfig, DetectionResult, detect_frame
from filtering import (
    FilterConfig,
    filter_association_hypotheses as _filter_association_hypotheses,
    filter_tracks as _filter_tracks,
    predict_tracks as _predict_tracks,
)
from gating import GateConfig, gate_observations
from graph import (
    ConflictGraph,
    encode_conflict_graph,
    logical_layout,
)
from likelihood import (
    BayesianConfig,
    apply_bayesian_updates,
    calculate_association_hypotheses,
    probability_to_log_odds,
)
from models import (
    AssociationHypothesis,
    GatedAssociation,
    Observation,
    TrackState,
    observations_from_detections,
)
from solver import (
    Solver,
    SolverInput,
    SolverResult,
    validate_result,
)


def _frame_number(value: int) -> int:
    """Normalize an integer-like, non-negative frame number."""

    try:
        frame = index(value)
    except TypeError as exc:
        raise ValueError("frame must be a non-negative integer") from exc
    if frame < 0:
        raise ValueError("frame must be a non-negative integer")
    return frame


@dataclass(frozen=True, slots=True)
class HPCConfig:
    """All declared parameters used before and after a solver call.

    Keeping the detector, Kalman filter, validation gate, and Bayesian settings
    together makes a prepared frame reproducible and lets the stale-state check
    detect configuration changes before an old solver result is applied.
    """

    seconds_per_frame: float = 1.0
    initial_velocity_std: float = 10.0
    observation_variance_px2: float = 4.0
    minimum_hypothesis_weight: float = 0.0
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    filtering: FilterConfig = field(default_factory=FilterConfig)
    gating: GateConfig = field(default_factory=GateConfig)
    bayesian: BayesianConfig = field(default_factory=BayesianConfig)

    def __post_init__(self) -> None:
        positive = (
            "seconds_per_frame",
            "initial_velocity_std",
            "observation_variance_px2",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, value)

        minimum_weight = float(self.minimum_hypothesis_weight)
        if not isfinite(minimum_weight) or minimum_weight < 0.0:
            raise ValueError(
                "minimum_hypothesis_weight must be finite and non-negative"
            )
        object.__setattr__(self, "minimum_hypothesis_weight", minimum_weight)


@dataclass(frozen=True, slots=True)
class ObservedFrame:
    """The interpretable boundary between image detection and tracking.

    ``detection`` retains the labelled image and detector diagnostics, while
    ``observations`` contains only the positions and measurement uncertainty
    consumed by the tracking stages.
    """

    detection: DetectionResult
    observations: tuple[Observation, ...]

    def __post_init__(self) -> None:
        observations = tuple(self.observations)
        if any(item.frame != self.detection.frame for item in observations):
            raise ValueError("observations must share the detected frame")
        expected = tuple(
            (event.detection_id, event.x_px, event.y_px)
            for event in self.detection.detections
        )
        actual = tuple(
            (item.observation_id, item.x, item.y)
            for item in observations
        )
        if actual != expected:
            raise ValueError("observations must represent every detection in order")
        object.__setattr__(self, "observations", observations)

    @property
    def frame(self) -> int:
        return self.detection.frame

    @property
    def sequence(self) -> str:
        return self.detection.sequence


@dataclass(frozen=True, slots=True)
class PreparedFrame:
    """One frame's immutable, solver-ready local association problem."""

    frame: int
    observations: tuple[Observation, ...]
    predicted_tracks: tuple[TrackState, ...]
    gated_associations: tuple[GatedAssociation, ...]
    hypotheses: tuple[AssociationHypothesis, ...]
    graph: ConflictGraph
    source_state_fingerprint: str
    observed_frame: ObservedFrame | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _frame_number(self.frame))
        for name in (
            "observations",
            "predicted_tracks",
            "gated_associations",
            "hypotheses",
        ):
            values = tuple(getattr(self, name))
            object.__setattr__(self, name, values)
        for name in (
            "observations",
            "predicted_tracks",
            "gated_associations",
            "hypotheses",
        ):
            if any(item.frame != self.frame for item in getattr(self, name)):
                raise ValueError(f"{name} must belong to the prepared frame")
        observation_ids = [item.observation_id for item in self.observations]
        track_ids = [item.track_id for item in self.predicted_tracks]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observation IDs must be unique within a frame")
        if len(track_ids) != len(set(track_ids)):
            raise ValueError("predicted track IDs must be unique")
        if self.graph != encode_conflict_graph(self.hypotheses):
            raise ValueError("graph must exactly encode the supplied hypotheses")
        if self.observed_frame is not None:
            if (
                self.observed_frame.frame != self.frame
                or self.observed_frame.observations != self.observations
            ):
                raise ValueError("observed_frame must be the source of observations")
        fingerprint = self.source_state_fingerprint
        if (
            len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError("source_state_fingerprint must be a SHA-256 hex digest")

    def solver_input(self) -> SolverInput:
        """Create the frame's single immutable full-graph solver input."""

        return SolverInput(
            problem_id=f"frame-{self.frame:04d}",
            frame=self.frame,
            graph=self.graph,
        )


@dataclass(frozen=True, slots=True)
class FrameResult:
    """The retained state after one chosen solver result has been applied."""

    frame: int
    tracks: tuple[TrackState, ...]
    assigned_observation_ids: tuple[int, ...]
    solver_result: SolverResult

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", _frame_number(self.frame))
        tracks = tuple(self.tracks)
        if any(track.frame != self.frame for track in tracks):
            raise ValueError("tracks must belong to the result frame")
        object.__setattr__(self, "tracks", tracks)
        assigned = tuple(self.assigned_observation_ids)
        if any(item < 1 for item in assigned):
            raise ValueError("assigned observation IDs must be positive integers")
        assigned = tuple(sorted(assigned))
        if len(assigned) != len(set(assigned)):
            raise ValueError("assigned observation IDs must be unique")
        object.__setattr__(self, "assigned_observation_ids", assigned)


@dataclass(frozen=True, slots=True)
class SequenceResult:
    """Frozen outcomes from processing an image iterable exactly once."""

    steps: tuple[FrameResult, ...]
    final_tracks: tuple[TrackState, ...]
    solver_name: str

    def __post_init__(self) -> None:
        steps = tuple(self.steps)
        final_tracks = tuple(self.final_tracks)
        frames = tuple(step.frame for step in steps)
        if any(right <= left for left, right in zip(frames, frames[1:])):
            raise ValueError("sequence-result frames must be strictly increasing")
        if steps and final_tracks != steps[-1].tracks:
            raise ValueError("final_tracks must equal the last frame's tracks")
        if not self.solver_name.strip():
            raise ValueError("solver_name must be a non-empty string")
        if any(step.solver_result.solver_name != self.solver_name for step in steps):
            raise ValueError("every sequence step must use solver_name")
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "final_tracks", final_tracks)


class HPC:
    """Coordinate readable preprocessing, solver calls, and state updates.

    The stage methods are intentionally public so a notebook can inspect one
    operation at a time.  Convenience methods compose those same operations;
    they do not use a second hidden preprocessing path.
    """

    def __init__(
        self,
        config: HPCConfig | None = None,
        *,
        sequence: str = "01",
    ) -> None:
        if not sequence.strip():
            raise ValueError("sequence must be a non-empty string")
        self.config = config or HPCConfig()
        self.sequence = sequence
        self._tracks: tuple[TrackState, ...] = ()
        self._next_track_id = 1
        self._last_frame: int | None = None

    @property
    def tracks(self) -> tuple[TrackState, ...]:
        """The only persistent per-object states retained by this instance."""

        return self._tracks

    @property
    def last_frame(self) -> int | None:
        """The most recent frame applied to state, or ``None`` before use."""

        return self._last_frame

    def observe(self, image: np.ndarray, *, frame: int) -> ObservedFrame:
        """Detect cells in one image and convert them into tracker observations."""

        frame = _frame_number(frame)
        detection = detect_frame(
            image,
            sequence=self.sequence,
            frame=frame,
            config=self.config.detection,
        )
        observations = observations_from_detections(
            detection.detections,
            variance_px2=self.config.observation_variance_px2,
        )
        return ObservedFrame(detection=detection, observations=observations)

    def predict(self, *, frame: int) -> tuple[TrackState, ...]:
        """Predict retained tracks to a later frame without using observations."""

        frame = _frame_number(frame)
        if self._last_frame is not None and frame <= self._last_frame:
            raise ValueError("frame must follow the last applied frame")
        if not self._tracks:
            return ()
        return _predict_tracks(
            self._tracks,
            frame=frame,
            seconds_per_frame=self.config.seconds_per_frame,
            config=self.config.filtering,
        )

    def gate(
        self,
        predicted_tracks: Iterable[TrackState],
        observations: Iterable[Observation],
    ) -> tuple[GatedAssociation, ...]:
        """Admit track/observation pairs inside the declared kinematic gate."""

        return gate_observations(
            tuple(predicted_tracks), tuple(observations), self.config.gating
        )

    def calculate_weights(
        self,
        predicted_tracks: Iterable[TrackState],
        gated_associations: Iterable[GatedAssociation],
    ) -> tuple[AssociationHypothesis, ...]:
        """Calculate the shared Bayesian weight of every gated local candidate."""

        return calculate_association_hypotheses(
            tuple(predicted_tracks),
            tuple(gated_associations),
            self.config.bayesian,
        )

    def filter_hypotheses(
        self,
        hypotheses: Iterable[AssociationHypothesis],
    ) -> tuple[AssociationHypothesis, ...]:
        """Remove local candidates below the declared solver-weight threshold."""

        return _filter_association_hypotheses(
            tuple(hypotheses),
            minimum_weight=self.config.minimum_hypothesis_weight,
        )

    def encode_graph(
        self,
        hypotheses: Iterable[AssociationHypothesis],
    ) -> ConflictGraph:
        """Encode local associations as a weighted mutual-exclusion graph."""

        return encode_conflict_graph(tuple(hypotheses))

    def graph_embedding(
        self,
        graph: ConflictGraph,
    ) -> dict[int, tuple[float, float]]:
        """Return deterministic logical plotting coordinates for graph nodes.

        These coordinates explain graph structure; they are not a physical
        neutral-atom register embedding.
        """

        return logical_layout(graph)

    def prepare_observations(
        self,
        observations: Iterable[Observation],
        *,
        frame: int,
    ) -> PreparedFrame:
        """Run each named preprocessing stage without changing retained state."""

        frame = _frame_number(frame)
        return self._prepare(tuple(observations), frame=frame, observed_frame=None)

    def prepare_frame(self, image: np.ndarray, *, frame: int) -> PreparedFrame:
        """Observe an image and prepare its immutable local association graph."""

        observed = self.observe(image, frame=frame)
        return self._prepare(
            observed.observations,
            frame=observed.frame,
            observed_frame=observed,
        )

    def _prepare(
        self,
        supplied_observations: tuple[Observation, ...],
        *,
        frame: int,
        observed_frame: ObservedFrame | None,
    ) -> PreparedFrame:
        """Compose the public preprocessing stages into one prepared frame."""

        ordered_observations = tuple(
            sorted(supplied_observations, key=lambda item: item.observation_id)
        )
        if any(item.frame != frame for item in ordered_observations):
            raise ValueError("every observation must belong to frame")
        observation_ids = [item.observation_id for item in ordered_observations]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observation IDs must be unique within a frame")

        predicted = self.predict(frame=frame)
        gated = self.gate(predicted, ordered_observations)
        calculated = self.calculate_weights(predicted, gated)
        hypotheses = self.filter_hypotheses(calculated)
        graph = self.encode_graph(hypotheses)
        return PreparedFrame(
            frame=frame,
            observations=ordered_observations,
            predicted_tracks=predicted,
            gated_associations=gated,
            hypotheses=hypotheses,
            graph=graph,
            source_state_fingerprint=self._state_fingerprint(),
            observed_frame=observed_frame,
        )

    def solve(self, prepared: PreparedFrame, solver: Solver) -> SolverResult:
        """Hand the complete frame graph to one solver without changing state."""

        return solver.solve(prepared.solver_input())

    def bayesian_update(
        self,
        prepared: PreparedFrame,
        solver_result: SolverResult,
    ) -> tuple[tuple[TrackState, ...], frozenset[int]]:
        """Apply shared hit/miss equations to one successful solver choice."""

        self._validate_result(prepared, solver_result)
        return apply_bayesian_updates(
            prepared.predicted_tracks,
            prepared.observations,
            prepared.hypotheses,
            solver_result.selected_ids,
            self.config.bayesian,
        )

    def filter_tracks(
        self,
        tracks: Iterable[TrackState],
    ) -> tuple[TrackState, ...]:
        """Apply the declared probability, miss-count, and track-cap rules."""

        return _filter_tracks(tuple(tracks), self.config.filtering)

    def advance(
        self,
        prepared: PreparedFrame,
        solver_result: SolverResult,
    ) -> FrameResult:
        """Atomically advance retained state using one explicitly chosen result."""

        if prepared.source_state_fingerprint != self._state_fingerprint():
            raise ValueError("prepared frame is stale relative to current HPC state")

        updated, assigned = self.bayesian_update(prepared, solver_result)
        next_track_id = self._next_track_id
        candidates = list(updated)
        for observation in prepared.observations:
            if observation.observation_id not in assigned:
                candidates.append(
                    self._initialize_track(observation, track_id=next_track_id)
                )
                next_track_id += 1
        retained = self.filter_tracks(candidates)

        self._tracks = retained
        self._next_track_id = next_track_id
        self._last_frame = prepared.frame
        return FrameResult(
            frame=prepared.frame,
            tracks=retained,
            assigned_observation_ids=tuple(sorted(assigned)),
            solver_result=solver_result,
        )

    def step(
        self,
        image: np.ndarray,
        solver: Solver,
        *,
        frame: int,
    ) -> FrameResult:
        """Process one image through every HPC, solver, and update stage."""

        prepared = self.prepare_frame(image, frame=frame)
        return self.advance(prepared, self.solve(prepared, solver))

    def step_observations(
        self,
        observations: Iterable[Observation],
        solver: Solver,
        *,
        frame: int,
    ) -> FrameResult:
        """Process precomputed observations through the same solver handoff."""

        prepared = self.prepare_observations(observations, frame=frame)
        return self.advance(prepared, self.solve(prepared, solver))

    def run_sequence(
        self,
        images: Mapping[int, np.ndarray] | Iterable[np.ndarray],
        solver: Solver,
        *,
        start_frame: int | None = None,
    ) -> SequenceResult:
        """Stream images through repeated ``HPC -> Solver -> HPC`` exchanges.

        A mapping supplies explicit frame numbers and is processed in ascending
        order.  Any other iterable is consumed once and numbered consecutively;
        numbering starts after the current state unless ``start_frame`` is
        supplied.  Outcomes are returned as immutable snapshots for inspection.
        """

        if isinstance(images, Mapping):
            if start_frame is not None:
                raise ValueError("start_frame cannot be combined with a frame mapping")
            explicit_frames = tuple(
                (_frame_number(frame), image) for frame, image in images.items()
            )
            stream = iter(sorted(explicit_frames, key=lambda item: item[0]))
        else:
            if start_frame is None:
                start = 0 if self._last_frame is None else self._last_frame + 1
            else:
                start = _frame_number(start_frame)
            stream = enumerate(images, start=start)

        steps: list[FrameResult] = []
        for frame, image in stream:
            steps.append(self.step(image, solver, frame=_frame_number(frame)))
        return SequenceResult(
            steps=tuple(steps),
            final_tracks=self._tracks,
            solver_name=solver.solver_name,
        )

    def _validate_result(
        self,
        prepared: PreparedFrame,
        solver_result: SolverResult,
    ) -> None:
        """Check that a result covers this prepared frame and can advance."""

        validate_result(prepared.solver_input(), solver_result)
        if not solver_result.successful:
            raise ValueError(
                f"cannot advance from solver status {solver_result.status!r}; "
                "choose a successful solver"
            )

    def _initialize_track(self, observation: Observation, *, track_id: int) -> TrackState:
        """Create one retained state for an observation unused by the solver."""

        position_covariance = np.asarray(observation.covariance)
        covariance = np.zeros((4, 4), dtype=float)
        covariance[:2, :2] = position_covariance
        covariance[2:, 2:] = self.config.initial_velocity_std**2 * np.eye(2)
        log_odds = probability_to_log_odds(
            self.config.bayesian.initial_existence_probability
        )
        return TrackState(
            track_id=track_id,
            frame=observation.frame,
            state=(observation.x, observation.y, 0.0, 0.0),
            covariance=tuple(map(tuple, covariance)),
            log_odds=log_odds,
            hits=1,
            misses=0,
            observation_history=((observation.frame, observation.observation_id),),
        )

    def _state_fingerprint(self) -> str:
        """Fingerprint every value that can affect application of solver output."""

        payload = {
            "config": asdict(self.config),
            "sequence": self.sequence,
            "last_frame": self._last_frame,
            "next_track_id": self._next_track_id,
            "tracks": [asdict(track) for track in self._tracks],
        }
        return sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()


# The lowercase alias follows the name requested in the project specification,
# while ``HPC`` remains the conventional Python class spelling.
hpc = HPC


__all__ = [
    "FrameResult",
    "HPC",
    "HPCConfig",
    "ObservedFrame",
    "PreparedFrame",
    "SequenceResult",
    "hpc",
]
