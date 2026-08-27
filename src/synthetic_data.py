"""Simulate deterministic tracking frames from explicit ground truth.

The simulator has one data path: create a clean ground-truth frame first, then
apply a two-parameter sensor model. ``gaussian_bluriness`` is the Gaussian blur
sigma in pixels, and ``grainyness`` is the additive Gaussian intensity noise
standard deviation. There is no dropout, clutter, severity sweep axis, or
real-data loader hidden behind this module.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from heapq import heappop, heappush
from math import isfinite
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi


DEFAULT_SYNTHETIC_DATA_ROOT = Path("data") / "synthetic"

_BACKGROUND_LEVEL = 58.0
_BACKGROUND_SHADING = 5.0
_CELL_SIGNAL = 175.0
_TRUTH_RADIUS_PX = 4
_REGION_FRACTION = 0.40
_TURN_SIGMA_RAD = 0.16
_NOISE_STREAM_ID = 9_973
_MAX_TRACKING_LABEL = int(np.iinfo(np.uint16).max)


def _finite_non_negative(value: float, name: str) -> float:
    normalized = float(value)
    if not isfinite(normalized) or normalized < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return normalized


def _finite_positive(value: float, name: str) -> float:
    normalized = float(value)
    if not isfinite(normalized) or normalized <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return normalized


@dataclass(frozen=True, slots=True)
class SyntheticDataConfig:
    """Configuration for one reproducible simulated tracking sequence."""

    gaussian_bluriness: float = 1.0
    grainyness: float = 3.0
    frame_count: int = 40
    object_count: int = 55
    seed: int = 0
    dataset_name: str = "SYN-MHT"
    sequence: str = "01"
    image_shape: tuple[int, int] = (576, 720)
    speed_px_per_frame: float = 4.0

    def __post_init__(self) -> None:
        try:
            image_shape = tuple(int(dimension) for dimension in self.image_shape)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "image_shape must contain two positive dimensions"
            ) from exc
        if len(image_shape) != 2 or min(image_shape) < 1:
            raise ValueError("image_shape must contain two positive dimensions")

        object.__setattr__(
            self,
            "gaussian_bluriness",
            _finite_non_negative(self.gaussian_bluriness, "gaussian_bluriness"),
        )
        object.__setattr__(
            self,
            "grainyness",
            _finite_non_negative(self.grainyness, "grainyness"),
        )
        object.__setattr__(
            self,
            "speed_px_per_frame",
            _finite_positive(self.speed_px_per_frame, "speed_px_per_frame"),
        )
        object.__setattr__(self, "frame_count", int(self.frame_count))
        object.__setattr__(self, "object_count", int(self.object_count))
        object.__setattr__(self, "seed", int(self.seed))
        object.__setattr__(self, "image_shape", image_shape)

        if self.frame_count < 1:
            raise ValueError("frame_count must be positive")
        if self.object_count < 1:
            raise ValueError("object_count must be positive")
        if not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be between 0 and 2**32 - 1")
        if self.object_count > _MAX_TRACKING_LABEL:
            raise ValueError(
                f"object_count cannot exceed the uint16 label limit "
                f"({_MAX_TRACKING_LABEL})"
            )
        if self.object_count > self.height * self.width:
            raise ValueError("object_count cannot exceed the available image pixels")
        for name, value in (
            ("dataset_name", self.dataset_name),
            ("sequence", self.sequence),
        ):
            if not value or value in {".", ".."} or Path(value).name != value:
                raise ValueError(f"{name} must be one directory name")

    @property
    def height(self) -> int:
        return int(self.image_shape[0])

    @property
    def width(self) -> int:
        return int(self.image_shape[1])


QUANTUM_DEMO_DATA_CONFIG = SyntheticDataConfig(
    gaussian_bluriness=1.0,
    grainyness=1.0,
    frame_count=8,
    object_count=4,
    seed=0,
    dataset_name="SYN-MHT-QUANTUM-v2",
    sequence="01",
    image_shape=(256, 320),
    speed_px_per_frame=3.0,
)


@dataclass(frozen=True, slots=True)
class GroundTruthFrame:
    """Noise-free labels and image signal for one frame."""

    frame: int
    positions: tuple[tuple[float, float], ...]
    labels: np.ndarray
    clean_image: np.ndarray

    def __post_init__(self) -> None:
        if self.frame < 0:
            raise ValueError("frame must be non-negative")
        if self.labels.ndim != 2 or self.clean_image.shape != self.labels.shape:
            raise ValueError("labels and clean_image must be aligned 2-D arrays")
        if self.labels.dtype != np.uint16:
            raise ValueError("labels must use uint16 tracking identifiers")
        if not np.isfinite(self.clean_image).all():
            raise ValueError("clean_image must contain only finite values")
        object.__setattr__(self, "frame", int(self.frame))
        object.__setattr__(
            self,
            "positions",
            tuple((float(x), float(y)) for x, y in self.positions),
        )


@dataclass(frozen=True, slots=True)
class SimulatedFrame:
    """One ground-truth frame after applying the sensor noise model."""

    ground_truth: GroundTruthFrame
    image: np.ndarray

    def __post_init__(self) -> None:
        if self.image.shape != self.ground_truth.labels.shape:
            raise ValueError("simulated image must align with ground truth")
        if self.image.dtype != np.uint8:
            raise ValueError("simulated image must be uint8")

    @property
    def frame(self) -> int:
        return self.ground_truth.frame

    @property
    def labels(self) -> np.ndarray:
        return self.ground_truth.labels

    @property
    def positions(self) -> tuple[tuple[float, float], ...]:
        return self.ground_truth.positions


@dataclass(frozen=True, slots=True)
class SyntheticDataset:
    """Paths and loaders for one generated simulated sequence."""

    root: Path
    config: SyntheticDataConfig

    @property
    def raw_directory(self) -> Path:
        return self.root / self.config.sequence

    @property
    def tracking_directory(self) -> Path:
        return self.root / f"{self.config.sequence}_GT" / "TRA"

    @property
    def track_manifest_path(self) -> Path:
        return self.tracking_directory / "man_track.txt"

    def raw_frame_path(self, frame: int) -> Path:
        self._check_frame(frame)
        return self.raw_directory / f"t{frame:03d}.tif"

    def tracking_frame_path(self, frame: int) -> Path:
        self._check_frame(frame)
        return self.tracking_directory / f"man_track{frame:03d}.tif"

    def load_frame(self, frame: int) -> np.ndarray:
        return self._load(self.raw_frame_path(frame))

    def load_tracking_labels(self, frame: int) -> np.ndarray:
        return self._load(self.tracking_frame_path(frame))

    def _check_frame(self, frame: int) -> None:
        if not 0 <= frame < self.config.frame_count:
            raise ValueError(
                f"frame must lie in [0, {self.config.frame_count - 1}]"
            )

    @staticmethod
    def _load(path: Path) -> np.ndarray:
        with Image.open(path) as image:
            return np.asarray(image).copy()


class SyntheticNoiseModel:
    """Apply only blur and grain to an already-rendered truth frame."""

    def __init__(self, config: SyntheticDataConfig) -> None:
        self.config = config

    def apply(
        self,
        ground_truth: GroundTruthFrame,
        rng: np.random.Generator,
    ) -> SimulatedFrame:
        image = np.asarray(ground_truth.clean_image, dtype=np.float32).copy()
        if self.config.gaussian_bluriness > 0.0:
            image = ndi.gaussian_filter(
                image,
                sigma=self.config.gaussian_bluriness,
            )
        if self.config.grainyness > 0.0:
            image += rng.normal(0.0, self.config.grainyness, image.shape)
        image = np.clip(image, 0.0, 255.0).astype(np.uint8)
        return SimulatedFrame(ground_truth=ground_truth, image=image)


class GroundTruthRenderer:
    """Render clean labels and image intensities for a sequence frame."""

    def __init__(self, config: SyntheticDataConfig) -> None:
        self.config = config
        self._background = self._background_image()

    def render(
        self,
        frame: int,
        positions: np.ndarray,
    ) -> GroundTruthFrame:
        labels = np.zeros(self.config.image_shape, dtype=np.uint16)
        free_pixels = self.config.height * self.config.width
        normalized_positions: list[tuple[float, float]] = []

        for object_index, (x, y) in enumerate(positions):
            object_id = object_index + 1
            row, column = int(round(float(y))), int(round(float(x)))
            used_pixels = self._place_truth_marker(
                labels,
                row,
                column,
                object_id,
                free_pixels=free_pixels,
                remaining_objects=self.config.object_count - object_id,
            )
            free_pixels -= used_pixels
            normalized_positions.append((float(x), float(y)))

        clean = self._background.copy()
        clean[labels > 0] += _CELL_SIGNAL
        return GroundTruthFrame(
            frame=frame,
            positions=tuple(normalized_positions),
            labels=labels,
            clean_image=clean,
        )

    def _background_image(self) -> np.ndarray:
        height, width = self.config.image_shape
        yy, xx = np.mgrid[0:height, 0:width]
        return (
            _BACKGROUND_LEVEL
            + _BACKGROUND_SHADING
            * np.sin(2.0 * np.pi * xx / (2.3 * width))
            * np.cos(2.0 * np.pi * yy / (1.9 * height))
        ).astype(np.float32)

    @classmethod
    def _place_truth_marker(
        cls,
        labels: np.ndarray,
        row: int,
        column: int,
        object_id: int,
        *,
        free_pixels: int,
        remaining_objects: int,
    ) -> int:
        radius = _TRUTH_RADIUS_PX
        disk = cls._disk_offsets(radius)
        if free_pixels - len(disk) >= remaining_objects:
            center = cls._nearest_free_marker(labels, row, column, disk)
            if center is not None:
                center_row, center_column = center
                for row_offset, column_offset in disk:
                    labels[center_row + row_offset, center_column + column_offset] = (
                        object_id
                    )
                return len(disk)

        pixel = cls._nearest_free_marker(labels, row, column, ((0, 0),))
        if pixel is None:
            raise RuntimeError("no free pixel remains for a tracking marker")
        labels[pixel] = object_id
        return 1

    @staticmethod
    def _disk_offsets(radius: int) -> tuple[tuple[int, int], ...]:
        return tuple(
            (row, column)
            for row in range(-radius, radius + 1)
            for column in range(-radius, radius + 1)
            if row * row + column * column <= radius * radius
        )

    @staticmethod
    def _nearest_free_marker(
        labels: np.ndarray,
        target_row: int,
        target_column: int,
        offsets: tuple[tuple[int, int], ...],
    ) -> tuple[int, int] | None:
        height, width = labels.shape
        rows = tuple(row for row, _ in offsets)
        columns = tuple(column for _, column in offsets)
        minimum_row = -min(rows)
        maximum_row = height - max(rows) - 1
        minimum_column = -min(columns)
        maximum_column = width - max(columns) - 1
        if minimum_row > maximum_row or minimum_column > maximum_column:
            return None

        start = (
            min(max(target_row, minimum_row), maximum_row),
            min(max(target_column, minimum_column), maximum_column),
        )
        queue = [
            (
                (start[0] - target_row) ** 2
                + (start[1] - target_column) ** 2,
                start[0],
                start[1],
            )
        ]
        visited = {start}

        while queue:
            _, row, column = heappop(queue)
            if all(
                labels[row + row_offset, column + column_offset] == 0
                for row_offset, column_offset in offsets
            ):
                return row, column

            for next_row, next_column in (
                (row - 1, column),
                (row, column - 1),
                (row, column + 1),
                (row + 1, column),
            ):
                candidate = (next_row, next_column)
                if (
                    minimum_row <= next_row <= maximum_row
                    and minimum_column <= next_column <= maximum_column
                    and candidate not in visited
                ):
                    visited.add(candidate)
                    heappush(
                        queue,
                        (
                            (next_row - target_row) ** 2
                            + (next_column - target_column) ** 2,
                            next_row,
                            next_column,
                        ),
                    )
        return None


class SyntheticScene:
    """Generate deterministic object trajectories and clean truth frames."""

    def __init__(self, config: SyntheticDataConfig | None = None) -> None:
        self.config = config or SyntheticDataConfig()

    def iter_ground_truth(self) -> Iterator[GroundTruthFrame]:
        rng = np.random.default_rng(self.config.seed)
        trajectories = self._generate_trajectories(rng)
        renderer = GroundTruthRenderer(self.config)
        for frame, positions in enumerate(trajectories):
            yield renderer.render(frame, positions)

    def _generate_trajectories(self, rng: np.random.Generator) -> np.ndarray:
        config = self.config
        box_width = _REGION_FRACTION * config.width
        box_height = _REGION_FRACTION * config.height
        x_min = (config.width - box_width) / 2.0
        y_min = (config.height - box_height) / 2.0
        bounds = ((x_min, x_min + box_width), (y_min, y_min + box_height))

        positions = np.column_stack(
            (
                rng.uniform(*bounds[0], config.object_count),
                rng.uniform(*bounds[1], config.object_count),
            )
        )
        headings = rng.uniform(0.0, 2.0 * np.pi, config.object_count)
        trajectories = np.empty(
            (config.frame_count, config.object_count, 2),
            dtype=float,
        )

        for frame in range(config.frame_count):
            trajectories[frame] = positions
            headings += rng.normal(0.0, _TURN_SIGMA_RAD, config.object_count)
            step_lengths = np.clip(
                config.speed_px_per_frame
                + rng.normal(
                    0.0,
                    0.08 * config.speed_px_per_frame,
                    config.object_count,
                ),
                0.35 * config.speed_px_per_frame,
                1.8 * config.speed_px_per_frame,
            )
            positions += np.column_stack(
                (step_lengths * np.cos(headings), step_lengths * np.sin(headings))
            )
            self._reflect(positions, headings, bounds)
        return trajectories

    @staticmethod
    def _reflect(
        positions: np.ndarray,
        headings: np.ndarray,
        bounds: tuple[tuple[float, float], tuple[float, float]],
    ) -> None:
        for dimension, (lower, upper) in enumerate(bounds):
            outside = (positions[:, dimension] < lower) | (
                positions[:, dimension] > upper
            )
            while np.any(outside):
                below = positions[:, dimension] < lower
                above = positions[:, dimension] > upper
                positions[below, dimension] = (
                    2.0 * lower - positions[below, dimension]
                )
                positions[above, dimension] = (
                    2.0 * upper - positions[above, dimension]
                )
                reflected = below | above
                if dimension == 0:
                    headings[reflected] = np.pi - headings[reflected]
                else:
                    headings[reflected] = -headings[reflected]
                outside = (positions[:, dimension] < lower) | (
                    positions[:, dimension] > upper
                )


class SyntheticDataGenerator:
    """Stream or write simulated data without changing global random state."""

    def __init__(self, config: SyntheticDataConfig | None = None) -> None:
        self.config = config or SyntheticDataConfig()
        self.scene = SyntheticScene(self.config)
        self.noise_model = SyntheticNoiseModel(self.config)

    def generate(
        self,
        output_root: str | Path = DEFAULT_SYNTHETIC_DATA_ROOT,
    ) -> SyntheticDataset:
        dataset = SyntheticDataset(
            root=Path(output_root) / self.config.dataset_name,
            config=self.config,
        )
        if dataset.root.exists() and (
            not dataset.root.is_dir() or any(dataset.root.iterdir())
        ):
            raise FileExistsError(
                f"refusing to overwrite nonempty synthetic dataset: {dataset.root}"
            )
        dataset.raw_directory.mkdir(parents=True, exist_ok=True)
        dataset.tracking_directory.mkdir(parents=True, exist_ok=True)

        for simulated in self.iter_simulated_frames():
            Image.fromarray(simulated.image).save(
                dataset.raw_frame_path(simulated.frame)
            )
            Image.fromarray(simulated.labels).save(
                dataset.tracking_frame_path(simulated.frame)
            )

        final_frame = self.config.frame_count - 1
        manifest = "".join(
            f"{object_id} 0 {final_frame} 0\n"
            for object_id in range(1, self.config.object_count + 1)
        )
        dataset.track_manifest_path.write_text(manifest, encoding="utf-8")
        return dataset

    def iter_ground_truth(self) -> Iterator[GroundTruthFrame]:
        """Yield clean ground-truth frames before sensor noise is applied."""

        yield from self.scene.iter_ground_truth()

    def iter_simulated_frames(self) -> Iterator[SimulatedFrame]:
        """Yield full simulated-frame records with truth and noisy image."""

        noise_rng = np.random.default_rng(
            np.random.SeedSequence((self.config.seed, _NOISE_STREAM_ID))
        )
        for ground_truth in self.iter_ground_truth():
            yield self.noise_model.apply(ground_truth, noise_rng)

    def iter_frames(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(image, labels)`` pairs for existing tracking workflows."""

        for simulated in self.iter_simulated_frames():
            yield simulated.image, simulated.labels


__all__ = [
    "DEFAULT_SYNTHETIC_DATA_ROOT",
    "GroundTruthFrame",
    "GroundTruthRenderer",
    "QUANTUM_DEMO_DATA_CONFIG",
    "SimulatedFrame",
    "SyntheticDataConfig",
    "SyntheticDataGenerator",
    "SyntheticDataset",
    "SyntheticNoiseModel",
    "SyntheticScene",
]
