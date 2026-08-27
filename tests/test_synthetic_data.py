"""Check ground-truth-first simulation and optional CTC file materialization."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from neutral_atom_mht import (
    DEFAULT_SYNTHETIC_DATA_ROOT,
    GroundTruthFrame,
    QUANTUM_DEMO_DATA_CONFIG,
    SimulatedFrame,
    SyntheticDataConfig,
    SyntheticDataGenerator,
    SyntheticScene,
)


def _small_config(**changes: object) -> SyntheticDataConfig:
    values = {
        "gaussian_bluriness": 1.0,
        "grainyness": 2.0,
        "frame_count": 3,
        "object_count": 4,
        "seed": 17,
        "dataset_name": "TINY-MHT",
        "sequence": "01",
        "image_shape": (64, 80),
        "speed_px_per_frame": 2.0,
    }
    values.update(changes)
    return SyntheticDataConfig(**values)


def test_scene_yields_clean_ground_truth_before_noise() -> None:
    config = _small_config()

    truth = tuple(SyntheticScene(config).iter_ground_truth())

    assert len(truth) == config.frame_count
    assert all(isinstance(frame, GroundTruthFrame) for frame in truth)
    assert tuple(frame.frame for frame in truth) == (0, 1, 2)
    assert all(frame.labels.dtype == np.uint16 for frame in truth)
    assert all(frame.clean_image.dtype == np.float32 for frame in truth)
    assert all(frame.labels.shape == config.image_shape for frame in truth)
    assert all(len(frame.positions) == config.object_count for frame in truth)
    expected_ids = set(range(1, config.object_count + 1))
    assert all(set(np.unique(frame.labels)) - {0} == expected_ids for frame in truth)


def test_simulated_frames_wrap_truth_and_apply_only_blur_and_grain() -> None:
    clean = _small_config(gaussian_bluriness=0.0, grainyness=0.0)
    blurred = replace(clean, gaussian_bluriness=1.4)
    grainy = replace(clean, grainyness=8.0)

    clean_frame = next(SyntheticDataGenerator(clean).iter_simulated_frames())
    blurred_frame = next(SyntheticDataGenerator(blurred).iter_simulated_frames())
    grainy_frame = next(SyntheticDataGenerator(grainy).iter_simulated_frames())

    assert isinstance(clean_frame, SimulatedFrame)
    assert clean_frame.frame == 0
    assert clean_frame.image.dtype == np.uint8
    assert np.array_equal(clean_frame.labels, clean_frame.ground_truth.labels)
    assert np.array_equal(
        clean_frame.image,
        np.clip(clean_frame.ground_truth.clean_image, 0, 255).astype(np.uint8),
    )

    assert np.array_equal(clean_frame.labels, blurred_frame.labels)
    assert np.array_equal(
        clean_frame.ground_truth.clean_image,
        blurred_frame.ground_truth.clean_image,
    )
    assert not np.array_equal(clean_frame.image, blurred_frame.image)

    assert np.array_equal(clean_frame.labels, grainy_frame.labels)
    assert np.array_equal(
        clean_frame.ground_truth.clean_image,
        grainy_frame.ground_truth.clean_image,
    )
    assert not np.array_equal(clean_frame.image, grainy_frame.image)


def test_same_seed_produces_identical_truth_and_images() -> None:
    config = _small_config(frame_count=2, object_count=3)
    first = tuple(SyntheticDataGenerator(config).iter_simulated_frames())
    second = tuple(SyntheticDataGenerator(config).iter_simulated_frames())

    for left, right in zip(first, second, strict=True):
        assert left.positions == right.positions
        assert np.array_equal(left.labels, right.labels)
        assert np.array_equal(left.ground_truth.clean_image, right.ground_truth.clean_image)
        assert np.array_equal(left.image, right.image)


def test_streamed_pairs_match_the_simulated_frame_records() -> None:
    config = _small_config(frame_count=2)
    records = tuple(SyntheticDataGenerator(config).iter_simulated_frames())
    pairs = tuple(SyntheticDataGenerator(config).iter_frames())

    assert len(records) == len(pairs) == config.frame_count
    for record, (image, labels) in zip(records, pairs, strict=True):
        assert np.array_equal(record.image, image)
        assert np.array_equal(record.labels, labels)


def test_generator_writes_ctc_sequence_and_loads_tiffs(tmp_path: Path) -> None:
    config = _small_config()
    dataset = SyntheticDataGenerator(config).generate(tmp_path)

    assert dataset.root == tmp_path / config.dataset_name
    assert dataset.raw_directory == dataset.root / "01"
    assert dataset.tracking_directory == dataset.root / "01_GT" / "TRA"
    assert {path.name for path in dataset.raw_directory.iterdir()} == {
        "t000.tif",
        "t001.tif",
        "t002.tif",
    }
    assert {path.name for path in dataset.tracking_directory.iterdir()} == {
        "man_track000.tif",
        "man_track001.tif",
        "man_track002.tif",
        "man_track.txt",
    }

    image = dataset.load_frame(1)
    labels = dataset.load_tracking_labels(1)
    assert image.shape == config.image_shape
    assert labels.shape == config.image_shape
    assert image.dtype == np.uint8
    assert labels.dtype == np.uint16
    assert dataset.track_manifest_path.read_text(encoding="utf-8") == (
        "1 0 2 0\n2 0 2 0\n3 0 2 0\n4 0 2 0\n"
    )


def test_written_frames_match_streamed_frames(tmp_path: Path) -> None:
    config = _small_config(frame_count=2)
    streamed = tuple(SyntheticDataGenerator(config).iter_frames())
    dataset = SyntheticDataGenerator(config).generate(tmp_path)

    for frame, (image, labels) in enumerate(streamed):
        assert np.array_equal(image, dataset.load_frame(frame))
        assert np.array_equal(labels, dataset.load_tracking_labels(frame))


def test_existing_dataset_is_not_partially_overwritten(tmp_path: Path) -> None:
    three_frames = _small_config(frame_count=3)
    original = SyntheticDataGenerator(three_frames).generate(tmp_path)
    original_bytes = {
        path.name: path.read_bytes() for path in original.raw_directory.iterdir()
    }

    one_frame = _small_config(frame_count=1)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        SyntheticDataGenerator(one_frame).generate(tmp_path)

    assert {
        path.name: path.read_bytes() for path in original.raw_directory.iterdir()
    } == original_bytes


def test_colliding_trajectories_keep_every_ground_truth_id() -> None:
    config = _small_config(
        gaussian_bluriness=0.0,
        grainyness=0.0,
        frame_count=4,
        object_count=20,
        image_shape=(6, 6),
    )
    expected_ids = set(range(1, config.object_count + 1))

    for labels in (frame.labels for frame in SyntheticDataGenerator(config).iter_simulated_frames()):
        assert set(np.unique(labels)) - {0} == expected_ids


def test_quantum_demo_preset_is_small_versioned_and_simulated_only() -> None:
    assert QUANTUM_DEMO_DATA_CONFIG == SyntheticDataConfig(
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


def test_configuration_and_frame_bounds_are_clear(tmp_path: Path) -> None:
    assert DEFAULT_SYNTHETIC_DATA_ROOT == Path("data") / "synthetic"
    with pytest.raises(ValueError, match="gaussian_bluriness"):
        _small_config(gaussian_bluriness=-0.1)
    with pytest.raises(ValueError, match="grainyness"):
        _small_config(grainyness=-1.0)
    with pytest.raises(ValueError, match="speed_px_per_frame"):
        _small_config(speed_px_per_frame=0.0)
    with pytest.raises(ValueError, match="directory name"):
        _small_config(dataset_name="../outside")
    with pytest.raises(ValueError, match="uint16"):
        _small_config(object_count=65_536, image_shape=(256, 256))
    with pytest.raises(ValueError, match="available image pixels"):
        _small_config(object_count=17, image_shape=(4, 4))
    with pytest.raises(TypeError):
        _small_config(noise=0.5)

    dataset = SyntheticDataGenerator(_small_config()).generate(tmp_path)
    with pytest.raises(ValueError, match="frame"):
        dataset.load_frame(-1)
    with pytest.raises(ValueError, match="frame"):
        dataset.load_tracking_labels(dataset.config.frame_count)
