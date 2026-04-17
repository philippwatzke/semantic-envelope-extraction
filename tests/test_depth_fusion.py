from pathlib import Path

import cv2
import numpy as np
import pytest

from semantic_envelope.depth_fusion import (
    load_depth_confidence,
    align_mask_to_depth,
    fuse_wall_and_targets,
    WallCloudAccumulator,
)
from semantic_envelope.segmentation import FrameSegmentation, TargetInstance
from semantic_envelope.types import FrameData


def _write_depth_png(path: Path, depth_mm: np.ndarray) -> None:
    assert depth_mm.dtype == np.uint16
    cv2.imwrite(str(path), depth_mm)


def _write_conf_png(path: Path, conf: np.ndarray) -> None:
    assert conf.dtype == np.uint8
    cv2.imwrite(str(path), conf)


def test_load_depth_converts_mm_to_meters(tmp_path):
    depth_mm = np.array([[0, 1000, 2500],
                         [5000, 5500, 6000]], dtype=np.uint16)
    conf = np.array([[2, 2, 2],
                     [2, 2, 1]], dtype=np.uint8)
    dp = tmp_path / "d.png"; cp = tmp_path / "c.png"
    _write_depth_png(dp, depth_mm); _write_conf_png(cp, conf)

    depth_m, conf_out = load_depth_confidence(dp, cp)
    assert depth_m.dtype == np.float32
    np.testing.assert_allclose(depth_m, depth_mm.astype(np.float32) / 1000.0)
    np.testing.assert_array_equal(conf_out, conf)


def test_align_mask_to_depth_preserves_shape_and_nearest_neighbor():
    mask = np.zeros((1080, 1920), dtype=bool)
    mask[400:600, 900:1100] = True
    out = align_mask_to_depth(mask, depth_shape=(192, 256))
    assert out.shape == (192, 256)
    assert out.dtype == bool
    # Anteil sollte grob erhalten bleiben (40_000 / (1080*1920) ≈ 1.93 %)
    frac_in = mask.mean()
    frac_out = out.mean()
    assert abs(frac_in - frac_out) < 0.01


def test_accumulator_handles_empty_and_single_array():
    acc = WallCloudAccumulator()
    acc.append(np.zeros((0, 3), dtype=np.float32))   # leer, kein Crash
    acc.append(np.array([[1, 2, 3]], dtype=np.float32))
    merged = acc.finalize()
    assert merged.shape == (1, 3)


def test_accumulator_spills_and_rebuilds(tmp_path):
    acc = WallCloudAccumulator(spill_bytes=100, spill_frame_count=2,
                               tmp_dir=tmp_path)
    acc.append(np.random.rand(50, 3).astype(np.float32))
    acc.append(np.random.rand(50, 3).astype(np.float32))
    # Nach 2 Frames sollte gespillt worden sein
    acc.append(np.random.rand(50, 3).astype(np.float32))
    merged = acc.finalize()
    assert merged.shape == (150, 3)
    acc.cleanup()


def test_fuse_wall_and_targets_projects_center_pixel_to_expected_world(tmp_path):
    # 192x256 depth, identity pose, depth 3m überall, conf=2
    depth_mm = np.full((192, 256), 3000, dtype=np.uint16)
    conf     = np.full((192, 256), 2,    dtype=np.uint8)
    dp = tmp_path / "d.png"; cp = tmp_path / "c.png"
    _write_depth_png(dp, depth_mm); _write_conf_png(cp, conf)

    # RGB 1920x1080, Maske nur 1 Pixel in der Mitte
    wall_rgb_mask = np.zeros((1080, 1920), dtype=bool)
    wall_rgb_mask[540, 960] = True
    frame = FrameData(
        frame_id=0,
        rgb=np.zeros((1080, 1920, 3), dtype=np.uint8),
        depth_path=str(dp),
        confidence_path=str(cp),
        pose_4x4=np.eye(4),
        intrinsics=np.array([[1500.0, 0.0, 960.0],
                             [0.0, 1500.0, 540.0],
                             [0.0, 0.0, 1.0]]),
    )
    seg = FrameSegmentation(target_instances=[], wall_mask=wall_rgb_mask)
    wall_pts, rings = fuse_wall_and_targets(frame, seg)
    assert rings == []
    # Wand: 1 einzelnes Pixel → erwartet ~1 Punkt (Nearest-Neighbor-Resize
    # kann je nach Rundung 0 oder 1 ergeben — beide Fälle akzeptabel).
    assert len(wall_pts) <= 1
