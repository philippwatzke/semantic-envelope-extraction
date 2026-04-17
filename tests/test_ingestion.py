from pathlib import Path

import numpy as np
import pytest

from semantic_envelope.ingestion import (
    parse_odometry,
    parse_camera_matrix,
    compute_keyframe_step,
    monitor_drift,
)


def _write_csv(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_parse_odometry_returns_pose_and_intrinsics_per_frame(tmp_path):
    csv = tmp_path / "odometry.csv"
    _write_csv(csv, [
        "timestamp,frame,x,y,z,qx,qy,qz,qw,fx,fy,cx,cy",
        "0.000,0,0.0,0.0,0.0,0,0,0,1,1500.0,1500.0,960.0,540.0",
        "0.033,1,0.1,0.0,0.0,0,0,0,1,1500.0,1500.0,960.0,540.0",
    ])
    by_frame = parse_odometry(csv)

    assert set(by_frame.keys()) == {0, 1}
    pose0, K0 = by_frame[0]
    assert pose0.shape == (4, 4)
    assert K0.shape == (3, 3)
    np.testing.assert_allclose(pose0[:3, 3], [0.0, 0.0, 0.0])
    np.testing.assert_allclose(by_frame[1][0][:3, 3], [0.1, 0.0, 0.0])


def test_parse_camera_matrix_returns_3x3(tmp_path):
    csv = tmp_path / "camera_matrix.csv"
    _write_csv(csv, [
        "1500.0,0.0,960.0",
        "0.0,1500.0,540.0",
        "0.0,0.0,1.0",
    ])
    K = parse_camera_matrix(csv)
    assert K.shape == (3, 3)
    assert K[0, 0] == 1500.0
    assert K[2, 2] == 1.0


def test_keyframe_step_targets_3fps_for_30fps_source():
    assert compute_keyframe_step(30.0) == 10


def test_keyframe_step_targets_3fps_for_60fps_source():
    assert compute_keyframe_step(60.0) == 20


def test_keyframe_step_minimum_one_for_low_fps():
    # Unter ~5 fps: jeden Frame nehmen (step=1)
    assert compute_keyframe_step(3.0) == 1
    assert compute_keyframe_step(1.0) == 1


def test_monitor_drift_no_warning_for_small_translations():
    # 1 cm pro Frame über 10 Frames — Median-Sprung weit unter 0.5 m
    poses = []
    for i in range(10):
        p = np.eye(4)
        p[:3, 3] = [0.01 * i, 0.0, 0.0]
        poses.append(p)
    warnings = monitor_drift(poses, window=5, threshold_m=0.5)
    assert warnings == []


def test_monitor_drift_warns_on_large_jump_window():
    poses = []
    for i in range(10):
        p = np.eye(4)
        # Konstante Sprünge von 1 m pro Frame
        p[:3, 3] = [1.0 * i, 0.0, 0.0]
        poses.append(p)
    warnings = monitor_drift(poses, window=5, threshold_m=0.5)
    assert len(warnings) > 0
