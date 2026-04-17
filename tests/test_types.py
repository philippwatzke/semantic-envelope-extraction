import numpy as np
import pytest

from semantic_envelope.types import FrameData, RingData, BoxObs, Window


def test_frame_data_constructs_with_all_fields():
    rgb = np.zeros((10, 10, 3), dtype=np.uint8)
    pose = np.eye(4, dtype=np.float64)
    K = np.eye(3, dtype=np.float64)

    frame = FrameData(
        frame_id=42,
        rgb=rgb,
        depth_path="foo/depth/000042.png",
        confidence_path="foo/confidence/000042.png",
        pose_4x4=pose,
        intrinsics=K,
    )

    assert frame.frame_id == 42
    assert frame.rgb.shape == (10, 10, 3)
    assert frame.pose_4x4.shape == (4, 4)
    assert frame.intrinsics.shape == (3, 3)


def test_ring_data_preserves_class_label():
    pts = np.zeros((100, 3), dtype=np.float32)
    ring = RingData(frame_id=1, instance_id=7, klasse="window", points=pts)
    assert ring.klasse == "window"
    assert ring.points.shape == (100, 3)


def test_box_obs_bottom_left_convention():
    obs = BoxObs(
        frame_id=1,
        instance_id=7,
        klasse="door",
        u_min=0.1,
        u_max=1.2,
        v_min=0.0,
        v_max=2.0,
    )
    assert obs.u_max > obs.u_min
    assert obs.v_max > obs.v_min


def test_window_has_observations_count():
    w = Window(
        id=0,
        klasse="window",
        u_min=0.0, u_max=1.0, v_min=0.0, v_max=1.5,
        n_observations=12,
    )
    assert w.n_observations == 12
