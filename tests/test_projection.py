import numpy as np
import pytest

from semantic_envelope.projection import (
    aabb_from_uv,
    aabb_aspect_ratio_ok,
    project_rings_to_box_observations,
)
from semantic_envelope.types import RingData


def test_aabb_from_uv_returns_min_max_tuple():
    uv = np.array([[0.1, 0.2], [1.5, 1.8], [0.8, -0.1]])
    aabb = aabb_from_uv(uv)
    assert aabb == pytest.approx((0.1, 1.5, -0.1, 1.8))


def test_aabb_aspect_ratio_accepts_square_window():
    assert aabb_aspect_ratio_ok((0.0, 1.0, 0.0, 1.0)) is True


def test_aabb_aspect_ratio_rejects_sliver():
    # 5 m breit, 10 cm hoch
    assert aabb_aspect_ratio_ok((0.0, 5.0, 0.0, 0.1), lo=0.1, hi=10.0) is False


def test_project_rings_produces_one_box_obs_per_ring():
    # Zwei Rings auf Ebene z = 0
    origin = np.array([0.0, 0.0, 0.0])
    u_axis = np.array([1.0, 0.0, 0.0])
    v_axis = np.array([0.0, 1.0, 0.0])
    r1 = RingData(frame_id=1, instance_id=0, klasse="window",
                  points=np.array([[0.0, 0.0, 0.0],
                                   [1.0, 1.5, 0.0]], dtype=np.float32))
    r2 = RingData(frame_id=1, instance_id=1, klasse="door",
                  points=np.array([[2.0, 0.0, 0.0],
                                   [3.0, 2.0, 0.0]], dtype=np.float32))
    obs = project_rings_to_box_observations([r1, r2], origin, u_axis, v_axis)
    assert len(obs) == 2
    assert obs[0].klasse == "window"
    assert obs[0].u_min == 0.0 and obs[0].u_max == 1.0
    assert obs[1].klasse == "door"
