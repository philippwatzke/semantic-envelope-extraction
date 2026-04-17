"""Modul 4b — Projiziere Ring-Punkte auf die Fassadenebene und bilde AABBs."""

from __future__ import annotations

import logging

import numpy as np

from .geometry import project_points_to_plane_uv
from .types import BoxObs, RingData

log = logging.getLogger(__name__)


def aabb_from_uv(uv: np.ndarray) -> tuple[float, float, float, float]:
    """``(u_min, u_max, v_min, v_max)`` aus Nx2 uv-Koordinaten."""
    u_min, v_min = uv.min(axis=0)
    u_max, v_max = uv.max(axis=0)
    return (float(u_min), float(u_max), float(v_min), float(v_max))


def aabb_aspect_ratio_ok(aabb: tuple[float, float, float, float],
                         lo: float = 0.1, hi: float = 10.0) -> bool:
    u_min, u_max, v_min, v_max = aabb
    w = max(1e-6, u_max - u_min)
    h = max(1e-6, v_max - v_min)
    r = w / h
    return lo <= r <= hi


def project_rings_to_box_observations(rings: list[RingData],
                                      origin: np.ndarray,
                                      u_axis: np.ndarray,
                                      v_axis: np.ndarray) -> list[BoxObs]:
    """Projiziere jeden Ring auf die Fassadenebene und erzeuge ``BoxObs``."""
    out: list[BoxObs] = []
    for ring in rings:
        uv = project_points_to_plane_uv(ring.points, origin, u_axis, v_axis)
        aabb = aabb_from_uv(uv)
        if not aabb_aspect_ratio_ok(aabb):
            log.warning("frame %d inst %d: aabb-aspect außerhalb bounds %s → verworfen",
                        ring.frame_id, ring.instance_id, aabb)
            continue
        u_min, u_max, v_min, v_max = aabb
        out.append(BoxObs(frame_id=ring.frame_id,
                          instance_id=ring.instance_id,
                          klasse=ring.klasse,
                          u_min=u_min, u_max=u_max,
                          v_min=v_min, v_max=v_max))
    return out
