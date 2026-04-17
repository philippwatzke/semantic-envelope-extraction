"""Mathematische Helfer: Pose, Intrinsics, Rückprojektion, Rechteck-Abstände."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def quaternion_to_pose_matrix(tx: float, ty: float, tz: float,
                              qx: float, qy: float, qz: float, qw: float,
                              ) -> np.ndarray:
    """Baut eine 4x4 Kamera-zu-Welt-Matrix aus Stray-Scanner-Odometry-Zeile.

    Stray Scanner speichert Quaternionen als (qx, qy, qz, qw) — dieselbe
    Reihenfolge wie scipy ``Rotation.from_quat`` erwartet.
    """
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = (tx, ty, tz)
    return pose


def scale_intrinsics(K: np.ndarray,
                     src_wh: tuple[int, int],
                     dst_wh: tuple[int, int]) -> np.ndarray:
    """Skaliert eine 3x3-Intrinsics-Matrix von ``src_wh`` auf ``dst_wh``.

    RGB-Intrinsics aus ``odometry.csv`` werden so auf Depth-Auflösung
    umgerechnet, weil LiDAR-Depth (typ. 256x192) und RGB (1920x1080) sich
    auf dasselbe optische Zentrum beziehen, aber unterschiedliche
    Pixel-Raster haben.
    """
    src_w, src_h = src_wh
    dst_w, dst_h = dst_wh
    sx = dst_w / src_w
    sy = dst_h / src_h
    K_out = K.copy()
    K_out[0, 0] *= sx   # fx
    K_out[1, 1] *= sy   # fy
    K_out[0, 2] *= sx   # cx
    K_out[1, 2] *= sy   # cy
    return K_out
def backproject_pixels(mask: np.ndarray,
                       depth_m: np.ndarray,
                       confidence: np.ndarray,
                       K: np.ndarray,
                       pose_4x4: np.ndarray,
                       max_depth: float = 5.5,
                       min_confidence: int = 2) -> np.ndarray:
    """Rückprojektion aller validen Maskenpixel in Weltkoordinaten.

    Ein Pixel gilt als valide, wenn:
      * ``mask[v, u]`` True ist
      * ``confidence[v, u] >= min_confidence``
      * ``0 < depth_m[v, u] <= max_depth``

    Shape-Verträge:
      * ``mask``, ``depth_m``, ``confidence``: H x W (gleiche Shape)
      * ``K``: 3x3, passend zur Depth-Auflösung (siehe ``scale_intrinsics``)
      * ``pose_4x4``: Kamera → Welt

    Returns
    -------
    points : np.ndarray, Shape (N, 3), float32
        Weltkoordinaten der validen Pixel. Reihenfolge ist
        Zeilen-major über die Maske, aber aufrufende Seiten sollten keine
        Reihenfolge annehmen.
    """
    assert mask.shape == depth_m.shape == confidence.shape, \
        f"shape mismatch: {mask.shape} {depth_m.shape} {confidence.shape}"

    valid = mask & (confidence >= min_confidence) & \
            (depth_m > 0) & (depth_m <= max_depth)
    if not valid.any():
        return np.zeros((0, 3), dtype=np.float32)

    vs, us = np.where(valid)
    z = depth_m[vs, us].astype(np.float64)

    # Pixel → Kamera-Koord: P_cam = z * K^-1 @ [u, v, 1]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x_cam = (us - cx) / fx * z
    y_cam = (vs - cy) / fy * z
    z_cam = z
    pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z)], axis=1)   # N x 4

    # Kamera → Welt
    pts_world = (pose_4x4 @ pts_cam.T).T[:, :3]
    return pts_world.astype(np.float32)
def rect_edge_distance_2d(rect_a: tuple[float, float, float, float],
                          rect_b: tuple[float, float, float, float]) -> float:
    """Minimaler Kante-zu-Kante-Abstand zweier achsenparalleler Rechtecke.

    Rechteck-Format: ``(u_min, u_max, v_min, v_max)``.
    Überlappende Rechtecke → 0.0. Sonst der euklidische Abstand der
    nächsten Kantenpaare (entspricht dem kürzesten Weg in der u/v-Ebene).
    """
    a_umin, a_umax, a_vmin, a_vmax = rect_a
    b_umin, b_umax, b_vmin, b_vmax = rect_b

    gap_u = max(0.0, max(a_umin, b_umin) - min(a_umax, b_umax))
    gap_v = max(0.0, max(a_vmin, b_vmin) - min(a_vmax, b_vmax))
    return float(np.sqrt(gap_u * gap_u + gap_v * gap_v))
def project_points_to_plane_uv(points_3d: np.ndarray,
                               origin: np.ndarray,
                               u_axis: np.ndarray,
                               v_axis: np.ndarray) -> np.ndarray:
    """Projiziere 3D-Punkte auf eine Ebene und drücke sie in u/v aus.

    ``u_axis`` und ``v_axis`` müssen bereits orthonormal sein und exakt in
    der Ebene liegen (siehe ``plane_fitting.compute_facade_frame``).
    """
    rel = points_3d - origin
    u = rel @ u_axis
    v = rel @ v_axis
    return np.stack([u, v], axis=1).astype(np.float64)
