"""Modul 4a — Gravity-constrained RANSAC + Fassaden-Koordinatensystem."""

from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

GRAVITY_VEC = np.array([0.0, -1.0, 0.0])
UP_VEC     = np.array([0.0,  1.0, 0.0])
GRAVITY_TOL_SIN = float(np.sin(np.deg2rad(5.0)))   # ~0.0872


class RansacFailure(RuntimeError):
    """Die Fassadenebene konnte nicht zuverlässig geschätzt werden."""


def voxel_downsample(points: np.ndarray, voxel: float = 0.02) -> np.ndarray:
    """Einfaches Voxel-Grid-Downsampling (ein Punkt pro Voxel).

    Für jedes belegte Voxel wird der Centroid der darin enthaltenen Punkte
    ausgegeben — das ist numerisch stabiler als ein willkürlicher Vertreter.
    """
    if len(points) == 0:
        return points.copy()
    pts = points.astype(np.float64, copy=False)
    keys = np.floor(pts / voxel).astype(np.int64)
    # Kompakter Hash durch Packen der 3 int64-Koordinaten in ein Structured-Array
    view = np.ascontiguousarray(keys).view([("x", np.int64),
                                            ("y", np.int64),
                                            ("z", np.int64)])
    _, idx_inverse = np.unique(view, return_inverse=True)
    idx_inverse = idx_inverse.ravel()   # np.unique on structured array yields (N,1) on some NumPy versions
    n_voxels = idx_inverse.max() + 1
    sums = np.zeros((n_voxels, 3), dtype=np.float64)
    counts = np.zeros(n_voxels, dtype=np.int64)
    np.add.at(sums, idx_inverse, pts)
    np.add.at(counts, idx_inverse, 1)
    return (sums / counts[:, None]).astype(np.float32)


def _fit_plane_svd(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    """Least-Squares-Ebene durch ``points``. Gibt (normal, d, centroid)."""
    centroid = points.mean(axis=0)
    rel = points - centroid
    _, _, vt = np.linalg.svd(rel, full_matrices=False)
    normal = vt[-1]
    normal /= np.linalg.norm(normal)
    d = -float(normal @ centroid)
    return normal, d, centroid


def _plane_distances(points: np.ndarray, normal: np.ndarray, d: float
                     ) -> np.ndarray:
    return np.abs(points @ normal + d)


def gravity_constrained_ransac(points: np.ndarray,
                               distance_threshold: float = 0.03,
                               num_iterations: int = 1000,
                               gravity: np.ndarray = GRAVITY_VEC,
                               rng: np.random.Generator | None = None,
                               ) -> tuple[np.ndarray, float, np.ndarray, float]:
    """RANSAC-Ebene mit Constraint ``|n · gravity| < sin(5°)``.

    Returns
    -------
    normal : (3,) float, Einheitsvektor, noch nicht zur Kamera orientiert
    d      : Ebenen-Offset in ``n·x + d = 0``
    inliers: Indizes der Inlier in ``points``
    mse    : Mean-Squared-Error der Inlier-Abstände

    Raises
    ------
    RansacFailure wenn keine Iteration den Gravity-Constraint erfüllt oder
    wenn weniger als 3 Inliers übrig bleiben.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(points)
    if n < 3:
        raise RansacFailure(f"zu wenige Punkte: {n}")

    best_inliers: np.ndarray | None = None
    best_normal = None
    best_d = 0.0
    accepted_iterations = 0

    for _ in range(num_iterations):
        sample_idx = rng.choice(n, size=3, replace=False)
        sample = points[sample_idx]
        # Degenerate-Check: drei (fast) kollineare Punkte
        v1 = sample[1] - sample[0]
        v2 = sample[2] - sample[0]
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        if norm_cross < 1e-6:
            continue
        normal = cross / norm_cross
        d = -float(normal @ sample[0])

        # Gravity-Constraint
        if abs(float(normal @ gravity)) >= GRAVITY_TOL_SIN:
            continue
        accepted_iterations += 1

        dists = _plane_distances(points, normal, d)
        inliers = np.where(dists < distance_threshold)[0]
        if best_inliers is None or len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_normal = normal
            best_d = d

    if accepted_iterations == 0 or best_inliers is None:
        raise RansacFailure(
            f"keine valide Ebene gefunden — {accepted_iterations} constraint-OK Iterationen"
        )

    if len(best_inliers) < 3:
        raise RansacFailure(f"nur {len(best_inliers)} Inliers")

    # Refit über Inliers mit SVD für höhere Genauigkeit
    refined_normal, refined_d, _ = _fit_plane_svd(points[best_inliers])
    # Refined Normal muss den Constraint ebenfalls erfüllen — sonst originalen behalten
    if abs(float(refined_normal @ gravity)) < GRAVITY_TOL_SIN:
        final_normal, final_d = refined_normal, refined_d
    else:
        final_normal, final_d = best_normal, best_d

    dists = _plane_distances(points[best_inliers], final_normal, final_d)
    mse = float(np.mean(dists ** 2))
    return final_normal, final_d, best_inliers, mse


def orient_normal_to_camera(normal: np.ndarray,
                            plane_centroid: np.ndarray,
                            cam_mean: np.ndarray) -> np.ndarray:
    """Flippe die Normale so, dass sie zur Kamera zeigt."""
    to_cam = cam_mean - plane_centroid
    if np.dot(normal, to_cam) < 0:
        return -normal
    return normal


def compute_facade_frame(plane_normal: np.ndarray,
                         origin: np.ndarray,
                         up: np.ndarray = UP_VEC,
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Erzeuge rechtshändiges u/v-System in der Fassadenebene.

    v-Achse: ``up`` auf die Ebene projiziert und normiert (→ zeigt nach oben,
    liegt exakt in der Ebene, kompensiert die bis zu 5° Gravity-Abweichung).
    u-Achse: ``cross(v, n)`` → horizontal, in der Ebene, rechtshändig.
    """
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    v_raw = up - np.dot(up, plane_normal) * plane_normal
    v_norm = np.linalg.norm(v_raw)
    if v_norm < 1e-6:
        raise RansacFailure("up-Vektor (fast) parallel zur Ebenen-Normalen")
    v_axis = v_raw / v_norm
    u_axis = np.cross(v_axis, plane_normal)
    u_axis /= np.linalg.norm(u_axis)
    return u_axis, v_axis, origin


def verify_ransac_quality(mse: float, inlier_ratio: float,
                          mse_max: float = 0.05,
                          min_inlier_ratio: float = 0.60) -> None:
    """Wirft ``RansacFailure`` wenn eine Metrik unter dem Threshold liegt."""
    if mse > mse_max:
        raise RansacFailure(
            f"MSE {mse:.4f} m² > {mse_max} — Ebene passt schlecht (Kurve/Balkon?)")
    if inlier_ratio < min_inlier_ratio:
        raise RansacFailure(
            f"Inlier-Ratio {inlier_ratio:.2%} < {min_inlier_ratio:.0%} — "
            "mehrere Fassaden oder Eckgebäude? MVP unterstützt nur eine Fassade.")
