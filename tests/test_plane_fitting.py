import numpy as np
import pytest

from semantic_envelope.plane_fitting import (
    voxel_downsample,
    gravity_constrained_ransac,
    orient_normal_to_camera,
    compute_facade_frame,
    verify_ransac_quality,
    RansacFailure,
)


def test_voxel_downsample_reduces_size_and_covers_range():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-5, 5, size=(10_000, 3)).astype(np.float32)
    out = voxel_downsample(pts, voxel=0.1)
    assert len(out) < len(pts)
    # Alle axes-Bereiche sind in etwa erhalten
    assert out.min(axis=0).tolist()[0] <= pts.min(axis=0)[0] + 0.2
    assert out.max(axis=0).tolist()[0] >= pts.max(axis=0)[0] - 0.2


def test_ransac_recovers_vertical_facade_plane():
    rng = np.random.default_rng(1)
    # Ebene: x = 2 (Normal = +X = horizontal, orthogonal zu Y = gravity)
    n_pts = 5000
    y = rng.uniform(0, 3, n_pts).astype(np.float32)
    z = rng.uniform(-4, 0, n_pts).astype(np.float32)
    x = 2.0 + rng.normal(0, 0.01, n_pts).astype(np.float32)
    points = np.stack([x, y, z], axis=1)

    normal, d, inliers, mse = gravity_constrained_ransac(
        points, distance_threshold=0.03, num_iterations=500)

    # Normale ist ±X
    assert abs(normal[0]) > 0.98
    assert abs(normal[1]) < 0.1
    assert abs(normal[2]) < 0.1
    # Fast alle Punkte Inliers
    assert len(inliers) / len(points) > 0.95
    assert mse < 0.05


def test_ransac_rejects_tilted_plane():
    """Wenn die Ebene 45° zur Gravitation gekippt ist, muss RANSAC die
    constraint-erfüllenden (vertikalen) Kandidaten bevorzugen."""
    rng = np.random.default_rng(2)
    # Ebene: x + y = 2 → Normale (1,1,0)/sqrt(2) → verletzt Gravity-Constraint
    t = rng.uniform(-3, 3, size=(2000, 2))
    x = (2.0 - t[:, 0]) + rng.normal(0, 0.005, 2000)
    y = t[:, 0] + rng.normal(0, 0.005, 2000)
    z = t[:, 1]
    points = np.stack([x, y, z], axis=1).astype(np.float32)

    # RANSAC darf die gekippte Ebene nicht als valid akzeptieren.
    with pytest.raises(RansacFailure):
        normal, d, inliers, mse = gravity_constrained_ransac(
            points, distance_threshold=0.03, num_iterations=300)
        # Falls es doch eine Ebene findet: muss Inlier-Ratio-Gate reißen
        verify_ransac_quality(mse=mse, inlier_ratio=len(inliers) / len(points),
                              mse_max=0.05, min_inlier_ratio=0.6)


def test_orient_normal_flips_when_facing_away():
    # Ebene bei x=0, Normale war (-1,0,0), Kamera bei x=+5
    normal = np.array([-1.0, 0.0, 0.0])
    centroid = np.array([0.0, 0.0, 0.0])
    cam_mean = np.array([5.0, 0.0, 0.0])
    oriented = orient_normal_to_camera(normal, centroid, cam_mean)
    np.testing.assert_allclose(oriented, [1.0, 0.0, 0.0])


def test_compute_facade_frame_produces_orthonormal_basis():
    # Ebene mit Normale (1, 0.05, 0) — leichte Yaw-Abweichung
    normal = np.array([1.0, 0.05, 0.0])
    normal /= np.linalg.norm(normal)
    origin = np.array([0.0, 1.5, -2.0])
    u, v, o = compute_facade_frame(normal, origin)
    # Orthonormal-Check
    np.testing.assert_allclose(np.dot(u, v), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.dot(u, normal), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.dot(v, normal), 0.0, atol=1e-9)
    np.testing.assert_allclose(np.linalg.norm(u), 1.0)
    np.testing.assert_allclose(np.linalg.norm(v), 1.0)
    # v zeigt "überwiegend nach oben" (ARKit Y-up)
    assert v[1] > 0.99
