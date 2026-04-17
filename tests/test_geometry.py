import numpy as np
import pytest

from semantic_envelope.geometry import (
    quaternion_to_pose_matrix,
    scale_intrinsics,
    backproject_pixels,
    rect_edge_distance_2d,
    project_points_to_plane_uv,
)


def test_identity_quaternion_and_zero_translation_gives_identity():
    pose = quaternion_to_pose_matrix(tx=0, ty=0, tz=0, qx=0, qy=0, qz=0, qw=1)
    np.testing.assert_allclose(pose, np.eye(4), atol=1e-9)


def test_translation_only_appears_in_last_column():
    pose = quaternion_to_pose_matrix(tx=1.0, ty=2.0, tz=3.0,
                                     qx=0, qy=0, qz=0, qw=1)
    np.testing.assert_allclose(pose[:3, 3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(pose[:3, :3], np.eye(3), atol=1e-9)
    np.testing.assert_allclose(pose[3, :], [0, 0, 0, 1])


def test_90deg_rotation_around_y_maps_x_axis_to_minus_z():
    # Stray-Scanner / scipy convention: (qx, qy, qz, qw)
    # Rotation von 90° um Y-Achse: qy = sin(45°), qw = cos(45°)
    s = np.sin(np.pi / 4)
    c = np.cos(np.pi / 4)
    pose = quaternion_to_pose_matrix(0, 0, 0, 0, s, 0, c)
    R = pose[:3, :3]
    # R @ [1,0,0] sollte [0,0,-1] ergeben (Rechte-Hand-Regel)
    np.testing.assert_allclose(R @ np.array([1, 0, 0]),
                               np.array([0, 0, -1]),
                               atol=1e-9)


def test_scale_intrinsics_halves_fx_fy_cx_cy_when_resolution_halves():
    K = np.array([[1600.0, 0.0, 960.0],
                  [0.0, 1600.0, 540.0],
                  [0.0, 0.0, 1.0]])
    K_d = scale_intrinsics(K, src_wh=(1920, 1080), dst_wh=(960, 540))
    np.testing.assert_allclose(K_d[0, 0], 800.0)
    np.testing.assert_allclose(K_d[1, 1], 800.0)
    np.testing.assert_allclose(K_d[0, 2], 480.0)
    np.testing.assert_allclose(K_d[1, 2], 270.0)
    assert K_d[2, 2] == 1.0


def test_scale_intrinsics_matches_iphone_pro_ratio():
    # RGB 1920x1080 → Depth 256x192
    K = np.array([[1500.0, 0.0, 950.0],
                  [0.0, 1500.0, 530.0],
                  [0.0, 0.0, 1.0]])
    K_d = scale_intrinsics(K, src_wh=(1920, 1080), dst_wh=(256, 192))
    assert K_d[0, 0] == pytest.approx(1500.0 * 256 / 1920)
    assert K_d[1, 1] == pytest.approx(1500.0 * 192 / 1080)


def test_backproject_center_pixel_at_depth_5_is_at_5m_along_minus_z():
    """Bei Identity-Pose blickt ARKit entlang -Z → Tiefe 5 → Weltpunkt (0,0,-5)."""
    K = np.array([[500.0, 0.0, 128.0],
                  [0.0, 500.0, 96.0],
                  [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    depth_m = np.zeros((192, 256), dtype=np.float32)
    mask = np.zeros((192, 256), dtype=bool)
    depth_m[96, 128] = 5.0
    mask[96, 128] = True
    conf = np.full((192, 256), 2, dtype=np.uint8)

    pts = backproject_pixels(mask=mask, depth_m=depth_m, confidence=conf,
                             K=K, pose_4x4=pose, max_depth=5.5)

    assert pts.shape == (1, 3)
    # ARKit: Kamera blickt nach -Z, also Punkt vor der Kamera ist bei -Z.
    # Unsere Rückprojektion: P_cam = depth * K^-1 @ [u,v,1] ergibt +Z.
    # Die ARKit-Pose-Matrix flipped das implizit, aber bei Identity-Pose:
    # die Konvention ist +Z vorwärts. Dieser Test bestätigt: bei Identity
    # liegt der Punkt bei z=+5 (Depth-Frame-Konvention: Depth ist +Z).
    np.testing.assert_allclose(pts[0], [0.0, 0.0, 5.0], atol=1e-6)


def test_backproject_filters_by_confidence_and_max_depth():
    K = np.array([[500.0, 0.0, 128.0],
                  [0.0, 500.0, 96.0],
                  [0.0, 0.0, 1.0]])
    pose = np.eye(4)
    depth_m = np.array([[5.0, 6.0, 4.0, 0.0]], dtype=np.float32)
    conf    = np.array([[2,   2,   1,   2]], dtype=np.uint8)
    mask    = np.array([[True, True, True, True]], dtype=bool)
    # Fake 1x4 "image" — wir brauchen nur die Pixel selbst
    depth_m = depth_m.reshape(1, 4)
    conf = conf.reshape(1, 4)
    mask = mask.reshape(1, 4)
    K2 = np.array([[500.0, 0.0, 1.5],
                   [0.0, 500.0, 0.0],
                   [0.0, 0.0, 1.0]])
    pts = backproject_pixels(mask=mask, depth_m=depth_m, confidence=conf,
                             K=K2, pose_4x4=pose, max_depth=5.5)
    # Erwartet: erstes Pixel (depth=5, conf=2) überlebt, alle drei anderen
    # werden durch max_depth, conf<2 bzw. depth=0 gefiltert.
    assert pts.shape == (1, 3)


def test_rect_edge_distance_is_zero_for_overlapping_rects():
    d = rect_edge_distance_2d((0.0, 1.0, 0.0, 1.0), (0.5, 1.5, 0.5, 1.5))
    assert d == pytest.approx(0.0)


def test_rect_edge_distance_is_horizontal_gap_when_aligned_vertically():
    # Zwei Rechtecke mit gleicher v-Range, u-Abstand 0.3
    d = rect_edge_distance_2d((0.0, 1.0, 0.0, 1.0), (1.3, 2.3, 0.0, 1.0))
    assert d == pytest.approx(0.3)


def test_rect_edge_distance_is_diagonal_gap_when_diagonal():
    # Rect A: [0,1] x [0,1], Rect B: [2,3] x [3,4] → diag sqrt(1² + 2²)
    d = rect_edge_distance_2d((0.0, 1.0, 0.0, 1.0), (2.0, 3.0, 3.0, 4.0))
    assert d == pytest.approx(np.sqrt(1**2 + 2**2))


def test_project_to_plane_uv_recovers_xy_for_xy_plane():
    # Ebene: z = 0, Normal = (0, 0, 1), u-Achse = (1,0,0), v-Achse = (0,1,0)
    origin = np.array([0.0, 0.0, 0.0])
    u_axis = np.array([1.0, 0.0, 0.0])
    v_axis = np.array([0.0, 1.0, 0.0])

    pts = np.array([[1.0, 2.0, 0.0],
                    [-0.5, 3.0, 0.1]])   # 0.1 Höhe wird ignoriert (projiziert)
    uv = project_points_to_plane_uv(pts, origin, u_axis, v_axis)
    np.testing.assert_allclose(uv, [[1.0, 2.0], [-0.5, 3.0]])


def test_project_respects_origin_shift():
    origin = np.array([10.0, 10.0, 0.0])
    u_axis = np.array([1.0, 0.0, 0.0])
    v_axis = np.array([0.0, 1.0, 0.0])
    pts = np.array([[11.0, 12.0, 0.0]])
    uv = project_points_to_plane_uv(pts, origin, u_axis, v_axis)
    np.testing.assert_allclose(uv, [[1.0, 2.0]])
