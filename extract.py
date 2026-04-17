"""Semantic Envelope Extraction — CLI."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from semantic_envelope.depth_fusion import (
    DEFAULT_MAX_DEPTH_M,
    DEFAULT_MIN_CONFIDENCE,
    WallCloudAccumulator,
    fuse_wall_and_targets,
)
from semantic_envelope.geometry import project_points_to_plane_uv
from semantic_envelope.ingestion import ingest_zip
from semantic_envelope.merging import merge_observations
from semantic_envelope.output import (
    render_facade_png,
    render_pointcloud_png,
    setup_logging,
    write_csv,
)
from semantic_envelope.plane_fitting import (
    RansacFailure,
    compute_facade_frame,
    gravity_constrained_ransac,
    orient_normal_to_camera,
    verify_ransac_quality,
    voxel_downsample,
)
from semantic_envelope.projection import project_rings_to_box_observations
from semantic_envelope.segmentation import ThreePromptSegmenter

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_EMPTY = 2

log = logging.getLogger("extract")


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Stray-Scanner -> CSV + PNGs")
    p.add_argument("--input", required=True, type=Path,
                   help="Pfad zum Stray-Scanner-ZIP")
    p.add_argument("--output", required=True, type=Path,
                   help="Ausgabeverzeichnis")
    p.add_argument("--blur-threshold", type=float, default=100.0)
    p.add_argument("--merge-distance", type=float, default=0.15)
    p.add_argument("--min-observations", type=int, default=3)
    p.add_argument("--sam-model", choices=["base", "large"], default="base")
    p.add_argument("--max-depth", type=float, default=DEFAULT_MAX_DEPTH_M)
    p.add_argument("--gdino-box-threshold", type=float, default=0.35)
    p.add_argument("--gdino-text-threshold", type=float, default=0.25)
    p.add_argument("--ransac-distance-threshold", type=float, default=0.03)
    p.add_argument("--ransac-mse-max", type=float, default=0.05)
    p.add_argument("--ransac-inlier-ratio-min", type=float, default=0.6)
    p.add_argument("--ransac-num-iterations", type=int, default=1000)
    p.add_argument("--voxel-size", type=float, default=0.02)
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--device", default="cuda")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    setup_logging(args.output, args.log_level)

    log.info("=" * 60)
    log.info("Semantic Envelope Extraction")
    log.info("Input : %s", args.input)
    log.info("Output: %s", args.output)
    log.info("=" * 60)

    # ---- Modul 1 ----
    frames, scan_root = ingest_zip(
        args.input,
        work_dir=args.output / "_unpacked",
        blur_threshold=args.blur_threshold,
    )
    if not frames:
        log.error("keine Keyframes akzeptiert — Scan vermutlich fehlerhaft")
        return EXIT_ERROR
    log.info("Phase 1: verarbeite %d keyframes", len(frames))

    # ---- Modul 2 + 3 pro Frame ----
    wall_acc = WallCloudAccumulator(tmp_dir=args.output / "_wallcloud_tmp")
    all_rings = []
    all_poses = []

    segmenter = ThreePromptSegmenter(
        sam_model_size=args.sam_model,
        box_threshold=args.gdino_box_threshold,
        text_threshold=args.gdino_text_threshold,
        device=args.device,
    )
    segmenter.load()
    try:
        for i, frame in enumerate(frames):
            log.info("[%d/%d] frame_id=%d", i + 1, len(frames), frame.frame_id)
            seg = segmenter.segment_frame(frame.rgb, frame.frame_id)
            wall_pts, rings = fuse_wall_and_targets(
                frame, seg,
                max_depth=args.max_depth,
                min_confidence=DEFAULT_MIN_CONFIDENCE,
            )
            wall_acc.append(wall_pts)
            all_rings.extend(rings)
            all_poses.append(frame.pose_4x4)
    finally:
        segmenter.unload()

    wall_cloud = wall_acc.finalize()
    wall_acc.cleanup()
    log.info("Phase 1 fertig: wall_cloud=%d Punkte, rings=%d Instanzen",
             len(wall_cloud), len(all_rings))
    if len(wall_cloud) < 1000:
        log.error("wall_cloud zu klein (%d) — Scan offenbar leer", len(wall_cloud))
        return EXIT_ERROR

    # ---- Modul 4a: Voxel + RANSAC ----
    wall_ds = voxel_downsample(wall_cloud, voxel=args.voxel_size)
    log.info("voxel-downsample: %d → %d Punkte", len(wall_cloud), len(wall_ds))
    try:
        normal, d, inliers, mse = gravity_constrained_ransac(
            wall_ds,
            distance_threshold=args.ransac_distance_threshold,
            num_iterations=args.ransac_num_iterations,
        )
        inlier_ratio = len(inliers) / len(wall_ds)
        log.info("RANSAC: |n|=%s, inliers=%.1f%%, mse=%.4f m²",
                 np.round(normal, 3), 100 * inlier_ratio, mse)
        verify_ransac_quality(
            mse=mse, inlier_ratio=inlier_ratio,
            mse_max=args.ransac_mse_max,
            min_inlier_ratio=args.ransac_inlier_ratio_min)
    except RansacFailure as exc:
        log.error("RANSAC-Failure: %s", exc)
        return EXIT_ERROR

    inlier_pts = wall_ds[inliers]
    plane_centroid = inlier_pts.mean(axis=0)
    cam_mean = np.mean([p[:3, 3] for p in all_poses], axis=0)
    normal = orient_normal_to_camera(normal, plane_centroid, cam_mean)
    u_axis, v_axis, origin = compute_facade_frame(normal, plane_centroid)
    log.info("Fassaden-Frame: origin=%s, u=%s, v=%s",
             np.round(origin, 3), np.round(u_axis, 3), np.round(v_axis, 3))

    # Plausibilitäts-Check Fassaden-Breite
    wall_uv = project_points_to_plane_uv(wall_cloud, origin, u_axis, v_axis)
    u_span = wall_uv[:, 0].max() - wall_uv[:, 0].min()
    v_span = wall_uv[:, 1].max() - wall_uv[:, 1].min()
    if u_span < 0.5 or u_span > 30.0:
        log.error("u-Breite der Fassade unplausibel: %.2f m", u_span)
        return EXIT_ERROR
    if v_span > 4.5:
        log.warning("v-Ausdehnung %.2f m nahe LiDAR-Limit — obere Geschosse"
                    " könnten fehlen", v_span)

    # ---- Modul 4b: AABB pro Instanz ----
    box_obs = project_rings_to_box_observations(all_rings, origin, u_axis, v_axis)
    log.info("box observations: %d", len(box_obs))

    # ---- Modul 5: Merging ----
    windows = merge_observations(
        box_obs,
        merge_distance=args.merge_distance,
        min_observations=args.min_observations,
    )
    log.info("merging: %d windows/doors", len(windows))

    # ---- Modul 6: Output ----
    csv_path = args.output / "results.csv"
    write_csv(windows, csv_path)

    # Projektion der Ringe nach Klasse für 3D-Plot
    rings_by_class: dict[str, np.ndarray] = {}
    for r in all_rings:
        rings_by_class.setdefault(r.klasse,
                                  np.zeros((0, 3), dtype=np.float32))
        rings_by_class[r.klasse] = np.concatenate(
            [rings_by_class[r.klasse], r.points], axis=0)

    render_facade_png(windows, wall_uv, args.output / "facade.png")
    render_pointcloud_png(wall_cloud, rings_by_class, normal, plane_centroid,
                          args.output / "pointcloud_debug.png")

    if not windows:
        log.warning("no instances detected — CSV enthält nur Header")
        return EXIT_EMPTY
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
