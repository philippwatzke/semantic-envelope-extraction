"""Modul 1 — Ingestion: ZIP entpacken, Odometry/Keyframes, Drift monitoren."""

from __future__ import annotations

import csv
import logging
import shutil
import zipfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .geometry import quaternion_to_pose_matrix
from .types import FrameData

log = logging.getLogger(__name__)


def parse_odometry(csv_path: Path | str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Parse Stray-Scanner-Odometry-CSV.

    Erwartet Spalten: ``timestamp, frame, x, y, z, qx, qy, qz, qw,
    fx, fy, cx, cy`` (in dieser Reihenfolge).

    Returns
    -------
    dict[int, (pose_4x4, K_3x3)]
    """
    result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    with Path(csv_path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Strip whitespace from keys to handle CSV files with spaces after commas
            row = {k.strip(): v for k, v in row.items()}
            frame_id = int(row["frame"])
            pose = quaternion_to_pose_matrix(
                tx=float(row["x"]), ty=float(row["y"]), tz=float(row["z"]),
                qx=float(row["qx"]), qy=float(row["qy"]),
                qz=float(row["qz"]), qw=float(row["qw"]),
            )
            K = np.array([
                [float(row["fx"]), 0.0,             float(row["cx"])],
                [0.0,              float(row["fy"]), float(row["cy"])],
                [0.0,              0.0,              1.0],
            ])
            result[frame_id] = (pose, K)
    return result


def parse_camera_matrix(csv_path: Path | str) -> np.ndarray:
    """Parse `camera_matrix.csv` als 3x3-Array (Fallback-Intrinsics)."""
    return np.loadtxt(Path(csv_path), delimiter=",", dtype=np.float64)


def compute_keyframe_step(fps: float, target_fps: float = 3.0) -> int:
    """Subsampling-Faktor: jeder n-te Frame aus dem Video.

    Ergebnis ist mindestens 1, damit bei Quell-Framerate <= target_fps
    alle Frames akzeptiert werden.
    """
    if fps <= 0:
        raise ValueError(f"invalid fps: {fps}")
    step = round(fps / target_fps)
    return max(1, int(step))


def monitor_drift(poses: list[np.ndarray],
                  window: int = 5,
                  threshold_m: float = 0.5) -> list[tuple[int, float]]:
    """Rollende-Median-Drift-Warnung über ARKit-Translationen.

    Für jedes Fenster aus ``window`` aufeinanderfolgenden Frames wird der
    Median der L2-Sprünge ``‖t_{i+1} - t_i‖`` gebildet. Überschreitet er
    ``threshold_m``, wird ein Warn-Eintrag (Index des Fensterstarts, Median)
    zurückgegeben.
    """
    if len(poses) < window + 1:
        return []
    translations = np.array([p[:3, 3] for p in poses])
    jumps = np.linalg.norm(np.diff(translations, axis=0), axis=1)

    warnings: list[tuple[int, float]] = []
    for start in range(len(jumps) - window + 1):
        win = jumps[start:start + window]
        m = float(np.median(win))
        if m > threshold_m:
            warnings.append((start, m))
    return warnings


def _laplacian_variance(rgb: np.ndarray) -> float:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def extract_keyframes(mp4_path: Path | str,
                      blur_threshold: float = 100.0,
                      target_fps: float = 3.0,
                      ) -> Iterable[tuple[int, np.ndarray]]:
    """Keyframe-Generator: FPS-adaptives Subsampling + Blur-Gate.

    Gibt Paare ``(frame_id, rgb_uint8_HWx3)`` zurück. Unscharfe Frames
    (Laplacian-Varianz < ``blur_threshold``) werden verworfen.
    """
    cap = cv2.VideoCapture(str(mp4_path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = compute_keyframe_step(fps, target_fps=target_fps)
        log.info("keyframe step = %d (source fps = %.2f)", step, fps)

        frame_id = -1
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frame_id += 1
            if frame_id % step != 0:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if _laplacian_variance(rgb) < blur_threshold:
                log.debug("blur reject frame %d", frame_id)
                continue
            yield frame_id, rgb
    finally:
        cap.release()


def ingest_zip(zip_path: Path | str,
               work_dir: Path | str,
               blur_threshold: float = 100.0,
               target_fps: float = 3.0,
               ) -> tuple[list[FrameData], Path]:
    """Entpackt ZIP nach ``work_dir/<scan_id>/`` und liefert ``FrameData``-Liste.

    Returns
    -------
    frames : list[FrameData]
        Akzeptierte Keyframes, nach frame_id sortiert.
    unpacked_dir : Path
        Verzeichnis, in das entpackt wurde (wird NICHT automatisch aufgeräumt,
        damit Zwischenergebnisse debugbar bleiben).
    """
    zip_path = Path(zip_path)
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    unpacked = work_dir / zip_path.stem
    if unpacked.exists():
        shutil.rmtree(unpacked)
    unpacked.mkdir()

    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(unpacked)
    log.info("unpacked %s → %s", zip_path, unpacked)

    # Stray Scanner legt manchmal einen Subordner mit Scan-ID an — normalisieren
    entries = list(unpacked.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        root = entries[0]
    else:
        root = unpacked

    odometry = parse_odometry(root / "odometry.csv")
    mp4 = root / "rgb.mp4"
    depth_dir = root / "depth"
    conf_dir = root / "confidence"

    if not mp4.exists() or not depth_dir.is_dir() or not conf_dir.is_dir():
        raise FileNotFoundError(
            f"erwartete Struktur fehlt in {root}: rgb.mp4/depth/confidence")

    frames: list[FrameData] = []
    for frame_id, rgb in extract_keyframes(mp4, blur_threshold=blur_threshold,
                                           target_fps=target_fps):
        if frame_id not in odometry:
            log.debug("no odometry for frame %d — skip", frame_id)
            continue
        depth_png = depth_dir / f"{frame_id:06d}.png"
        conf_png = conf_dir / f"{frame_id:06d}.png"
        if not depth_png.exists() or not conf_png.exists():
            log.debug("no depth/confidence for frame %d — skip", frame_id)
            continue
        pose, K = odometry[frame_id]
        frames.append(FrameData(
            frame_id=frame_id,
            rgb=rgb,
            depth_path=str(depth_png),
            confidence_path=str(conf_png),
            pose_4x4=pose,
            intrinsics=K,
        ))

    frames.sort(key=lambda f: f.frame_id)
    log.info("accepted %d keyframes", len(frames))

    # Drift-Monitoring
    poses = [f.pose_4x4 for f in frames]
    for idx, med in monitor_drift(poses):
        log.warning("ARKit-Drift bei Fenster-Start %d: median-jump=%.2f m", idx, med)

    return frames, root
