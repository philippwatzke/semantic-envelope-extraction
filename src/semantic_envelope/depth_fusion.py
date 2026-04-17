"""Modul 3 — Depth Fusion (Zwei-Strom: Wand / Fenster-Instanzen)."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .geometry import backproject_pixels, scale_intrinsics
from .segmentation import FrameSegmentation
from .types import FrameData, RingData

log = logging.getLogger(__name__)

MIN_DEPTH_HITS_PER_INSTANCE = 50
DEFAULT_MAX_DEPTH_M = 5.5
DEFAULT_MIN_CONFIDENCE = 2


def load_depth_confidence(depth_path: Path | str,
                          conf_path: Path | str,
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Lade 16-bit Depth-PNG (mm) und 8-bit Confidence-PNG (0/1/2).

    Returns
    -------
    depth_m : HxW float32 (Meter)
    conf    : HxW uint8   (0, 1, 2)
    """
    depth_mm = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth_mm is None or depth_mm.dtype != np.uint16:
        raise ValueError(f"{depth_path} ist kein 16-bit PNG")
    depth_m = depth_mm.astype(np.float32) / 1000.0

    conf = cv2.imread(str(conf_path), cv2.IMREAD_UNCHANGED)
    if conf is None:
        raise ValueError(f"{conf_path} nicht lesbar")
    if conf.ndim == 3:
        conf = conf[..., 0]
    conf = conf.astype(np.uint8)

    if depth_m.shape != conf.shape:
        raise ValueError(
            f"depth/conf shape mismatch: {depth_m.shape} vs {conf.shape}")
    return depth_m, conf


def align_mask_to_depth(mask_rgb: np.ndarray,
                        depth_shape: tuple[int, int]) -> np.ndarray:
    """Resize RGB-Auflösungs-Maske auf Depth-Auflösung via Nearest-Neighbor.

    ``depth_shape`` ist ``(H_d, W_d)`` (numpy-Konvention).
    """
    H_d, W_d = depth_shape
    resized = cv2.resize(mask_rgb.astype(np.uint8),
                         (W_d, H_d),
                         interpolation=cv2.INTER_NEAREST)
    return resized.astype(bool)


def fuse_wall_and_targets(frame: FrameData,
                          segmentation: FrameSegmentation,
                          max_depth: float = DEFAULT_MAX_DEPTH_M,
                          min_confidence: int = DEFAULT_MIN_CONFIDENCE,
                          min_depth_hits: int = MIN_DEPTH_HITS_PER_INSTANCE,
                          ) -> tuple[np.ndarray, list[RingData]]:
    """Projiziere Wand-Maske und pro Instanz in Weltkoordinaten.

    Returns
    -------
    wall_points_Nx3 : float32
    rings           : list[RingData] (nur Instanzen mit >= ``min_depth_hits`` Treffern)
    """
    depth_m, conf = load_depth_confidence(frame.depth_path, frame.confidence_path)
    H_d, W_d = depth_m.shape
    H_rgb, W_rgb = frame.rgb.shape[:2]
    K_depth = scale_intrinsics(frame.intrinsics, (W_rgb, H_rgb), (W_d, H_d))

    # Wand-Strom
    wall_mask_d = align_mask_to_depth(segmentation.wall_mask, (H_d, W_d))
    wall_points = backproject_pixels(
        mask=wall_mask_d, depth_m=depth_m, confidence=conf,
        K=K_depth, pose_4x4=frame.pose_4x4,
        max_depth=max_depth, min_confidence=min_confidence,
    )

    # Fenster-Strom (pro Instanz)
    rings: list[RingData] = []
    for inst in segmentation.target_instances:
        inst_mask_d = align_mask_to_depth(inst.mask, (H_d, W_d))
        pts = backproject_pixels(
            mask=inst_mask_d, depth_m=depth_m, confidence=conf,
            K=K_depth, pose_4x4=frame.pose_4x4,
            max_depth=max_depth, min_confidence=min_confidence,
        )
        if len(pts) < min_depth_hits:
            log.warning("frame %d inst %d: nur %d depth-hits (<%d) → verworfen",
                        frame.frame_id, inst.instance_id, len(pts), min_depth_hits)
            continue
        rings.append(RingData(frame_id=frame.frame_id,
                              instance_id=inst.instance_id,
                              klasse=inst.klasse,
                              points=pts))
    return wall_points, rings


class WallCloudAccumulator:
    """Akkumuliert Wand-Punktwolken über viele Frames ohne O(n²)-Kopien.

    In-RAM-Liste von Nx3-Arrays; einmaliges ``np.concatenate`` am Ende.
    Wenn die akkumulierte Größe ``spill_bytes`` überschreitet oder die
    Anzahl der Frames ``spill_frame_count`` übersteigt, werden die
    aktuellen Arrays als ``.npy`` auf Disk ausgelagert.
    """

    def __init__(self,
                 spill_bytes: int = 2 * 1024 ** 3,      # 2 GB
                 spill_frame_count: int = 500,
                 tmp_dir: Path | str | None = None):
        self._arrays: list[np.ndarray] = []
        self._spill_files: list[Path] = []
        self._bytes = 0
        self._frames = 0
        self._spill_bytes = spill_bytes
        self._spill_frame_count = spill_frame_count
        self._tmp_dir = Path(tmp_dir) if tmp_dir else Path(tempfile.mkdtemp(
            prefix="wallcloud_"))
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

    def append(self, points: np.ndarray) -> None:
        if len(points) == 0:
            return
        self._arrays.append(points.astype(np.float32, copy=False))
        self._bytes += points.nbytes
        self._frames += 1
        if (self._bytes >= self._spill_bytes
                or self._frames >= self._spill_frame_count):
            self._spill_to_disk()

    def _spill_to_disk(self) -> None:
        if not self._arrays:
            return
        merged = np.concatenate(self._arrays, axis=0)
        path = self._tmp_dir / f"wall_chunk_{len(self._spill_files):04d}.npy"
        np.save(path, merged)
        log.info("spilled %d wall-points to %s", len(merged), path)
        self._spill_files.append(path)
        self._arrays.clear()
        self._bytes = 0
        self._frames = 0

    def finalize(self) -> np.ndarray:
        chunks = []
        if self._spill_files:
            for p in self._spill_files:
                chunks.append(np.load(p))
        if self._arrays:
            chunks.append(np.concatenate(self._arrays, axis=0))
        self._arrays.clear()
        if not chunks:
            return np.zeros((0, 3), dtype=np.float32)
        return np.concatenate(chunks, axis=0)

    def cleanup(self) -> None:
        for p in self._spill_files:
            p.unlink(missing_ok=True)
        self._spill_files.clear()
