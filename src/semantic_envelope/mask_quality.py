"""Pre-Flight-Checks für GDINO-Boxen und SAM-Masken.

Jede Target-Instanz muss alle vier Gates bestehen (siehe Spec §6 Tabelle).
Diese Funktionen sind rein; die Entscheidung "verwerfen vs. behalten"
trifft der Aufrufer in ``segmentation.py``.
"""

from __future__ import annotations

import cv2
import numpy as np


def box_aspect_ratio_ok(xyxy: tuple[float, float, float, float],
                        lo: float = 0.2, hi: float = 5.0) -> bool:
    """GDINO-Box-Seitenverhältnis im Bereich [lo, hi]?"""
    x1, y1, x2, y2 = xyxy
    w = max(1e-6, x2 - x1)
    h = max(1e-6, y2 - y1)
    aspect = w / h
    return lo <= aspect <= hi


def mask_area_ok(mask: np.ndarray, lo: float = 0.005, hi: float = 0.80) -> bool:
    """SAM-Maskenflächen-Anteil zwischen 0.5 % und 80 % der Bildfläche?"""
    area = float(mask.sum())
    total = float(mask.size)
    if total == 0:
        return False
    frac = area / total
    return lo <= frac <= hi


def convex_hull_aspect_ok(mask: np.ndarray,
                          lo: float = 0.2, hi: float = 5.0) -> bool:
    """Aspect-Ratio der konvexen Hülle der Maske im Bereich [lo, hi]?"""
    ys, xs = np.where(mask)
    if len(xs) < 3:
        return False
    pts = np.stack([xs, ys], axis=1).astype(np.int32)
    hull = cv2.convexHull(pts)
    x, y, w, h = cv2.boundingRect(hull)
    if h == 0:
        return False
    aspect = w / h
    return lo <= aspect <= hi


def disambiguate_overlapping_masks(masks: list[np.ndarray],
                                   scores: list[float]) -> list[np.ndarray]:
    """Weise in Überlappungszonen jedes Pixel der Maske mit höherem Score zu.

    Erzeugt disjunkte Masken (keine gemeinsamen True-Pixel zwischen zwei
    Rückgabemasken). Die Eingabe-Masken werden nicht mutiert.
    """
    if not masks:
        return []
    assert len(masks) == len(scores)

    # Sortiere Indizes nach Score absteigend
    order = sorted(range(len(masks)), key=lambda i: scores[i], reverse=True)

    claimed = np.zeros_like(masks[0], dtype=bool)
    out = [np.zeros_like(m, dtype=bool) for m in masks]
    for idx in order:
        exclusive = masks[idx] & ~claimed
        out[idx] = exclusive
        claimed |= exclusive
    return out
