import numpy as np
import pytest

from semantic_envelope.mask_quality import (
    box_aspect_ratio_ok,
    mask_area_ok,
    convex_hull_aspect_ok,
    disambiguate_overlapping_masks,
)


def test_box_aspect_ratio_ok_for_square():
    # (x1, y1, x2, y2) im Pixel-Raum
    assert box_aspect_ratio_ok((0, 0, 100, 100)) is True


def test_box_aspect_ratio_rejects_diagonal_sliver():
    # Sehr breite, flache Box — aspect = 100/2 = 50
    assert box_aspect_ratio_ok((0, 0, 100, 2), lo=0.2, hi=5.0) is False


def test_mask_area_accepts_mid_size_window():
    img_area = 1920 * 1080
    mask = np.zeros((1080, 1920), dtype=bool)
    mask[400:600, 800:1000] = True   # 40_000 px = 1.9 %
    assert mask_area_ok(mask) is True


def test_mask_area_rejects_tiny_blob():
    mask = np.zeros((1080, 1920), dtype=bool)
    mask[0:5, 0:5] = True   # 25 px ≪ 0.5 %
    assert mask_area_ok(mask) is False


def test_mask_area_rejects_fullscreen_mask():
    mask = np.ones((1080, 1920), dtype=bool)
    assert mask_area_ok(mask, hi=0.80) is False


def test_convex_hull_aspect_ok_for_clean_rectangle():
    mask = np.zeros((200, 200), dtype=bool)
    mask[50:150, 50:150] = True
    assert convex_hull_aspect_ok(mask) is True


def test_convex_hull_aspect_rejects_zero_area_mask():
    mask = np.zeros((200, 200), dtype=bool)
    assert convex_hull_aspect_ok(mask) is False


def test_overlap_disambiguation_assigns_pixels_to_higher_score():
    m1 = np.zeros((10, 10), dtype=bool); m1[:, :6] = True
    m2 = np.zeros((10, 10), dtype=bool); m2[:, 4:] = True
    # Überlappung: Spalten 4..5 (beide True)
    out = disambiguate_overlapping_masks([m1, m2], scores=[0.9, 0.5])
    # Überlappung gehört zu m1 (höherer Score)
    assert out[0][:, 4].all()
    assert not out[1][:, 4].any()
    assert not out[1][:, 5].any()
    # m2 verliert die Überlappung, behält aber Spalte 6–9
    assert out[1][:, 6:].all()
