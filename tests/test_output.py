from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semantic_envelope.output import (
    write_csv,
    render_facade_png,
    setup_logging,
)
from semantic_envelope.types import Window


def test_write_csv_includes_expected_columns(tmp_path):
    windows = [
        Window(id=0, klasse="window",
               u_min=0.0, u_max=1.2, v_min=0.5, v_max=1.8, n_observations=12),
        Window(id=1, klasse="door",
               u_min=2.0, u_max=3.0, v_min=0.0, v_max=2.1, n_observations=20),
    ]
    csv = tmp_path / "results.csv"
    write_csv(windows, csv)
    df = pd.read_csv(csv)
    expected_cols = {"window_id", "klasse", "breite_m", "hoehe_m",
                     "flaeche_m2", "pos_u_m", "pos_v_m", "n_observations"}
    assert expected_cols <= set(df.columns)
    assert len(df) == 2
    # Erste Zeile
    row = df.iloc[0]
    assert row["klasse"] == "window"
    assert row["breite_m"] == pytest.approx(1.2)
    assert row["hoehe_m"] == pytest.approx(1.3)
    assert row["pos_u_m"] == pytest.approx(0.0)
    assert row["pos_v_m"] == pytest.approx(0.5)


def test_write_csv_empty_list_still_writes_header(tmp_path):
    csv = tmp_path / "empty.csv"
    write_csv([], csv)
    df = pd.read_csv(csv)
    assert len(df) == 0
    assert "window_id" in df.columns


def test_render_facade_png_creates_non_empty_file(tmp_path):
    windows = [Window(id=0, klasse="window",
                      u_min=0.0, u_max=1.2, v_min=0.5, v_max=1.8,
                      n_observations=5)]
    wall_uv = np.random.default_rng(0).uniform(-2, 3, size=(500, 2))
    out = tmp_path / "facade.png"
    render_facade_png(windows, wall_uv, out)
    assert out.exists()
    assert out.stat().st_size > 1000
