"""End-to-End-Test: extract.py auf einem Sample-ZIP.

Läuft nur mit ``pytest -m gpu`` manuell. Erwartungen:
  * Exit-Code 0 oder 2
  * Alle drei Output-Files existieren
  * results.csv hat mindestens eine Daten-Zeile ODER Exit 2
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ZIP = REPO_ROOT / "data" / "stray_scanner" / "228fb53d88.zip"


@pytest.mark.gpu
@pytest.mark.slow
def test_smoke_end_to_end(tmp_path):
    if not SAMPLE_ZIP.exists():
        pytest.skip(f"kein Sample-ZIP: {SAMPLE_ZIP}")

    out_dir = tmp_path / "results"
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "extract.py"),
         "--input", str(SAMPLE_ZIP),
         "--output", str(out_dir),
         "--log-level", "INFO"],
        capture_output=True, text=True, cwd=REPO_ROOT, env=env,
    )
    print("STDOUT:", proc.stdout[-2000:])
    print("STDERR:", proc.stderr[-2000:])
    assert proc.returncode in (0, 2), f"exit code {proc.returncode}"

    assert (out_dir / "results.csv").exists()
    assert (out_dir / "facade.png").exists()
    assert (out_dir / "pointcloud_debug.png").exists()
    assert (out_dir / "run.log").exists()

    df = pd.read_csv(out_dir / "results.csv")
    expected_cols = {"window_id", "klasse", "breite_m", "hoehe_m",
                     "flaeche_m2", "pos_u_m", "pos_v_m", "n_observations"}
    assert expected_cols <= set(df.columns)

    if proc.returncode == 0:
        # Exit 0 = mindestens ein Fenster gefunden
        assert len(df) >= 1
        # Plausibilitäts-Bereiche
        assert df["breite_m"].between(0.3, 5.0).all()
        assert df["hoehe_m"].between(0.3, 4.0).all()
