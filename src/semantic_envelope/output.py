"""Modul 6 — Ergebnis-Persistenz: CSV, 2D-Fassaden-PNG, 3D-Debug-PNG, Log."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")    # Headless — keine X-Display-Abhängigkeit
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .types import Window

log = logging.getLogger(__name__)

CSV_COLUMNS = [
    "window_id", "klasse", "breite_m", "hoehe_m", "flaeche_m2",
    "pos_u_m", "pos_v_m", "n_observations",
]


def setup_logging(out_dir: Path | str, level: str = "INFO") -> None:
    """Konfiguriere root-logger mit Konsole + ``run.log`` in ``out_dir``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / "run.log"

    root = logging.getLogger()
    # Bestehende Handler entfernen (für Tests)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(logfile, encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))


def write_csv(windows: list[Window], csv_path: Path | str) -> None:
    """Schreibe ``results.csv`` gemäß Spec §10."""
    rows = []
    for w in windows:
        rows.append({
            "window_id": w.id,
            "klasse": w.klasse,
            "breite_m": round(w.breite_m, 4),
            "hoehe_m": round(w.hoehe_m, 4),
            "flaeche_m2": round(w.flaeche_m2, 4),
            "pos_u_m": round(w.u_min, 4),
            "pos_v_m": round(w.v_min, 4),
            "n_observations": w.n_observations,
        })
    df = pd.DataFrame(rows, columns=CSV_COLUMNS)
    df.to_csv(csv_path, index=False)
    log.info("results.csv geschrieben (%d Einträge) → %s", len(df), csv_path)


def render_facade_png(windows: list[Window],
                      wall_uv: np.ndarray | None,
                      out_path: Path | str) -> None:
    """Zeichne Fassaden-Plot mit allen AABBs + optional Wand-Kontur.

    ``wall_uv`` kann eine Nx2-Wand-Punktwolke in der u/v-Ebene sein (für
    den Hintergrund-Scatter) oder ``None``.
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    if wall_uv is not None and len(wall_uv) > 0:
        ax.scatter(wall_uv[:, 0], wall_uv[:, 1], s=0.5,
                   c="#cccccc", alpha=0.4, label="wall")

    for w in windows:
        color = "#1f77b4" if w.klasse == "window" else "#d62728"
        rect = mpatches.Rectangle(
            (w.u_min, w.v_min), w.breite_m, w.hoehe_m,
            linewidth=1.5, edgecolor=color, facecolor="none")
        ax.add_patch(rect)
        ax.text(w.u_min + w.breite_m / 2, w.v_min + w.hoehe_m / 2,
                f"#{w.id}\n{w.breite_m:.2f}×{w.hoehe_m:.2f}",
                ha="center", va="center", fontsize=8, color=color)

    ax.set_aspect("equal")
    ax.set_xlabel("u (m)")
    ax.set_ylabel("v (m, Lot)")
    ax.set_title("Fassadenebene — Fenster/Türen")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("facade.png geschrieben → %s", out_path)


def render_pointcloud_png(wall_cloud: np.ndarray,
                          rings_by_class: dict[str, np.ndarray],
                          plane_normal: np.ndarray,
                          plane_origin: np.ndarray,
                          out_path: Path | str) -> None:
    """3D-Scatter der Wand-Punkte + Ringe, RANSAC-Ebene als transparente Fläche."""
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registriert 3d-projection)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Stark downsamplen für Plot-Performance
    if len(wall_cloud) > 30000:
        idx = np.random.default_rng(0).choice(len(wall_cloud), 30000, replace=False)
        wc = wall_cloud[idx]
    else:
        wc = wall_cloud
    ax.scatter(wc[:, 0], wc[:, 1], wc[:, 2], s=0.3, c="#888888", alpha=0.3,
               label="wall")

    klasse_colors = {"window": "#1f77b4", "door": "#d62728"}
    for klasse, pts in rings_by_class.items():
        if len(pts) == 0:
            continue
        color = klasse_colors.get(klasse, "#2ca02c")
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=0.6, c=color,
                   alpha=0.6, label=klasse)

    # Ebene als transparentes Rechteck um den Zentroiden
    if len(wc) > 0:
        center = plane_origin
        # Spannweite der Wand
        span = np.linalg.norm(wc.max(axis=0) - wc.min(axis=0)) / 2
        # zwei in-plane-Achsen wählen (orthogonal zu plane_normal)
        up = np.array([0.0, 1.0, 0.0])
        v = up - np.dot(up, plane_normal) * plane_normal
        v = v / (np.linalg.norm(v) + 1e-9)
        u = np.cross(v, plane_normal)
        u = u / (np.linalg.norm(u) + 1e-9)
        grid = np.array([[-span, -span], [-span, span], [span, span], [span, -span]])
        corners = np.array([center + gu * u + gv * v for gu, gv in grid])
        ax.plot_trisurf(corners[:, 0], corners[:, 1], corners[:, 2],
                        color="#ffaa00", alpha=0.15)

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("pointcloud_debug.png geschrieben → %s", out_path)
