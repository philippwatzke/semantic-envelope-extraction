"""Modul 5 — Spatial-Graph-Merging klassengebunden mit 5./95.-Perzentil."""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
from scipy.sparse.csgraph import connected_components

from .geometry import rect_edge_distance_2d
from .types import BoxObs, Window

log = logging.getLogger(__name__)


def build_adjacency(obs: list[BoxObs],
                    merge_distance: float = 0.15) -> np.ndarray:
    """Dichte Adjazenzmatrix: Kante ⟺ gleiche Klasse ∧ edge-Distanz < ``merge_distance``."""
    n = len(obs)
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        a = obs[i]
        rect_a = (a.u_min, a.u_max, a.v_min, a.v_max)
        for j in range(i + 1, n):
            b = obs[j]
            if a.klasse != b.klasse:
                continue
            rect_b = (b.u_min, b.u_max, b.v_min, b.v_max)
            d = rect_edge_distance_2d(rect_a, rect_b)
            if d < merge_distance:
                adj[i, j] = 1
                adj[j, i] = 1
    return adj


def _aggregate_cluster(cluster: list[BoxObs],
                       cluster_id: int,
                       percentile_lo: float = 5.0,
                       percentile_hi: float = 95.0) -> Window:
    """Fasse Beobachtungen eines Clusters über 5./95. Perzentil zusammen."""
    u_mins = np.array([b.u_min for b in cluster])
    u_maxs = np.array([b.u_max for b in cluster])
    v_mins = np.array([b.v_min for b in cluster])
    v_maxs = np.array([b.v_max for b in cluster])
    return Window(
        id=cluster_id,
        klasse=cluster[0].klasse,
        u_min=float(np.percentile(u_mins, percentile_lo, method="higher")),
        u_max=float(np.percentile(u_maxs, percentile_hi, method="lower")),
        v_min=float(np.percentile(v_mins, percentile_lo, method="higher")),
        v_max=float(np.percentile(v_maxs, percentile_hi, method="lower")),
        n_observations=len(cluster),
    )


def merge_observations(obs: list[BoxObs],
                       merge_distance: float = 0.15,
                       min_observations: int = 3,
                       ) -> list[Window]:
    """Baue Graph, extrahiere Connected Components, aggregiere pro Cluster.

    Cluster mit weniger als ``min_observations`` werden verworfen (WARN).
    """
    if not obs:
        return []

    adj = build_adjacency(obs, merge_distance=merge_distance)
    n_components, labels = connected_components(
        csgraph=adj, directed=False, return_labels=True)

    buckets: dict[int, list[BoxObs]] = defaultdict(list)
    for i, lbl in enumerate(labels):
        buckets[int(lbl)].append(obs[i])

    windows: list[Window] = []
    next_id = 0
    for lbl, cluster in buckets.items():
        if len(cluster) < min_observations:
            log.warning("cluster %d (%s, %d beobachtungen) < min_obs=%d → verworfen",
                        lbl, cluster[0].klasse, len(cluster), min_observations)
            continue
        windows.append(_aggregate_cluster(cluster, cluster_id=next_id))
        next_id += 1

    # Deterministische Reihenfolge: zuerst nach Klasse, dann nach u_min
    windows.sort(key=lambda w: (w.klasse, w.u_min, w.v_min))
    for i, w in enumerate(windows):
        w.id = i
    return windows
