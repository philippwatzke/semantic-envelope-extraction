import numpy as np
import pytest

from semantic_envelope.merging import (
    build_adjacency,
    merge_observations,
)
from semantic_envelope.types import BoxObs


def _obs(frame, inst, klasse, u_min, u_max, v_min, v_max):
    return BoxObs(frame_id=frame, instance_id=inst, klasse=klasse,
                  u_min=u_min, u_max=u_max, v_min=v_min, v_max=v_max)


def test_adjacency_only_within_same_class():
    # Ein Fenster und eine Tür direkt nebeneinander — dürfen NICHT
    # verbunden sein, obwohl sie räumlich überlappen.
    obs = [
        _obs(1, 0, "window", 0.0, 1.0, 0.0, 1.5),
        _obs(1, 1, "door",   0.5, 1.2, 0.0, 2.0),
    ]
    adj = build_adjacency(obs, merge_distance=0.15)
    assert adj[0, 1] == 0
    assert adj[1, 0] == 0


def test_adjacency_connects_partial_views_of_wide_window():
    # Linke und rechte Hälfte desselben Fensters: IoU ~ 0, aber Abstand 0.05 m
    obs = [
        _obs(1, 0, "window", 0.0, 1.0, 0.0, 1.5),
        _obs(2, 0, "window", 1.05, 2.0, 0.0, 1.5),
    ]
    adj = build_adjacency(obs, merge_distance=0.15)
    assert adj[0, 1] == 1


def test_adjacency_does_not_connect_neighbour_windows():
    # Zwei getrennte Fenster mit 30 cm Laibung
    obs = [
        _obs(1, 0, "window", 0.0, 1.0, 0.0, 1.5),
        _obs(2, 0, "window", 1.30, 2.3, 0.0, 1.5),
    ]
    adj = build_adjacency(obs, merge_distance=0.15)
    assert adj[0, 1] == 0


def test_merge_discards_cluster_with_fewer_than_min_obs():
    obs = [
        _obs(1, 0, "window", 0.0, 1.0, 0.0, 1.5),
        _obs(2, 0, "window", 0.02, 1.03, 0.0, 1.5),
    ]  # nur 2 Beobachtungen → verwerfen
    windows = merge_observations(obs, merge_distance=0.15, min_observations=3)
    assert windows == []


def test_merge_uses_percentile_over_observations():
    # 10 Beobachtungen desselben Fensters, davon eine Ausreißer-Maske
    # mit einem um 0.5 m aufgeblasenen Rechteck.
    obs = [_obs(i, 0, "window", 0.0, 1.0, 0.0, 1.5) for i in range(10)]
    obs.append(_obs(99, 0, "window", -0.5, 1.5, -0.3, 2.0))   # Ausreißer
    obs = list(sorted(obs, key=lambda b: b.frame_id))
    windows = merge_observations(obs, merge_distance=0.15, min_observations=3)
    assert len(windows) == 1
    w = windows[0]
    # 95. Perzentil liegt deutlich näher an 1.0 als an 1.5
    assert w.u_max < 1.1
    assert w.u_min > -0.1
    assert w.n_observations == 11
