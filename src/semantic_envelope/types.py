"""Dataclasses für die Pipeline-Daten."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

Klasse = Literal["window", "door"]


@dataclass
class FrameData:
    """Ein akzeptierter Keyframe mit allem, was Modul 2/3 brauchen."""
    frame_id: int
    rgb: np.ndarray           # HxWx3 uint8, RGB-Reihenfolge
    depth_path: str           # Pfad zum Depth-PNG (16-bit, mm)
    confidence_path: str      # Pfad zum Confidence-PNG (0/1/2)
    pose_4x4: np.ndarray      # 4x4 float64, Kamera → Welt (ARKit)
    intrinsics: np.ndarray    # 3x3 float64, RGB-Auflösung


@dataclass
class RingData:
    """Punktwolke einer einzelnen Fenster/Tür-Instanz in einem Frame."""
    frame_id: int
    instance_id: int
    klasse: Klasse
    points: np.ndarray        # Nx3 float32, Weltkoordinaten


@dataclass
class BoxObs:
    """Eine Beobachtung einer Instanz als AABB in Fassaden-u/v-Koordinaten."""
    frame_id: int
    instance_id: int
    klasse: Klasse
    u_min: float
    u_max: float
    v_min: float
    v_max: float


@dataclass
class Window:
    """Fusioniertes Ergebnis-Fenster (oder Tür) nach Modul 5."""
    id: int
    klasse: Klasse
    u_min: float
    u_max: float
    v_min: float
    v_max: float
    n_observations: int

    @property
    def breite_m(self) -> float:
        return self.u_max - self.u_min

    @property
    def hoehe_m(self) -> float:
        return self.v_max - self.v_min

    @property
    def flaeche_m2(self) -> float:
        return self.breite_m * self.hoehe_m
