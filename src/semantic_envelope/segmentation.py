"""Modul 2 — Semantische Segmentierung (GDINO + SAM 2).

Drei-Prompt-Strategie:
  * target prompt    → Fenster/Türen (pro Instanz SAM-Maske)
  * positive wall    → Wandmasken
  * negative         → Distraktoren, werden als SAM-Silhouetten von der
                       Wandmaske subtrahiert

Die Modelle werden lazy in ``ThreePromptSegmenter.load()`` instanziiert
und in ``unload()`` wieder verworfen, damit Phase 2 den VRAM für
Open3D-Visualisierung nutzen kann.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import torch
from PIL import Image

from .mask_quality import (
    box_aspect_ratio_ok,
    convex_hull_aspect_ok,
    disambiguate_overlapping_masks,
    mask_area_ok,
)
from .types import Klasse

log = logging.getLogger(__name__)

TARGET_PROMPT = "window . door . glass panel"
WALL_PROMPT = "brick wall . concrete wall . facade . plaster wall"
NEGATIVE_PROMPT = "tree . bush . car . pedestrian . sky . ground . cable . signage"

# GDINO-Label → innere Klasse
TARGET_LABEL_MAP: dict[str, Klasse] = {
    "window": "window",
    "glass panel": "window",
    "door": "door",
}


@dataclass
class TargetInstance:
    instance_id: int
    klasse: Klasse
    mask: np.ndarray          # HxW bool, RGB-Auflösung
    score: float              # GDINO-Box-Score


@dataclass
class FrameSegmentation:
    target_instances: list[TargetInstance]
    wall_mask: np.ndarray     # HxW bool, RGB-Auflösung


class GroundingDinoDetector:
    """Dünner Wrapper um HuggingFace's ``AutoModelForZeroShotObjectDetection``."""

    def __init__(self,
                 model_id: str = "IDEA-Research/grounding-dino-tiny",
                 box_threshold: float = 0.35,
                 text_threshold: float = 0.25,
                 device: str = "cuda"):
        from transformers import (AutoProcessor,
                                  AutoModelForZeroShotObjectDetection)
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id).to(device).eval()

    @torch.inference_mode()
    def detect(self, rgb: np.ndarray, prompt: str,
               ) -> list[tuple[tuple[float, float, float, float], float, str]]:
        """Liefert ``[(xyxy, score, label), ...]`` für einen einzelnen Prompt."""
        img = Image.fromarray(rgb)
        inputs = self.processor(images=img, text=prompt,
                                return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[img.size[::-1]],   # (H, W)
        )[0]
        out = []
        for box, score, label in zip(results["boxes"], results["scores"],
                                     results["labels"]):
            xyxy = tuple(box.tolist())
            out.append((xyxy, float(score), str(label)))
        return out

    def to(self, device: str) -> None:
        self.device = device
        self.model = self.model.to(device)

    def close(self) -> None:
        del self.model
        del self.processor
        torch.cuda.empty_cache()


class Sam2BoxSegmenter:
    """Wrapper um ``SAM2ImagePredictor`` — Box-Prompts → binäre Masken."""

    def __init__(self, model_size: Literal["base", "large"] = "base",
                 device: str = "cuda"):
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        hf_id = {
            "base":  "facebook/sam2-hiera-base-plus",
            "large": "facebook/sam2-hiera-large",
        }[model_size]
        self.device = device
        self.predictor = SAM2ImagePredictor.from_pretrained(hf_id, device=device)

    def set_image(self, rgb: np.ndarray) -> None:
        self.predictor.set_image(rgb)

    @torch.inference_mode()
    def masks_for_boxes(self,
                        boxes_xyxy: list[tuple[float, float, float, float]],
                        ) -> list[np.ndarray]:
        """Liefere pro Box eine HxW-Bool-Maske. Leere Eingabe → []."""
        if not boxes_xyxy:
            return []
        box_arr = np.array(boxes_xyxy, dtype=np.float32)
        masks, _scores, _logits = self.predictor.predict(
            point_coords=None, point_labels=None,
            box=box_arr, multimask_output=False,
        )
        # masks: (N, 1, H, W) oder (N, H, W) je nach SAM-2-Version
        if masks.ndim == 4:
            masks = masks[:, 0]
        return [m.astype(bool) for m in masks]

    def close(self) -> None:
        del self.predictor
        torch.cuda.empty_cache()


@dataclass
class ThreePromptSegmenter:
    gdino_model_id: str = "IDEA-Research/grounding-dino-tiny"
    sam_model_size: Literal["base", "large"] = "base"
    box_threshold: float = 0.35
    text_threshold: float = 0.25
    device: str = "cuda"

    _gdino: GroundingDinoDetector | None = field(default=None, init=False)
    _sam: Sam2BoxSegmenter | None = field(default=None, init=False)

    def load(self) -> None:
        log.info("lade GDINO (%s) + SAM 2 (%s) auf %s",
                 self.gdino_model_id, self.sam_model_size, self.device)
        self._gdino = GroundingDinoDetector(
            model_id=self.gdino_model_id,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
        )
        self._sam = Sam2BoxSegmenter(
            model_size=self.sam_model_size, device=self.device)

    def unload(self) -> None:
        if self._gdino is not None:
            self._gdino.close()
            self._gdino = None
        if self._sam is not None:
            self._sam.close()
            self._sam = None

    def __enter__(self) -> "ThreePromptSegmenter":
        self.load()
        return self

    def __exit__(self, *exc_info) -> None:
        self.unload()

    def segment_frame(self, rgb: np.ndarray, frame_id: int
                      ) -> FrameSegmentation:
        """Führe GDINO+SAM-2 für alle drei Prompts auf einem Frame aus."""
        assert self._gdino is not None and self._sam is not None, "call load() first"
        H, W = rgb.shape[:2]
        self._sam.set_image(rgb)

        # ---- Targets (Fenster/Türen) ----
        target_dets = self._gdino.detect(rgb, TARGET_PROMPT)
        target_dets = [(b, s, l) for (b, s, l) in target_dets
                       if box_aspect_ratio_ok(b) and _label_to_klasse(l) is not None]
        target_masks = self._sam.masks_for_boxes([d[0] for d in target_dets])

        # Qualitätsfilter + Klassenzuordnung
        accepted: list[TargetInstance] = []
        for i, (mask, (xyxy, score, label)) in enumerate(zip(target_masks, target_dets)):
            klasse = _label_to_klasse(label)
            if klasse is None:
                continue
            if not mask_area_ok(mask):
                log.warning("frame %d inst %d: mask-area gate → verworfen", frame_id, i)
                continue
            if not convex_hull_aspect_ok(mask):
                log.warning("frame %d inst %d: hull-aspect gate → verworfen", frame_id, i)
                continue
            accepted.append(TargetInstance(instance_id=i, klasse=klasse,
                                           mask=mask, score=score))

        # Überlappungen desambiguieren
        if accepted:
            masks_exclusive = disambiguate_overlapping_masks(
                [t.mask for t in accepted], [t.score for t in accepted])
            for t, m in zip(accepted, masks_exclusive):
                t.mask = m

        # ---- Wände ----
        wall_dets = self._gdino.detect(rgb, WALL_PROMPT)
        wall_masks = self._sam.masks_for_boxes([d[0] for d in wall_dets])
        wall_mask = np.zeros((H, W), dtype=bool)
        for m in wall_masks:
            wall_mask |= m

        # ---- Negativ-Distraktoren → präzise SAM-Silhouette ----
        neg_dets = self._gdino.detect(rgb, NEGATIVE_PROMPT)
        neg_masks = self._sam.masks_for_boxes([d[0] for d in neg_dets])
        for m in neg_masks:
            wall_mask &= ~m

        # Target-Pixel sind ebenfalls nicht Wand
        for t in accepted:
            wall_mask &= ~t.mask

        log.debug("frame %d: %d targets, %d wall-boxes, %d neg-boxes",
                  frame_id, len(accepted), len(wall_dets), len(neg_dets))

        return FrameSegmentation(target_instances=accepted, wall_mask=wall_mask)


def _label_to_klasse(label: str) -> Klasse | None:
    """Mappt den GDINO-Textlabel auf ``Klasse`` — None = unbekannt."""
    label = label.strip().lower()
    for key, klasse in TARGET_LABEL_MAP.items():
        if key in label:
            return klasse
    return None
