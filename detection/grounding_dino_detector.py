from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import cv2
import torch
from pathlib import Path

from common.types import Detection, FramePacket


@dataclass
class _TrackedDetection:
    label: str
    bbox_xyxy: Tuple[float, float, float, float]
    score: float
    hits: int
    missed: int


class GroundingDinoDetector:
    def __init__(self, cfg: Dict[str, Any]):
        self.model_name = cfg.get("model_name", "IDEA-Research/grounding-dino-tiny")
        self.box_threshold = float(cfg.get("box_threshold", 0.3))
        self.text_threshold = float(cfg.get("text_threshold", 0.25))
        self.prompt = cfg.get("prompt", "")
        self.labels = [x.strip() for x in self.prompt.split('.') if x.strip()]
        self.device = cfg.get("device", "cuda")
        self.use_real_model = bool(cfg.get("use_real_model", True))
        self.stabilization_enabled = bool(cfg.get("stabilization_enabled", True))
        self.track_iou_threshold = float(cfg.get("track_iou_threshold", 0.45))
        self.track_smoothing_alpha = float(cfg.get("track_smoothing_alpha", 0.65))
        self.track_min_hits = max(1, int(cfg.get("track_min_hits", 2)))
        self.track_max_missed = max(0, int(cfg.get("track_max_missed", 3)))
        self.track_score_decay = float(cfg.get("track_score_decay", 0.92))
        self.track_min_score = float(cfg.get("track_min_score", 0.2))

        self.processor = None
        self.model = None
        self.is_real_backend = False
        self.backend_name = "fallback"
        self.init_error: str | None = None
        self._tracks: Dict[int, _TrackedDetection] = {}
        self._next_track_id = 1

    def initialize(self) -> None:
        if not self.use_real_model:
            raise RuntimeError("Detection module requires a real model. Fallback mode is disabled.")

        try:
            from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

            model_ref = self._resolve_model_ref(self.model_name)
            self.processor = AutoProcessor.from_pretrained(model_ref)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_ref)

            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            else:
                self.model = self.model.to("cpu")

            self.model.eval()
            self.is_real_backend = True
            self.backend_name = "grounding_dino_transformers"
        except Exception as exc:
            self.init_error = str(exc)
            self.is_real_backend = False
            self.backend_name = "init_failed"
            raise RuntimeError(f"Failed to initialize Grounding DINO backend: {self.init_error}") from exc

    def _resolve_model_ref(self, model_name: str) -> str:
        candidate = Path(model_name)
        if candidate.is_absolute() and candidate.exists():
            return str(candidate)
        project_root = Path(__file__).resolve().parents[1]
        project_candidate = project_root / model_name
        if project_candidate.exists():
            return str(project_candidate)
        return model_name

    def detect(self, packet: FramePacket) -> List[Detection]:
        if not (self.is_real_backend and self.model is not None and self.processor is not None):
            raise RuntimeError("Detection backend is unavailable. Fallback mode is disabled.")
        return self._detect_real(packet)

    def _detect_real(self, packet: FramePacket) -> List[Detection]:
        rgb = cv2.cvtColor(packet.rgb, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, text=self.prompt, return_tensors="pt")
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = self.model(**inputs)

        results = self.processor.post_process_grounded_object_detection(
            outputs,
            inputs["input_ids"],
            threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            target_sizes=[rgb.shape[:2]],
        )
        parsed = results[0]
        boxes = parsed.get("boxes", [])
        scores = parsed.get("scores", [])
        labels = parsed.get("text_labels", parsed.get("labels", []))

        detections: List[Detection] = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            label_text = str(label).strip().lower()
            if not label_text:
                continue
            detections.append(
                Detection(
                    label=label_text,
                    score=float(score),
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        del outputs
        del inputs
        if not self.stabilization_enabled:
            return detections
        return self._stabilize_detections(detections)

    def _stabilize_detections(self, detections: List[Detection]) -> List[Detection]:
        unmatched_track_ids = set(self._tracks.keys())
        unmatched_detection_indices = set(range(len(detections)))

        # Greedy one-to-one assignment by IoU for same-label detections.
        match_candidates: List[Tuple[float, int, int]] = []
        for det_idx, det in enumerate(detections):
            for track_id, track in self._tracks.items():
                if det.label != track.label:
                    continue
                iou = self._bbox_iou(det.bbox_xyxy, track.bbox_xyxy)
                if iou >= self.track_iou_threshold:
                    match_candidates.append((iou, det_idx, track_id))
        match_candidates.sort(reverse=True, key=lambda x: x[0])

        for _, det_idx, track_id in match_candidates:
            if det_idx not in unmatched_detection_indices or track_id not in unmatched_track_ids:
                continue
            det = detections[det_idx]
            track = self._tracks[track_id]
            track.bbox_xyxy = self._smooth_bbox(track.bbox_xyxy, det.bbox_xyxy, self.track_smoothing_alpha)
            track.score = max(det.score, track.score * self.track_score_decay)
            track.hits += 1
            track.missed = 0
            unmatched_detection_indices.remove(det_idx)
            unmatched_track_ids.remove(track_id)

        # New tracks for unmatched detections.
        for det_idx in unmatched_detection_indices:
            det = detections[det_idx]
            self._tracks[self._next_track_id] = _TrackedDetection(
                label=det.label,
                bbox_xyxy=det.bbox_xyxy,
                score=det.score,
                hits=1,
                missed=0,
            )
            self._next_track_id += 1

        # Age out unmatched tracks, but keep for a few frames to reduce flicker.
        for track_id in list(unmatched_track_ids):
            track = self._tracks[track_id]
            track.missed += 1
            track.score *= self.track_score_decay
            if track.missed > self.track_max_missed or track.score < self.track_min_score:
                del self._tracks[track_id]

        stabilized: List[Detection] = []
        for track in self._tracks.values():
            if track.hits < self.track_min_hits:
                continue
            stabilized.append(
                Detection(
                    label=track.label,
                    score=track.score,
                    bbox_xyxy=track.bbox_xyxy,
                )
            )
        return stabilized

    @staticmethod
    def _smooth_bbox(
        prev_bbox: Tuple[float, float, float, float],
        new_bbox: Tuple[float, float, float, float],
        alpha: float,
    ) -> Tuple[float, float, float, float]:
        return tuple(
            float(alpha * prev_v + (1.0 - alpha) * new_v)
            for prev_v, new_v in zip(prev_bbox, new_bbox)
        )

    @staticmethod
    def _bbox_iou(
        box_a: Tuple[float, float, float, float],
        box_b: Tuple[float, float, float, float],
    ) -> float:
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_w = max(0.0, inter_x2 - inter_x1)
        inter_h = max(0.0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h
        if inter_area <= 0.0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0.0:
            return 0.0
        return inter_area / union

    def status(self) -> Dict[str, Any]:
        return {
            "module": "detection",
            "backend": self.backend_name,
            "real_backend_active": self.is_real_backend,
            "model_name": self.model_name,
            "init_error": self.init_error,
        }
