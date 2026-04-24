from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np
import torch
from pathlib import Path

from common.types import Detection, FramePacket


class GroundingDinoDetector:
    def __init__(self, cfg: Dict[str, Any]):
        self.model_name = cfg.get("model_name", "IDEA-Research/grounding-dino-tiny")
        self.box_threshold = float(cfg.get("box_threshold", 0.3))
        self.text_threshold = float(cfg.get("text_threshold", 0.25))
        self.prompt = cfg.get("prompt", "")
        self.labels = [x.strip() for x in self.prompt.split('.') if x.strip()]
        self.device = cfg.get("device", "cuda")
        self.use_real_model = bool(cfg.get("use_real_model", True))

        self.processor = None
        self.model = None
        self.is_real_backend = False
        self.backend_name = "fallback"
        self.init_error: str | None = None

    def initialize(self) -> None:
        if not self.use_real_model:
            self.backend_name = "fallback_disabled_by_config"
            return

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
            self.backend_name = "fallback_init_failed"

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
        if self.is_real_backend and self.model is not None and self.processor is not None:
            return self._detect_real(packet)
        return self._detect_fallback(packet)

    def _detect_real(self, packet: FramePacket) -> List[Detection]:
        rgb = cv2.cvtColor(packet.rgb, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, text=self.prompt, return_tensors="pt")
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.no_grad():
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
        labels = parsed.get("labels", [])

        detections: List[Detection] = []
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = [float(v) for v in box.tolist()]
            detections.append(
                Detection(
                    label=str(label),
                    score=float(score),
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        return detections

    def _detect_fallback(self, packet: FramePacket) -> List[Detection]:
        h, w = packet.rgb.shape[:2]
        if not self.labels:
            return []

        rng = np.random.default_rng(seed=packet.frame_id)
        n = min(3, len(self.labels))
        detections: List[Detection] = []
        for idx in range(n):
            x1 = float(rng.integers(0, max(1, int(w * 0.6))))
            y1 = float(rng.integers(0, max(1, int(h * 0.6))))
            bw = float(rng.integers(max(10, int(w * 0.1)), max(20, int(w * 0.3))))
            bh = float(rng.integers(max(10, int(h * 0.1)), max(20, int(h * 0.3))))
            x2 = min(float(w - 1), x1 + bw)
            y2 = min(float(h - 1), y1 + bh)
            detections.append(
                Detection(label=self.labels[idx], score=0.55 + 0.1 * idx, bbox_xyxy=(x1, y1, x2, y2))
            )
        return detections

    def status(self) -> Dict[str, Any]:
        return {
            "module": "detection",
            "backend": self.backend_name,
            "real_backend_active": self.is_real_backend,
            "model_name": self.model_name,
            "init_error": self.init_error,
        }
