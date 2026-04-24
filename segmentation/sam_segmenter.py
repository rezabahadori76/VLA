from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np
from pathlib import Path

from common.types import Detection, FramePacket, SegmentationMask


def _binary_mask_to_rle(mask: np.ndarray) -> Dict[str, Any]:
    flat = mask.astype(np.uint8).flatten(order='F')
    counts = []
    prev = 0
    count = 0
    for pix in flat:
        if pix == prev:
            count += 1
        else:
            counts.append(count)
            count = 1
            prev = int(pix)
    counts.append(count)
    return {"counts": counts, "size": list(mask.shape)}


class SAMSegmenter:
    def __init__(self, cfg: Dict[str, Any]):
        self.model_type = cfg.get("model_type", "vit_h")
        self.checkpoint = cfg.get("checkpoint", "sam_vit_h_4b8939.pth")
        self.device = cfg.get("device", "cuda")
        self.use_real_model = bool(cfg.get("use_real_model", True))

        self.predictor = None
        self.is_real_backend = False
        self.backend_name = "fallback"
        self.init_error: str | None = None

    def initialize(self) -> None:
        if not self.use_real_model:
            raise RuntimeError("Segmentation module requires a real model. Fallback mode is disabled.")
        try:
            from segment_anything import SamPredictor, sam_model_registry

            checkpoint = self._resolve_checkpoint(self.checkpoint)
            sam = sam_model_registry[self.model_type](checkpoint=checkpoint)
            if self.device == "cuda":
                sam.to(device="cuda")
            self.predictor = SamPredictor(sam)
            self.is_real_backend = True
            self.backend_name = "segment_anything"
        except Exception as exc:
            self.init_error = str(exc)
            self.is_real_backend = False
            self.backend_name = "init_failed"
            raise RuntimeError(f"Failed to initialize SAM backend: {self.init_error}") from exc

    def _resolve_checkpoint(self, checkpoint_path: str) -> str:
        candidate = Path(checkpoint_path)
        if candidate.is_absolute() and candidate.exists():
            return str(candidate)
        project_root = Path(__file__).resolve().parents[1]
        project_candidate = project_root / checkpoint_path
        if project_candidate.exists():
            return str(project_candidate)
        return checkpoint_path

    def segment(self, packet: FramePacket, detections: List[Detection]) -> List[SegmentationMask]:
        if not (self.is_real_backend and self.predictor is not None):
            raise RuntimeError("Segmentation backend is unavailable. Fallback mode is disabled.")
        return self._segment_real(packet, detections)

    def _segment_real(self, packet: FramePacket, detections: List[Detection]) -> List[SegmentationMask]:
        rgb = cv2.cvtColor(packet.rgb, cv2.COLOR_BGR2RGB)
        self.predictor.set_image(rgb)

        outputs: List[SegmentationMask] = []
        for det in detections:
            box = np.array(det.bbox_xyxy, dtype=np.float32)
            masks, scores, _ = self.predictor.predict(
                point_coords=None,
                point_labels=None,
                box=box[None, :],
                multimask_output=False,
            )
            mask = masks[0].astype(np.uint8)
            outputs.append(
                SegmentationMask(
                    label=det.label,
                    score=float(scores[0]),
                    mask_rle=_binary_mask_to_rle(mask),
                    bbox_xyxy=det.bbox_xyxy,
                )
            )
        return outputs

    def status(self) -> Dict[str, Any]:
        return {
            "module": "segmentation",
            "backend": self.backend_name,
            "real_backend_active": self.is_real_backend,
            "checkpoint": self.checkpoint,
            "init_error": self.init_error,
        }
