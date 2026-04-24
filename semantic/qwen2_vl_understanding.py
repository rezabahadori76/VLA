from __future__ import annotations

import json
import re
from typing import Any, Dict, List

import cv2
import torch
from pathlib import Path
from PIL import Image

from common.types import Detection, FramePacket, SemanticFrame


class Qwen2VLSemanticModule:
    def __init__(self, cfg: Dict[str, Any]):
        self.model_name = cfg.get("model_name", "Qwen/Qwen2-VL-2B-Instruct")
        self.room_labels = cfg.get(
            "room_labels",
            ["kitchen", "bedroom", "living_room", "bathroom", "hallway", "office", "dining_room"],
        )
        self.device = cfg.get("device", "cuda")
        self.use_real_model = bool(cfg.get("use_real_model", True))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 120))

        self.processor = None
        self.model = None
        self.is_real_backend = False
        self.backend_name = "fallback"
        self.init_error: str | None = None

    def initialize(self) -> None:
        if not self.use_real_model:
            raise RuntimeError("Semantic module requires a real VLM backend. Fallback mode is disabled.")
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            model_ref = self._resolve_model_ref(self.model_name)
            self.processor = AutoProcessor.from_pretrained(model_ref, trust_remote_code=True)
            dtype = torch.float16 if (self.device == "cuda" and torch.cuda.is_available()) else torch.float32
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_ref,
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            if self.device == "cuda" and torch.cuda.is_available():
                self.model = self.model.to("cuda")
            else:
                self.model = self.model.to("cpu")

            self.model.eval()
            self.is_real_backend = True
            self.backend_name = "qwen_vl_transformers"
        except Exception as exc:
            self.init_error = str(exc)
            self.is_real_backend = False
            self.backend_name = "init_failed"
            raise RuntimeError(f"Failed to initialize Qwen-VL backend: {self.init_error}") from exc

    def _resolve_model_ref(self, model_name: str) -> str:
        candidate = Path(model_name)
        if candidate.is_absolute() and candidate.exists():
            return str(candidate)
        project_root = Path(__file__).resolve().parents[1]
        project_candidate = project_root / model_name
        if project_candidate.exists():
            return str(project_candidate)
        return model_name

    def infer(self, packet: FramePacket, detections: List[Detection]) -> SemanticFrame:
        if not (self.is_real_backend and self.model is not None and self.processor is not None):
            raise RuntimeError("Semantic backend is unavailable. Fallback mode is disabled.")
        return self._infer_real(packet, detections)

    def _infer_real(self, packet: FramePacket, detections: List[Detection]) -> SemanticFrame:
        rgb = cv2.cvtColor(packet.rgb, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        objects = [d.label for d in detections]
        unique_objects = sorted({o for o in objects if o})
        room_choices = ", ".join(self.room_labels)
        prompt = (
            "You are an accurate robotics indoor semantic mapper.\n"
            "Return EXACTLY one JSON object and nothing else.\n"
            "Use this strict schema:\n"
            '{"room_label":"<one_of_allowed_labels>","caption":"<short factual sentence>"}\n'
            f"Allowed room_label values: [{room_choices}].\n"
            f"Detected objects hint (unique): {unique_objects}.\n"
            "Prefer room decision using durable structures (walls, floor type, counters, bed, toilet, desk, sofa).\n"
            "Caption should mention key furniture and layout cues.\n"
            "Do not add markdown, code fences, explanations, or extra keys."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text_prompt], images=[image], return_tensors="pt")
        if self.device == "cuda" and torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        # Decode only newly generated tokens (not the full input prompt).
        generated_trimmed = [
            out_ids[in_ids.shape[0] :] for in_ids, out_ids in zip(inputs["input_ids"], generated)
        ]
        text = self.processor.batch_decode(generated_trimmed, skip_special_tokens=True)[0]
        del generated
        del inputs
        parsed = self._safe_parse_json(text)
        if not parsed:
            parsed = self._fallback_parse_from_text(text)
        room = str(parsed.get("room_label", "unknown")).strip().lower().replace(" ", "_")
        if room not in set(self.room_labels):
            room = self._infer_room_from_objects(unique_objects)
        caption = str(parsed.get("caption", "")).strip()
        if not caption:
            caption = "Indoor scene observed."
        return SemanticFrame(room_label=room, caption=caption, attributes={"objects": objects, "raw_text": text})

    def _safe_parse_json(self, text: str) -> Dict[str, Any] | None:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            lowered = text.lower()
            room = None
            for candidate in self.room_labels:
                if candidate.lower() in lowered:
                    room = candidate.lower()
                    break
            caption = None
            m = re.search(r"caption\s*[:=]\s*['\"]?(.+?)['\"]?(?:\n|$)", text, flags=re.IGNORECASE)
            if m:
                caption = m.group(1).strip()
            if room and caption:
                return {"room_label": room, "caption": caption}
            return None
        try:
            return json.loads(text[start : end + 1])
        except Exception:
            return None

    def _fallback_parse_from_text(self, text: str) -> Dict[str, Any]:
        lowered = text.lower()
        room = "unknown"
        for candidate in self.room_labels:
            if candidate.lower() in lowered:
                room = candidate.lower()
                break

        caption = ""
        for line in text.splitlines():
            cleaned = line.strip().strip("`").strip()
            if not cleaned:
                continue
            if cleaned.startswith("{") or cleaned.startswith("}"):
                continue
            caption = cleaned
            break
        if not caption:
            caption = "Indoor scene observed."
        return {"room_label": room, "caption": caption}

    def _infer_room_from_objects(self, objects: List[str]) -> str:
        if not objects:
            return "unknown"
        joined = " ".join(objects).lower()
        scores = {k: 0 for k in self.room_labels}
        rules = {
            "kitchen": ["fridge", "oven", "microwave", "sink", "faucet", "dishwasher", "stove"],
            "bedroom": ["bed", "pillow", "blanket", "wardrobe", "nightstand"],
            "living_room": ["sofa", "couch", "tv", "coffee table", "armchair"],
            "bathroom": ["toilet", "bathtub", "shower"],
            "office": ["desk", "laptop", "monitor", "bookshelf"],
            "dining_room": ["dining table", "chair", "chandelier"],
            "laundry_room": ["washing machine"],
        }
        for room, keys in rules.items():
            if room not in scores:
                continue
            for key in keys:
                if key in joined:
                    scores[room] += 1
        best_room = max(scores, key=scores.get)
        return best_room if scores[best_room] > 0 else "unknown"

    def status(self) -> Dict[str, Any]:
        return {
            "module": "semantic",
            "backend": self.backend_name,
            "real_backend_active": self.is_real_backend,
            "model_name": self.model_name,
            "init_error": self.init_error,
        }
