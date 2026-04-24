from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from common.types import FrameWorldState


@dataclass
class MemoryRecord:
    frame_id: int
    room: str
    caption: str
    objects: List[str]
    pose: Dict[str, float]


class EmbeddingProvider:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._dim = 384

    @property
    def dim(self) -> int:
        return self._dim

    def initialize(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            self._dim = int(self._model.get_sentence_embedding_dimension())
        except Exception:
            self._model = None
            self._dim = 384

    def encode(self, texts: List[str]) -> np.ndarray:
        if self._model is None:
            vecs = np.zeros((len(texts), self._dim), dtype=np.float32)
            for i, txt in enumerate(texts):
                vecs[i, abs(hash(txt)) % self._dim] = 1.0
            return vecs
        arr = self._model.encode(texts, normalize_embeddings=True)
        return np.asarray(arr, dtype=np.float32)


class FaissMemoryStore:
    def __init__(self, embedding_model: str, top_k: int = 5):
        self.embedder = EmbeddingProvider(embedding_model)
        self.top_k = top_k
        self.records: List[MemoryRecord] = []
        self.index = None

    def initialize(self) -> None:
        self.embedder.initialize()
        try:
            import faiss
            self.index = faiss.IndexFlatIP(self.embedder.dim)
        except Exception:
            self.index = None

    def add_state(self, state: FrameWorldState) -> None:
        rec = MemoryRecord(
            frame_id=state.frame_id,
            room=state.semantics.room_label,
            caption=state.semantics.caption,
            objects=[d.label for d in state.detections],
            pose=state.pose.__dict__,
        )
        self.records.append(rec)
        vec = self.embedder.encode([self._record_text(rec)])
        if self.index is not None:
            self.index.add(vec)

    def query(self, text: str, top_k: int | None = None) -> List[Dict[str, Any]]:
        k = top_k or self.top_k
        if not self.records:
            return []
        q = self.embedder.encode([text])

        if self.index is not None:
            scores, idxs = self.index.search(q, min(k, len(self.records)))
            out = []
            for score, idx in zip(scores[0], idxs[0]):
                if idx >= 0:
                    out.append({"score": float(score), "record": self.records[int(idx)].__dict__})
            return out

        mat = self.embedder.encode([self._record_text(r) for r in self.records])
        sims = (mat @ q[0]).tolist()
        order = np.argsort(sims)[::-1][:k]
        return [{"score": float(sims[i]), "record": self.records[int(i)].__dict__} for i in order]

    def export_metadata(self, out_path: Path) -> None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "size": len(self.records),
            "embedding_dim": self.embedder.dim,
            "records": [r.__dict__ for r in self.records],
        }, indent=2), encoding='utf-8')

    def _record_text(self, rec: MemoryRecord) -> str:
        objs = ', '.join(rec.objects) if rec.objects else 'no objects'
        return f"frame {rec.frame_id}; room={rec.room}; objects={objs}; caption={rec.caption}"

    def status(self) -> Dict[str, Any]:
        return {
            "module": "memory",
            "backend": "faiss" if self.index is not None else "numpy_fallback",
            "real_backend_active": self.index is not None,
            "embedding_model": self.embedder.model_name,
            "embedding_dim": self.embedder.dim,
        }
