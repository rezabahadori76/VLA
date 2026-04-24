from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from common.types import FrameWorldState, SceneGraph


class SceneGraphBuilder:
    def __init__(self, cfg: Dict[str, Any]):
        self.include_room_nodes = bool(cfg.get("include_room_nodes", True))

    def build(self, states: List[FrameWorldState]) -> SceneGraph:
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []
        room_to_objects: Dict[str, List[str]] = defaultdict(list)

        for st in states:
            room_id = f"room::{st.semantics.room_label}"
            if self.include_room_nodes:
                nodes.append({"id": room_id, "type": "room", "label": st.semantics.room_label})

            for det in st.detections:
                obj_id = f"frame{st.frame_id}::{det.label}::{int(det.bbox_xyxy[0])}_{int(det.bbox_xyxy[1])}"
                nodes.append({"id": obj_id, "type": "object", "label": det.label, "bbox": list(det.bbox_xyxy), "frame": st.frame_id})
                edges.append({"source": room_id, "target": obj_id, "relation": "contains"})
                room_to_objects[room_id].append(obj_id)

            edges.extend(self._spatial_feature_edges(st))

        dedup_nodes = {n["id"]: n for n in nodes}
        return SceneGraph(nodes=list(dedup_nodes.values()), edges=edges, metadata={"frames": len(states), "rooms": len(room_to_objects)})

    def _spatial_feature_edges(self, st: FrameWorldState) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        dets = st.detections
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                a, b = dets[i], dets[j]
                aid = f"frame{st.frame_id}::{a.label}::{int(a.bbox_xyxy[0])}_{int(a.bbox_xyxy[1])}"
                bid = f"frame{st.frame_id}::{b.label}::{int(b.bbox_xyxy[0])}_{int(b.bbox_xyxy[1])}"
                out.append({"source": aid, "target": bid, "relation": "spatial_features", "features": self._pair_features(a.bbox_xyxy, b.bbox_xyxy)})
        return out

    def _pair_features(self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> Dict[str, float]:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        acx = 0.5 * (ax1 + ax2)
        acy = 0.5 * (ay1 + ay2)
        bcx = 0.5 * (bx1 + bx2)
        bcy = 0.5 * (by1 + by2)
        aw = max(ax2 - ax1, 1e-6)
        ah = max(ay2 - ay1, 1e-6)
        bw = max(bx2 - bx1, 1e-6)
        bh = max(by2 - by1, 1e-6)
        inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = inter_w * inter_h
        union = aw * ah + bw * bh - inter
        iou = inter / union if union > 0.0 else 0.0
        return {
            "delta_cx_norm": (bcx - acx) / max(aw, bw),
            "delta_cy_norm": (bcy - acy) / max(ah, bh),
            "iou": iou,
            "area_ratio_ab": (aw * ah) / (bw * bh),
            "area_ratio_ba": (bw * bh) / (aw * ah),
        }
