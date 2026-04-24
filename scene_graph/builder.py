from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from common.types import FrameWorldState, SceneGraph


class SceneGraphBuilder:
    def __init__(self, cfg: Dict[str, Any]):
        self.include_room_nodes = bool(cfg.get("include_room_nodes", True))
        self.relation_threshold_px = float(cfg.get("relation_threshold_px", 60))

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

            edges.extend(self._spatial_edges(st))

        dedup_nodes = {n["id"]: n for n in nodes}
        return SceneGraph(nodes=list(dedup_nodes.values()), edges=edges, metadata={"frames": len(states), "rooms": len(room_to_objects)})

    def _spatial_edges(self, st: FrameWorldState) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        dets = st.detections
        for i in range(len(dets)):
            for j in range(i + 1, len(dets)):
                a, b = dets[i], dets[j]
                rel = self._relation(a.bbox_xyxy, b.bbox_xyxy)
                if not rel:
                    continue
                aid = f"frame{st.frame_id}::{a.label}::{int(a.bbox_xyxy[0])}_{int(a.bbox_xyxy[1])}"
                bid = f"frame{st.frame_id}::{b.label}::{int(b.bbox_xyxy[0])}_{int(b.bbox_xyxy[1])}"
                out.append({"source": aid, "target": bid, "relation": rel})
        return out

    def _relation(self, a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> str:
        ax = 0.5 * (a[0] + a[2])
        bx = 0.5 * (b[0] + b[2])
        if abs(ax - bx) < self.relation_threshold_px:
            return "overlapping_horizontal"
        return "left_of" if ax < bx else "right_of"
