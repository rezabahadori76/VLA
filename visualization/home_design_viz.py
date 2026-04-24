from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from common.types import FrameWorldState


def export_home_design(
    states: List[FrameWorldState],
    output_png: Path,
    output_json: Path,
    canvas_size: int = 1200,
) -> None:
    if not states:
        raise RuntimeError("No states available for home design export.")

    xy = np.asarray([[float(s.pose.x), float(s.pose.z)] for s in states], dtype=np.float32)
    mn = xy.min(axis=0)
    mx = xy.max(axis=0)
    span = np.maximum(mx - mn, 1e-3)

    margin = 80
    width = canvas_size
    height = canvas_size
    canvas = np.full((height, width, 3), 250, dtype=np.uint8)

    # Room footprints: derive from pose distribution per room label.
    room_points: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    for st in states:
        room = str(st.semantics.room_label or "unknown")
        room_points[room].append((float(st.pose.x), float(st.pose.z)))

    def to_px(x: float, z: float) -> Tuple[int, int]:
        nx = (x - mn[0]) / span[0]
        nz = (z - mn[1]) / span[1]
        px = int(margin + nx * (width - 2 * margin))
        py = int(height - (margin + nz * (height - 2 * margin)))
        return px, py

    room_colors = _room_palette(list(room_points.keys()))
    room_boxes: Dict[str, Dict[str, int]] = {}
    for room, pts in room_points.items():
        arr = np.asarray(pts, dtype=np.float32)
        rmn = arr.min(axis=0)
        rmx = arr.max(axis=0)
        pad_x = max((rmx[0] - rmn[0]) * 0.25, span[0] * 0.03)
        pad_z = max((rmx[1] - rmn[1]) * 0.25, span[1] * 0.03)
        x1, y1 = to_px(rmn[0] - pad_x, rmn[1] - pad_z)
        x2, y2 = to_px(rmx[0] + pad_x, rmx[1] + pad_z)
        left, right = min(x1, x2), max(x1, x2)
        top, bottom = min(y1, y2), max(y1, y2)
        color = room_colors[room]
        cv2.rectangle(canvas, (left, top), (right, bottom), color, thickness=2)
        fill = canvas[top:bottom, left:right]
        if fill.size > 0:
            tint = np.full_like(fill, color, dtype=np.uint8)
            canvas[top:bottom, left:right] = cv2.addWeighted(fill, 0.9, tint, 0.1, 0)
        cv2.putText(
            canvas,
            room,
            (left + 6, max(24, top + 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
        room_boxes[room] = {"left": left, "top": top, "right": right, "bottom": bottom}

    # Furniture/object placement via trajectory + bbox-center heuristic.
    obj_pts: Dict[str, List[Tuple[float, float, str]]] = defaultdict(list)
    for st in states:
        room = str(st.semantics.room_label or "unknown")
        for det in st.detections:
            x1, y1, x2, y2 = map(float, det.bbox_xyxy)
            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            # Assume source frame width/height scale from common mobile footage proportions.
            nx = (cx / 480.0) - 0.5
            ny = (cy / 270.0) - 0.5
            wx = float(st.pose.x) + nx * max(span[0] * 0.2, 0.15)
            wz = float(st.pose.z) + ny * max(span[1] * 0.2, 0.15)
            obj_pts[det.label].append((wx, wz, room))

    object_layout = []
    for label, pts in obj_pts.items():
        arr = np.asarray([[p[0], p[1]] for p in pts], dtype=np.float32)
        room_votes = defaultdict(int)
        for _, _, room in pts:
            room_votes[room] += 1
        room = max(room_votes, key=room_votes.get)
        ox, oz = float(arr[:, 0].mean()), float(arr[:, 1].mean())
        px, py = to_px(ox, oz)
        cv2.circle(canvas, (px, py), 3, (30, 30, 30), -1, cv2.LINE_AA)
        cv2.putText(
            canvas,
            label[:24],
            (px + 5, py - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
        object_layout.append({"label": label, "x": ox, "z": oz, "room": room, "observations": len(pts)})

    # Draw full path.
    traj_px = [to_px(float(s.pose.x), float(s.pose.z)) for s in states]
    for i in range(1, len(traj_px)):
        cv2.line(canvas, traj_px[i - 1], traj_px[i], (0, 80, 220), 2, cv2.LINE_AA)
    if traj_px:
        cv2.circle(canvas, traj_px[0], 5, (0, 180, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, traj_px[-1], 5, (0, 0, 220), -1, cv2.LINE_AA)

    cv2.putText(canvas, "Home Design Layout (rooms + furniture placement)", (24, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (15, 15, 15), 2, cv2.LINE_AA)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_png), canvas)

    payload = {
        "rooms": room_boxes,
        "objects": object_layout,
        "trajectory_points": [{"x": float(s.pose.x), "z": float(s.pose.z), "room": str(s.semantics.room_label)} for s in states],
        "summary": {
            "frames": len(states),
            "rooms_count": len(room_boxes),
            "objects_count": len(object_layout),
        },
    }
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _room_palette(rooms: List[str]) -> Dict[str, Tuple[int, int, int]]:
    base = [
        (70, 130, 180),
        (46, 139, 87),
        (178, 34, 34),
        (218, 112, 214),
        (255, 140, 0),
        (106, 90, 205),
        (47, 79, 79),
        (255, 99, 71),
        (60, 179, 113),
        (199, 21, 133),
    ]
    out: Dict[str, Tuple[int, int, int]] = {}
    for i, room in enumerate(sorted(rooms)):
        out[room] = base[i % len(base)]
    return out
