from __future__ import annotations

from pathlib import Path
from typing import List

import cv2
import numpy as np

from common.types import FrameWorldState


def _rle_to_mask(rle: dict) -> np.ndarray:
    size = rle['size']
    counts = rle['counts']
    flat = []
    val = 0
    for c in counts:
        flat.extend([val] * int(c))
        val = 1 - val
    return np.array(flat, dtype=np.uint8).reshape((size[0], size[1]), order='F')


def render_perception_overlay(
    states: List[FrameWorldState],
    frames_dir: Path,
    output_dir: Path,
    draw_masks: bool,
    draw_boxes: bool,
    box_line_thickness: int = 1,
    label_font_thickness: int = 1,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for st in states:
        frame_path = frames_dir / f'frame_{st.frame_id:06d}.jpg'
        if not frame_path.exists():
            continue
        img = cv2.imread(str(frame_path))
        if img is None:
            continue

        if draw_masks:
            for seg in st.segments:
                mask = _rle_to_mask(seg.mask_rle)
                color = np.array([0, 255, 0], dtype=np.uint8)
                img[mask == 1] = (0.6 * img[mask == 1] + 0.4 * color).astype(np.uint8)

        if draw_boxes:
            t = max(1, int(box_line_thickness))
            lt = max(1, int(label_font_thickness))
            for det in st.detections:
                x1, y1, x2, y2 = map(int, det.bbox_xyxy)
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 120, 0), t)
                cv2.putText(
                    img,
                    f'{det.label}:{det.score:.2f}',
                    (x1, max(12, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 120, 0),
                    lt,
                )

        cv2.putText(img, f'room={st.semantics.room_label}', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30, 220, 255), 2)
        cv2.imwrite(str(output_dir / f'overlay_{st.frame_id:06d}.jpg'), img)
