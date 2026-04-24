from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import cv2
import yaml

from common.types import FramePacket


def load_config(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_output_dirs(base: Path) -> Dict[str, Path]:
    dirs = {
        'map': base / 'map',
        'detections': base / 'detections',
        'segments': base / 'segments',
        'semantics': base / 'semantics',
        'scene_graph': base / 'scene_graph',
        'memory': base / 'memory',
        'logs': base / 'logs',
        'viz': base / 'viz',
        'frames': base / 'frames',
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def iter_video_frames(
    video_path: Path,
    depth_video_path: Optional[Path],
    frame_stride: int,
    max_frames: int,
    save_rgb_dir: Path,
    intrinsics: Optional[Dict[str, float]] = None,
) -> Generator[FramePacket, None, None]:
    cap = cv2.VideoCapture(str(video_path))
    depth_cap = cv2.VideoCapture(str(depth_video_path)) if depth_video_path else None

    frame_id = 0
    emitted = 0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        depth_frame = None
        if depth_cap is not None:
            ok_d, depth_frame = depth_cap.read()
            if not ok_d:
                depth_frame = None

        if frame_id % frame_stride == 0:
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            rgb_path = save_rgb_dir / f'frame_{emitted:06d}.jpg'
            cv2.imwrite(str(rgb_path), frame)
            yield FramePacket(
                frame_id=emitted,
                timestamp=t,
                rgb_path=str(rgb_path),
                rgb=frame,
                depth=depth_frame,
                intrinsics=intrinsics,
            )
            emitted += 1
            if emitted >= max_frames:
                break

        frame_id += 1

    cap.release()
    if depth_cap is not None:
        depth_cap.release()


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
