from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Pose:
    x: float
    y: float
    z: float
    qx: float = 0.0
    qy: float = 0.0
    qz: float = 0.0
    qw: float = 1.0


@dataclass
class FramePacket:
    frame_id: int
    timestamp: float
    rgb_path: Optional[str]
    rgb: Any
    depth: Any = None
    intrinsics: Optional[Dict[str, float]] = None


@dataclass
class Detection:
    label: str
    score: float
    bbox_xyxy: Tuple[float, float, float, float]


@dataclass
class SegmentationMask:
    label: str
    score: float
    mask_rle: Dict[str, Any]
    bbox_xyxy: Tuple[float, float, float, float]


@dataclass
class SemanticFrame:
    room_label: str
    caption: str
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SlamFrameResult:
    pose: Pose
    keyframe_id: int
    map_update: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameWorldState:
    frame_id: int
    timestamp: float
    pose: Pose
    detections: List[Detection]
    segments: List[SegmentationMask]
    semantics: SemanticFrame


@dataclass
class SceneGraph:
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)
