from __future__ import annotations

from typing import Any, Dict

from common.types import FramePacket, SlamFrameResult
from slam.base import SlamBackend
from slam.orbslam3_adapter import ORBSLAM3Backend
from slam.rtabmap_adapter import RTABMapBackend


class SlamModule:
    def __init__(self, cfg: Dict[str, Any]):
        backend = cfg.get("backend", "rtabmap").lower()
        resolution = float(cfg.get("map_resolution", 0.05))
        if backend == "rtabmap":
            rcfg = cfg.get("rtabmap", {})
            self.backend: SlamBackend = RTABMapBackend(
                executable=rcfg.get("executable", "rtabmap-rgbd_dataset"),
                db_path=rcfg.get("db_path", "outputs/rtabmap.db"),
                resolution=resolution,
                camera_intrinsics=cfg.get("camera_intrinsics", {}),
                require_depth=bool(cfg.get("require_depth", True)),
            )
        elif backend == "orbslam3":
            ocfg = cfg.get("orbslam3", {})
            self.backend = ORBSLAM3Backend(
                executable=ocfg.get("executable", "./ORB_SLAM3/mono_tum"),
                vocabulary_path=ocfg.get("vocabulary_path", "./ORB_SLAM3/Vocabulary/ORBvoc.txt"),
                settings_path=ocfg.get("settings_path", "./ORB_SLAM3/Examples/Monocular/TUM1.yaml"),
                resolution=resolution,
            )
        else:
            raise ValueError(f"Unsupported SLAM backend: {backend}")

    def initialize(self) -> None:
        self.backend.initialize()

    def process_frame(self, packet: FramePacket) -> SlamFrameResult:
        return self.backend.process_frame(packet)

    def finalize(self) -> Dict[str, Any]:
        return self.backend.finalize()

    def status(self) -> Dict[str, Any]:
        if hasattr(self.backend, "status"):
            return self.backend.status()
        return {"module": "slam", "backend": "unknown", "real_backend_active": False}
