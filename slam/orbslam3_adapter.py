from __future__ import annotations

from typing import Dict

from common.types import FramePacket, SlamFrameResult
from slam.base import SlamBackend


class ORBSLAM3Backend(SlamBackend):
    def __init__(self, executable: str, vocabulary_path: str, settings_path: str, resolution: float):
        self.executable = executable
        self.vocabulary_path = vocabulary_path
        self.settings_path = settings_path
        self.is_real_backend = False
        self.init_error: str | None = None

    def initialize(self) -> None:
        self.init_error = (
            "ORB-SLAM3 backend is not wired in this build. "
            "Use slam.backend=rtabmap or implement ORB-SLAM3 runtime integration."
        )
        raise RuntimeError(self.init_error)

    def process_frame(self, packet: FramePacket) -> SlamFrameResult:
        raise RuntimeError("ORB-SLAM3 process_frame cannot run: backend is not initialized.")

    def finalize(self) -> Dict:
        raise RuntimeError("ORB-SLAM3 finalize cannot run: backend is not initialized.")

    def status(self) -> Dict:
        return {
            "module": "slam",
            "backend": "orbslam3",
            "real_backend_active": self.is_real_backend,
            "executable": self.executable,
            "vocabulary_path": self.vocabulary_path,
            "settings_path": self.settings_path,
            "init_error": self.init_error,
        }
