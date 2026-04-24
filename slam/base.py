from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict

from common.types import FramePacket, SlamFrameResult


class SlamBackend(ABC):
    @abstractmethod
    def initialize(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def process_frame(self, packet: FramePacket) -> SlamFrameResult:
        raise NotImplementedError

    @abstractmethod
    def finalize(self) -> Dict:
        raise NotImplementedError
