from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detection.grounding_dino_detector import GroundingDinoDetector
from memory.vector_store import FaissMemoryStore
from pipeline.io_utils import load_config
from segmentation.sam_segmenter import SAMSegmenter
from semantic.qwen2_vl_understanding import Qwen2VLSemanticModule
from slam.slam_module import SlamModule


def main() -> None:
    parser = argparse.ArgumentParser(description="Check which backends are active.")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)

    slam = SlamModule(cfg["slam"])
    det = GroundingDinoDetector(cfg["detection"])
    seg = SAMSegmenter(cfg["segmentation"])
    sem = Qwen2VLSemanticModule(cfg["semantic"])
    mem = FaissMemoryStore(cfg["memory"]["embedding_model"], int(cfg["memory"].get("top_k", 5)))

    slam.initialize()
    det.initialize()
    seg.initialize()
    sem.initialize()
    mem.initialize()

    print(slam.status())
    print(det.status())
    print(seg.status())
    print(sem.status())
    print(mem.status())


if __name__ == "__main__":
    main()
