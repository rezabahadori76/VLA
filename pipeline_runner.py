from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.io_utils import ensure_output_dirs, load_config
from pipeline.orchestrator import HomeWorldModelPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Home World Model Builder from Video - Phase 1')
    parser.add_argument('--video', required=True, type=Path, help='Input RGB video path')
    parser.add_argument('--depth-video', type=Path, default=None, help='Optional depth video path')
    parser.add_argument('--config', required=True, type=Path, help='YAML config path')
    parser.add_argument('--output', required=True, type=Path, help='Output directory')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_dirs = ensure_output_dirs(args.output)

    pipeline = HomeWorldModelPipeline(cfg=cfg, output_dirs=output_dirs)
    pipeline.initialize()
    summary = pipeline.run(video_path=args.video, depth_video_path=args.depth_video)

    print('Phase 1 pipeline complete.')
    print(summary)


if __name__ == '__main__':
    main()
