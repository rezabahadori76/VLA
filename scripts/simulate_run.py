from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from visualization.simulation_viz import export_2d_simulation_video, export_3d_simulation_video


def _export_frame_replay_video(frames_dir: Path, output_mp4: Path, fps: int) -> None:
    frames = sorted(frames_dir.glob("frame_*.jpg"))
    if not frames:
        raise RuntimeError(f"No frame_*.jpg files found in {frames_dir}")

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Failed reading first frame: {frames[0]}")

    h, w = first.shape[:2]
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_mp4}")

    total = len(frames)
    for idx, frame_path in enumerate(frames, start=1):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue
        cv2.putText(
            frame,
            f"Replay | frame {idx}/{total}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (240, 240, 240),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build simulator videos from a VLA run output directory.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Path like outputs/run_gpu_design_23s",
    )
    parser.add_argument("--fps", type=int, default=20, help="Output FPS for simulator videos")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    viz_dir = run_dir / "viz"
    map_dir = run_dir / "map"
    frames_dir = run_dir / "frames"

    trajectory_json = map_dir / "trajectory.json"
    occupancy_grid_npy = map_dir / "occupancy_grid.npy"
    cloud_ply = map_dir / "pointcloud_rgb.ply"

    if not trajectory_json.exists():
        raise FileNotFoundError(f"Missing trajectory file: {trajectory_json}")
    if not occupancy_grid_npy.exists():
        raise FileNotFoundError(f"Missing occupancy grid file: {occupancy_grid_npy}")
    if not cloud_ply.exists():
        raise FileNotFoundError(f"Missing point cloud file: {cloud_ply}")
    if not frames_dir.exists():
        raise FileNotFoundError(f"Missing frames dir: {frames_dir}")

    replay_out = viz_dir / "sim_replay.mp4"
    sim2d_out = viz_dir / "sim_2d.mp4"
    sim3d_out = viz_dir / "sim_3d.mp4"

    _export_frame_replay_video(frames_dir=frames_dir, output_mp4=replay_out, fps=args.fps)
    export_2d_simulation_video(
        occupancy_grid_npy=occupancy_grid_npy,
        trajectory_json=trajectory_json,
        output_mp4=sim2d_out,
        fps=args.fps,
    )
    export_3d_simulation_video(
        cloud_ply=cloud_ply,
        trajectory_json=trajectory_json,
        output_mp4=sim3d_out,
        fps=args.fps,
        semantics_dir=run_dir / "semantics",
    )

    print("Simulator assets generated:")
    print(f"- {replay_out}")
    print(f"- {sim2d_out}")
    print(f"- {sim3d_out}")


if __name__ == "__main__":
    main()
