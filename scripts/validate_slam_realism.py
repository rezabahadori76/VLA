from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise RuntimeError(f"Missing required file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _trajectory_checks(trajectory: List[dict]) -> None:
    if len(trajectory) < 3:
        raise RuntimeError("Trajectory has fewer than 3 poses, not enough for SLAM validation.")

    xyz = np.asarray([[p["x"], p["y"], p["z"]] for p in trajectory], dtype=np.float64)
    steps = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    if np.allclose(steps, 0.0):
        raise RuntimeError("Trajectory has zero motion for all steps.")
    if np.std(steps) < 1e-5:
        raise RuntimeError("Trajectory step sizes are nearly constant (likely synthetic).")


def _pose_graph_checks(pose_graph: dict) -> None:
    nodes = pose_graph.get("nodes", [])
    edges = pose_graph.get("edges", [])
    if not nodes or not edges:
        raise RuntimeError("Pose graph is empty.")

    odom_edges = [e for e in edges if e.get("type") == "odometry"]
    loop_edges = [e for e in edges if e.get("type") not in ("odometry", None)]
    if not odom_edges:
        raise RuntimeError("No odometry edges in pose graph.")
    if not loop_edges:
        print("[warn] No explicit loop-closure edges found in pose graph export.")


def _map_checks(occupancy_npy: Path, pointcloud_ply: Path) -> None:
    if not occupancy_npy.exists():
        raise RuntimeError(f"Missing occupancy grid: {occupancy_npy}")
    if not pointcloud_ply.exists():
        raise RuntimeError(f"Missing point cloud: {pointcloud_ply}")

    grid = np.load(occupancy_npy)
    if grid.size == 0:
        raise RuntimeError("Occupancy grid is empty.")
    nonzero = int(np.count_nonzero(grid))
    if nonzero <= 10:
        raise RuntimeError(f"Occupancy grid has too few occupied cells ({nonzero}).")

    header = pointcloud_ply.read_text(encoding="utf-8", errors="ignore").splitlines()[:20]
    vertex_line = next((ln for ln in header if ln.startswith("element vertex ")), "")
    if not vertex_line:
        raise RuntimeError("Point cloud PLY has no vertex declaration.")
    vertex_count = int(vertex_line.split()[-1])
    if vertex_count <= 0:
        raise RuntimeError("Point cloud has zero vertices.")


def _backend_checks(backend_status: dict) -> None:
    slam = backend_status.get("slam", {})
    if slam.get("backend") != "rtabmap":
        raise RuntimeError(f"Unexpected SLAM backend: {slam.get('backend')}")
    if not slam.get("real_backend_active", False):
        raise RuntimeError("SLAM backend reports non-real execution.")


def _source_checks(project_root: Path) -> None:
    bad_tokens = ["FallbackPseudoSlam", "x=prev.x + 0.04", "source\": \"fallback\""]
    files = list((project_root / "slam").glob("*.py"))
    for fp in files:
        text = fp.read_text(encoding="utf-8")
        for token in bad_tokens:
            if token in text:
                raise RuntimeError(f"Synthetic SLAM token detected in {fp}: {token}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate that SLAM outputs are real and non-synthetic.")
    parser.add_argument("--output", type=Path, required=True, help="Pipeline output run directory.")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()

    output = args.output
    backend_status = _load_json(output / "logs" / "backend_status.json")
    trajectory = _load_json(output / "map" / "trajectory.json").get("trajectory", [])
    pose_graph = _load_json(output / "map" / "pose_graph.json")

    _backend_checks(backend_status)
    _trajectory_checks(trajectory)
    _pose_graph_checks(pose_graph)
    _map_checks(output / "map" / "occupancy_grid.npy", output / "map" / "pointcloud_rgb.ply")
    _source_checks(args.project_root)

    print("SLAM realism validation PASSED.")


if __name__ == "__main__":
    main()
