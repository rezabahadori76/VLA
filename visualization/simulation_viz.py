from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


def export_overlay_video(overlays_dir: Path, output_mp4: Path, fps: int = 30) -> None:
    frames = sorted(overlays_dir.glob("overlay_*.jpg"))
    if not frames:
        raise RuntimeError(f"No overlay frames found in {overlays_dir}")

    first = cv2.imread(str(frames[0]))
    if first is None:
        raise RuntimeError(f"Failed to read overlay frame: {frames[0]}")

    h, w = first.shape[:2]
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_mp4}")

    for fp in frames:
        frame = cv2.imread(str(fp))
        if frame is None:
            continue
        writer.write(frame)
    writer.release()


def export_2d_simulation_video(
    occupancy_grid_npy: Path,
    trajectory_json: Path,
    output_mp4: Path,
    fps: int = 20,
) -> None:
    grid = np.load(occupancy_grid_npy)
    if grid.ndim != 2:
        raise RuntimeError(f"Expected 2D occupancy grid, got shape {grid.shape}")

    traj = _load_trajectory_points(trajectory_json)
    if len(traj) < 2:
        raise RuntimeError("Trajectory has fewer than 2 poses.")

    canvas = _grid_to_canvas(grid)
    h, w = canvas.shape[:2]
    mapped = _map_xy_to_pixels(traj, w, h)

    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_mp4}")

    for idx in range(len(mapped)):
        frame = canvas.copy()
        # Draw full trajectory in faint color for global context.
        for i in range(1, len(mapped)):
            cv2.line(frame, mapped[i - 1], mapped[i], (80, 80, 180), 1, cv2.LINE_AA)
        # Draw traversed trajectory in strong color.
        for i in range(1, idx + 1):
            cv2.line(frame, mapped[i - 1], mapped[i], (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, mapped[idx], 4, (0, 255, 0), -1, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"2D Simulation | frame {idx+1}/{len(mapped)}",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()


def export_3d_simulation_video(
    cloud_ply: Path,
    trajectory_json: Path,
    output_mp4: Path,
    fps: int = 20,
    max_points: int = 18000,
) -> None:
    xyz = _read_ascii_ply_xyz(cloud_ply)
    if xyz.shape[0] == 0:
        raise RuntimeError(f"No points loaded from cloud: {cloud_ply}")
    if xyz.shape[0] > max_points:
        step = max(1, xyz.shape[0] // max_points)
        xyz = xyz[::step]

    traj = _load_trajectory_points(trajectory_json)
    if len(traj) < 2:
        raise RuntimeError("Trajectory has fewer than 2 poses.")

    width, height = 960, 720
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_mp4), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create video writer: {output_mp4}")

    # Normalize world scale for stable projection.
    all_pts = np.vstack([xyz, traj])
    center = all_pts.mean(axis=0, keepdims=True)
    scale = float(np.max(np.linalg.norm(all_pts - center, axis=1)))
    if scale < 1e-6:
        scale = 1.0

    xyz_n = (xyz - center) / scale
    traj_n = (traj - center) / scale

    for idx in range(len(traj_n)):
        # Orbit camera around scene while advancing robot marker.
        yaw = (idx / max(1, len(traj_n) - 1)) * 2.0 * np.pi
        pitch = np.deg2rad(25.0)
        frame = np.full((height, width, 3), 22, dtype=np.uint8)

        proj_cloud = _project_points(xyz_n, yaw=yaw, pitch=pitch, width=width, height=height, focal=760.0)
        for x, y, z in proj_cloud:
            if 0 <= x < width and 0 <= y < height:
                shade = int(np.clip(255 - (z + 1.5) * 55, 80, 255))
                color = (shade // 2, shade // 2, shade)
                cv2.circle(frame, (x, y), 1, color, -1, cv2.LINE_AA)

        past = traj_n[: idx + 1]
        proj_traj = _project_points(past, yaw=yaw, pitch=pitch, width=width, height=height, focal=760.0)
        for i in range(1, len(proj_traj)):
            x1, y1, _ = proj_traj[i - 1]
            x2, y2, _ = proj_traj[i]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2, cv2.LINE_AA)
        if len(proj_traj) > 0:
            cx, cy, _ = proj_traj[-1]
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1, cv2.LINE_AA)

        cv2.putText(
            frame,
            f"3D Simulation | frame {idx+1}/{len(traj_n)}",
            (14, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
    writer.release()


def _load_trajectory_points(path: Path) -> np.ndarray:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("trajectory", [])
    pts: List[Tuple[float, float, float]] = []
    for p in raw:
        pts.append((float(p["x"]), float(p["y"]), float(p["z"])))
    return np.asarray(pts, dtype=np.float32)


def _grid_to_canvas(grid: np.ndarray) -> np.ndarray:
    g = grid.astype(np.float32)
    gmin, gmax = float(g.min()), float(g.max())
    if gmax - gmin < 1e-6:
        norm = np.full_like(g, 180, dtype=np.uint8)
    else:
        norm = ((g - gmin) / (gmax - gmin) * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(norm, cv2.COLORMAP_BONE)


def _map_xy_to_pixels(traj_xyz: np.ndarray, width: int, height: int) -> List[Tuple[int, int]]:
    xy = traj_xyz[:, [0, 2]]
    mn = xy.min(axis=0)
    mx = xy.max(axis=0)
    rng = np.maximum(mx - mn, 1e-6)
    out: List[Tuple[int, int]] = []
    for p in xy:
        x = int((p[0] - mn[0]) / rng[0] * (width - 1))
        y = int((p[1] - mn[1]) / rng[1] * (height - 1))
        out.append((x, height - 1 - y))
    return out


def _read_ascii_ply_xyz(path: Path) -> np.ndarray:
    pts: List[Tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8") as f:
        in_data = False
        for raw in f:
            line = raw.strip()
            if not in_data:
                if line == "end_header":
                    in_data = True
                continue
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                pts.append((float(parts[0]), float(parts[1]), float(parts[2])))
            except ValueError:
                continue
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    arr = np.asarray(pts, dtype=np.float32)
    return arr[np.isfinite(arr).all(axis=1)]


def _project_points(
    pts: np.ndarray,
    yaw: float,
    pitch: float,
    width: int,
    height: int,
    focal: float,
) -> List[Tuple[int, int, float]]:
    # Rotation: yaw around Y, pitch around X.
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    r = rx @ ry

    cam = (pts @ r.T).astype(np.float32)
    cam[:, 2] += 2.6  # move scene in front of camera
    valid = cam[:, 2] > 0.2
    cam = cam[valid]
    out: List[Tuple[int, int, float]] = []
    for x, y, z in cam:
        u = int(width * 0.5 + focal * (x / z))
        v = int(height * 0.5 - focal * (y / z))
        out.append((u, v, float(z)))
    return out
