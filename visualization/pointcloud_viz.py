from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

from common.types import FrameWorldState


def export_rtabmap_style_pointcloud(
    states: List[FrameWorldState],
    frames_dir: Path,
    output_png: Path,
    output_ply: Path | None = None,
    sample_step: int = 24,
) -> None:
    points_xyz: List[Tuple[float, float, float]] = []
    points_rgb: List[Tuple[int, int, int]] = []

    for st in states:
        frame_path = frames_dir / f"frame_{st.frame_id:06d}.jpg"
        if not frame_path.exists():
            continue
        image_bgr = cv2.imread(str(frame_path))
        if image_bgr is None:
            continue

        h, w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        for v in range(0, h, sample_step):
            for u in range(0, w, sample_step):
                color = image_rgb[v, u]
                intensity = float(np.mean(color)) / 255.0

                # Pseudo 3D lift for RGB-only video:
                # - pose anchors global motion,
                # - image coordinates spread local neighborhood,
                # - brightness and row index emulate vertical structure.
                local_x = (u / max(1, w - 1) - 0.5) * 4.0
                local_y = (v / max(1, h - 1) - 0.5) * 3.0
                local_z = (1.0 - v / max(1, h - 1)) * 2.0 + 0.8 * intensity

                world_x = st.pose.x * 5.0 + local_x
                world_y = st.pose.y * 5.0 + local_y
                world_z = st.pose.z + local_z

                points_xyz.append((world_x, world_y, world_z))
                points_rgb.append((int(color[0]), int(color[1]), int(color[2])))

    if not points_xyz:
        return

    xyz = np.asarray(points_xyz, dtype=np.float32)
    rgb = np.asarray(points_rgb, dtype=np.uint8)

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 7))
    plt.style.use("dark_background")
    plt.scatter(
        xyz[:, 0],
        xyz[:, 1],
        c=xyz[:, 2],
        s=0.45,
        cmap="autumn",
        alpha=0.8,
        linewidths=0,
    )
    plt.title("RTAB-Map Style 3D Cloud (Top View)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_png, dpi=220)
    plt.close()

    if output_ply is not None:
        _write_ascii_ply(output_ply, xyz, rgb)


def _write_ascii_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {xyz.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(xyz, rgb):
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")


def export_pointcloud_simulator_previews(
    cloud_ply: Path,
    output_realistic_png: Path,
    output_heatmap_png: Path,
    point_budget: int = 250000,
) -> None:
    xyz, rgb = _read_ascii_ply_xyzrgb(cloud_ply)
    if xyz.shape[0] == 0:
        raise RuntimeError(f"No points loaded from cloud: {cloud_ply}")

    if xyz.shape[0] > point_budget:
        step = max(1, xyz.shape[0] // point_budget)
        xyz = xyz[::step]
        rgb = rgb[::step]

    # Normalize for stable projection.
    center = xyz.mean(axis=0, keepdims=True)
    scale = float(np.max(np.linalg.norm(xyz - center, axis=1)))
    if scale < 1e-6:
        scale = 1.0
    xyz_n = (xyz - center) / scale

    width, height = 1400, 820
    proj = _project_points_perspective(xyz_n, yaw=np.deg2rad(40.0), pitch=np.deg2rad(26.0), width=width, height=height)
    if proj.shape[0] == 0:
        raise RuntimeError("Point projection is empty for simulator preview rendering.")

    # Sort far-to-near so near points stay visible.
    order = np.argsort(proj[:, 2])
    proj = proj[order]
    rgb = rgb[order]

    output_realistic_png.parent.mkdir(parents=True, exist_ok=True)
    output_heatmap_png.parent.mkdir(parents=True, exist_ok=True)

    realistic = np.full((height, width, 3), 245, dtype=np.uint8)
    heatmap = np.full((height, width, 3), 28, dtype=np.uint8)

    depth = proj[:, 2]
    dmin, dmax = float(depth.min()), float(depth.max())
    drange = max(dmax - dmin, 1e-6)
    depth_norm = ((depth - dmin) / drange).clip(0.0, 1.0)

    cmap = plt.get_cmap("inferno")
    heat_colors = (cmap(depth_norm)[:, :3] * 255.0).astype(np.uint8)

    for i in range(proj.shape[0]):
        x = int(proj[i, 0])
        y = int(proj[i, 1])
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        r, g, b = int(rgb[i, 0]), int(rgb[i, 1]), int(rgb[i, 2])
        shade = int(90 + 120 * (1.0 - depth_norm[i]))
        realistic_color = (min(255, b * shade // 200), min(255, g * shade // 200), min(255, r * shade // 200))
        cv2.circle(realistic, (x, y), 1, realistic_color, -1, cv2.LINE_AA)

        hr, hg, hb = int(heat_colors[i, 0]), int(heat_colors[i, 1]), int(heat_colors[i, 2])
        cv2.circle(heatmap, (x, y), 1, (hb, hg, hr), -1, cv2.LINE_AA)

    cv2.putText(realistic, "Simulator 3D View (Realistic)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (40, 40, 40), 2, cv2.LINE_AA)
    cv2.putText(heatmap, "Simulator 3D View (Heatmap)", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 230, 230), 2, cv2.LINE_AA)

    cv2.imwrite(str(output_realistic_png), realistic)
    cv2.imwrite(str(output_heatmap_png), heatmap)


def _read_ascii_ply_xyzrgb(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    xyz: List[Tuple[float, float, float]] = []
    rgb: List[Tuple[int, int, int]] = []
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
                x, y, z = float(parts[0]), float(parts[1]), float(parts[2])
            except ValueError:
                continue
            if not np.isfinite([x, y, z]).all():
                continue
            xyz.append((x, y, z))
            if len(parts) >= 6:
                try:
                    r, g, b = int(parts[3]), int(parts[4]), int(parts[5])
                    rgb.append((np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)))
                except ValueError:
                    rgb.append((230, 230, 230))
            else:
                rgb.append((230, 230, 230))

    if not xyz:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
    return np.asarray(xyz, dtype=np.float32), np.asarray(rgb, dtype=np.uint8)


def _project_points_perspective(
    pts: np.ndarray,
    yaw: float,
    pitch: float,
    width: int,
    height: int,
    focal: float = 860.0,
) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    r = rx @ ry

    cam = (pts @ r.T).astype(np.float32)
    cam[:, 2] += 2.8
    valid = cam[:, 2] > 0.15
    cam = cam[valid]
    if cam.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    out = np.zeros((cam.shape[0], 3), dtype=np.float32)
    out[:, 0] = width * 0.5 + focal * (cam[:, 0] / cam[:, 2])
    out[:, 1] = height * 0.5 - focal * (cam[:, 1] / cam[:, 2])
    out[:, 2] = cam[:, 2]
    return out
