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
