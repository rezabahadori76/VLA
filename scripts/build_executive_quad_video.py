#!/usr/bin/env python3
"""Build a single 2x2 dashboard video: original | overlay / home layout | 3D sim."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def letterbox_bgr(img: np.ndarray, w: int, h: int, color: tuple[int, int, int] = (16, 16, 20)) -> np.ndarray:
    if img is None or img.size == 0:
        return np.full((h, w, 3), color, dtype=np.uint8)
    ih, iw = img.shape[:2]
    if iw < 1 or ih < 1:
        return np.full((h, w, 3), color, dtype=np.uint8)
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    out = np.full((h, w, 3), color, dtype=np.uint8)
    x0, y0 = (w - nw) // 2, (h - nh) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def draw_label(img: np.ndarray, text: str) -> None:
    x, y = 12, 28
    cv2.putText(img, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 2, cv2.LINE_AA)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2x2 executive dashboard video (original / overlay / home / 3D).")
    p.add_argument(
        "--original",
        type=Path,
        default=Path("/workspace/VLA/data/video.mp4"),
        help="Source RGB video",
    )
    p.add_argument(
        "--overlay",
        type=Path,
        required=True,
        help="Overlay preview mp4",
    )
    p.add_argument(
        "--home-png",
        type=Path,
        required=True,
        help="home_design_layout.png",
    )
    p.add_argument(
        "--sim-3d",
        type=Path,
        required=True,
        help="simulation_3d.mp4",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output mp4 path",
    )
    p.add_argument("--cell-w", type=int, default=960, help="Width of each quadrant")
    p.add_argument("--cell-h", type=int, default=540, help="Height of each quadrant")
    p.add_argument("--fps", type=float, default=30.0, help="Output FPS")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cw, ch = int(args.cell_w), int(args.cell_h)
    out_w, out_h = cw * 2, ch * 2

    cap_o = cv2.VideoCapture(str(args.original))
    cap_v = cv2.VideoCapture(str(args.overlay))
    cap_3 = cv2.VideoCapture(str(args.sim_3d))
    if not cap_o.isOpened():
        raise RuntimeError(f"Cannot open original: {args.original}")
    if not cap_v.isOpened():
        raise RuntimeError(f"Cannot open overlay: {args.overlay}")
    if not cap_3.isOpened():
        raise RuntimeError(f"Cannot open 3D sim: {args.sim_3}")

    home_img = cv2.imread(str(args.home_png))
    if home_img is None:
        raise RuntimeError(f"Cannot read home layout: {args.home_png}")
    home_cell = letterbox_bgr(home_img, cw, ch)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, float(args.fps), (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot create writer: {args.output}")

    labels = (
        "Original video",
        "Detection overlay",
        "Home layout map",
        "3D simulation",
    )

    frames = 0
    while True:
        r0, f0 = cap_o.read()
        r1, f1 = cap_v.read()
        r3, f3 = cap_3.read()
        if not (r0 and r1 and r3):
            break

        p0 = letterbox_bgr(f0, cw, ch)
        p1 = letterbox_bgr(f1, cw, ch)
        p2 = home_cell.copy()
        p3 = letterbox_bgr(f3, cw, ch)

        draw_label(p0, labels[0])
        draw_label(p1, labels[1])
        draw_label(p2, labels[2])
        draw_label(p3, labels[3])

        top = np.hstack([p0, p1])
        bottom = np.hstack([p2, p3])
        quad = np.vstack([top, bottom])
        writer.write(quad)
        frames += 1

    cap_o.release()
    cap_v.release()
    cap_3.release()
    writer.release()

    print(f"Wrote {args.output} ({frames} frames @ {args.fps} fps, {out_w}x{out_h})")


if __name__ == "__main__":
    main()
