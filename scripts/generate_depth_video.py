from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate monocular depth video from RGB video.")
    parser.add_argument("--input", type=Path, required=True, help="Input RGB video path.")
    parser.add_argument("--output", type=Path, required=True, help="Output depth video path (.mp4).")
    parser.add_argument(
        "--model",
        type=str,
        default="depth-anything/Depth-Anything-V2-Small-hf",
        help="Hugging Face depth-estimation model id.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap on frames (0 = all).")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input video not found: {args.input}")

    device = "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    print(f"[depth] loading model={args.model} on {device}")

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    processor = AutoImageProcessor.from_pretrained(args.model)
    model = AutoModelForDepthEstimation.from_pretrained(args.model).to(device).eval()

    cap = cv2.VideoCapture(str(args.input))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {args.input}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height), True)
    if not writer.isOpened():
        raise RuntimeError(f"Failed to create output writer: {args.output}")

    frame_idx = 0
    try:
        while cap.isOpened():
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_idx += 1
            if args.max_frames > 0 and frame_idx > args.max_frames:
                break

            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            inputs = processor(images=pil, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.inference_mode():
                out = model(**inputs)
                pred = out.predicted_depth
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=(height, width),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            depth = pred.detach().cpu().numpy().astype(np.float32)
            lo, hi = np.percentile(depth, 2.0), np.percentile(depth, 98.0)
            if hi <= lo:
                depth_norm = np.zeros_like(depth, dtype=np.uint8)
            else:
                depth_norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
                # Invert so near surfaces are brighter, useful for pseudo-depth input.
                depth_norm = (1.0 - depth_norm) * 255.0
                depth_norm = depth_norm.astype(np.uint8)

            depth_bgr = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)
            writer.write(depth_bgr)

            if frame_idx % 30 == 0:
                print(f"[depth] processed {frame_idx} frames")
    finally:
        cap.release()
        writer.release()

    print(f"[depth] done -> {args.output}")


if __name__ == "__main__":
    main()
