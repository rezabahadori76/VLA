#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[phase1] Installing Python dependencies..."
python3 -m pip install -r "${ROOT_DIR}/requirements.txt"

echo "[phase1] Installing Segment Anything package..."
python3 -m pip install "git+https://github.com/facebookresearch/segment-anything.git"

echo "[phase1] Downloading model weights (GroundingDINO, SAM, Qwen2-VL)..."
python3 "${ROOT_DIR}/scripts/download_models.py" --root "${ROOT_DIR}/models"

echo "[phase1] Setup complete."
