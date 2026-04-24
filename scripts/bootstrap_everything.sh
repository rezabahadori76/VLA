#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
MODELS_DIR="${ROOT_DIR}/models"
RTABMAP_DIR="${ROOT_DIR}/third_party/rtabmap"
RTABMAP_BUILD_DIR="${RTABMAP_DIR}/build"
RTABMAP_EXE="${RTABMAP_BUILD_DIR}/bin/rtabmap-rgbd_dataset"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/bootstrap_everything.sh [--skip-apt] [--skip-models] [--no-health-check]

What it does:
  1) Installs system dependencies (Ubuntu/Debian via apt)
  2) Builds RTAB-Map executables used by this project
  3) Creates Python virtualenv and installs Python dependencies
  4) Downloads models (GroundingDINO, SAM, Qwen2-VL)
  5) Generates config/local_cpu_portable.yaml with correct absolute paths
  6) (Optional) Runs backend health check
EOF
}

SKIP_APT=0
SKIP_MODELS=0
NO_HEALTH_CHECK=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-apt)
      SKIP_APT=1
      shift
      ;;
    --skip-models)
      SKIP_MODELS=1
      shift
      ;;
    --no-health-check)
      NO_HEALTH_CHECK=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      usage
      exit 1
      ;;
  esac
done

echo "[bootstrap] root: ${ROOT_DIR}"

if [[ "${SKIP_APT}" -eq 0 ]] && command -v apt-get >/dev/null 2>&1; then
  echo "[bootstrap] Installing system packages via apt..."
  export DEBIAN_FRONTEND=noninteractive
  apt-get update
  apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    python3 \
    python3-pip \
    python3-venv \
    libopencv-dev \
    libpcl-dev
else
  echo "[bootstrap] Skipping apt step."
fi

echo "[bootstrap] Preparing RTAB-Map source..."
mkdir -p "${ROOT_DIR}/third_party"
if [[ ! -d "${RTABMAP_DIR}/.git" ]]; then
  git clone --depth 1 https://github.com/introlab/rtabmap.git "${RTABMAP_DIR}"
fi

echo "[bootstrap] Configuring and building RTAB-Map tools..."
cmake -S "${RTABMAP_DIR}" -B "${RTABMAP_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
cmake --build "${RTABMAP_BUILD_DIR}" --target rgbd_dataset export -j"$(nproc)"

if [[ ! -x "${RTABMAP_EXE}" ]]; then
  echo "[bootstrap] ERROR: RTAB-Map executable not found: ${RTABMAP_EXE}"
  exit 1
fi

echo "[bootstrap] Creating Python virtualenv..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip

echo "[bootstrap] Installing Python dependencies..."
python -m pip install -r "${ROOT_DIR}/requirements.txt"
python -m pip install "git+https://github.com/facebookresearch/segment-anything.git"

if [[ "${SKIP_MODELS}" -eq 0 ]]; then
  echo "[bootstrap] Downloading model weights..."
  python "${ROOT_DIR}/scripts/download_models.py" --root "${MODELS_DIR}"
else
  echo "[bootstrap] Skipping model download."
fi

echo "[bootstrap] Generating portable config..."
python - <<'PY'
from pathlib import Path
import yaml

root = Path.cwd()
src = root / "config" / "local_cpu_quality.yaml"
if not src.exists():
    src = root / "config" / "local_cpu.yaml"

cfg = yaml.safe_load(src.read_text(encoding="utf-8"))
cfg["system"]["run_name"] = "phase1_portable_bootstrap"
cfg["system"]["device"] = "cpu"
cfg["slam"]["rtabmap"]["executable"] = str(root / "third_party" / "rtabmap" / "build" / "bin" / "rtabmap-rgbd_dataset")
cfg["slam"]["rtabmap"]["db_path"] = str(root / "outputs" / "rtabmap.db")
cfg["detection"]["model_name"] = "models/grounding_dino_tiny"
cfg["segmentation"]["checkpoint"] = "models/sam/sam_vit_h_4b8939.pth"
cfg["semantic"]["model_name"] = "models/qwen2_vl_2b"

out = root / "config" / "local_cpu_portable.yaml"
out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(out)
PY

if [[ -d "/workspace" && ! -e "/workspace/phase1" ]]; then
  ln -s "${ROOT_DIR}" /workspace/phase1 || true
fi

if [[ "${NO_HEALTH_CHECK}" -eq 0 ]]; then
  echo "[bootstrap] Running backend health check..."
  python "${ROOT_DIR}/scripts/check_backends.py" --config "${ROOT_DIR}/config/local_cpu_portable.yaml"
else
  echo "[bootstrap] Health check skipped."
fi

cat <<EOF

[bootstrap] Done.
Activate env:
  source "${VENV_DIR}/bin/activate"

Run pipeline:
  python "${ROOT_DIR}/pipeline_runner.py" \\
    --video /path/to/video.mp4 \\
    --config "${ROOT_DIR}/config/local_cpu_portable.yaml" \\
    --output "${ROOT_DIR}/outputs/run_portable"

EOF
