# Home World Model Builder from Video (Phase 1)

A research-grade modular robotics perception + mapping + memory system that builds a structured, queryable world model from household RGB video (optionally RGB-D).

## Modules
- SLAM (`slam/`): RTAB-Map or ORB-SLAM3 backend abstraction
- Detection (`detection/`): Grounding DINO adapter
- Segmentation (`segmentation/`): SAM adapter
- Semantic (`semantic/`): Qwen2-VL adapter
- Scene Graph (`scene_graph/`): rooms -> objects -> relations
- Memory (`memory/`): FAISS-based retrieval
- Pipeline (`pipeline/`): orchestration and I/O

## Full Setup (Models + Weights)

```bash
cd phase1
bash scripts/setup_full_phase1.sh
```

This downloads and prepares:
- Grounding DINO (`IDEA-Research/grounding-dino-tiny`)
- SAM checkpoint (`sam_vit_h_4b8939.pth`)
- Qwen2-VL (`Qwen/Qwen2-VL-2B-Instruct`)

## Backend Health Check

```bash
python3 scripts/check_backends.py --config config/default.yaml
```

The pipeline also writes backend status to:
- `outputs/<run_name>/logs/backend_status.json`

## Run
```bash
python3 pipeline_runner.py --video /path/to/video.mp4 --config config/default.yaml --output outputs/run_001
```

Optional depth stream:
```bash
python3 pipeline_runner.py \
  --video /path/to/rgb.mp4 \
  --depth-video /path/to/depth.mp4 \
  --config config/default.yaml \
  --output outputs/run_rgbd
```

## New RTAB-Map Style Output

Each run now also exports:
- `viz/rtabmap_style_pointcloud.png` (top-view heatmap-like cloud visualization)
- `map/pointcloud_rgb.ply` (queryable 3D point cloud artifact)
