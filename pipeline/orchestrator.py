from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from tqdm import tqdm

from common.logging_utils import JsonlLogger, to_jsonable
from common.types import FrameWorldState, SemanticFrame
from detection.grounding_dino_detector import GroundingDinoDetector
from memory.vector_store import FaissMemoryStore
from pipeline.io_utils import iter_video_frames, write_json
from scene_graph.builder import SceneGraphBuilder
from segmentation.sam_segmenter import SAMSegmenter
from semantic.qwen2_vl_understanding import Qwen2VLSemanticModule
from slam.slam_module import SlamModule
from visualization.map_viz import export_map_visuals
from visualization.memory_demo import export_memory_demo
from visualization.overlay import render_perception_overlay
from visualization.pointcloud_viz import export_pointcloud_simulator_previews
from visualization.home_design_viz import export_home_design
from visualization.scene_graph_viz import export_scene_graph_visuals
from visualization.simulation_viz import (
    export_2d_simulation_video,
    export_3d_simulation_video,
    export_overlay_video,
)


class HomeWorldModelPipeline:
    def __init__(self, cfg: Dict[str, Any], output_dirs: Dict[str, Path]):
        self.cfg = cfg
        self.output_dirs = output_dirs
        self.logger = JsonlLogger(output_dirs['logs'] / 'pipeline_events.jsonl')

        self.slam = SlamModule(cfg['slam'])
        self.detector = GroundingDinoDetector(cfg['detection'])
        self.segmenter = SAMSegmenter(cfg['segmentation'])
        self.semantic = Qwen2VLSemanticModule(cfg['semantic'])
        self.scene_graph_builder = SceneGraphBuilder(cfg['scene_graph'])
        self.memory = FaissMemoryStore(cfg['memory']['embedding_model'], int(cfg['memory'].get('top_k', 5)))
        # Keep only lightweight per-frame state in memory.
        self.states: List[FrameWorldState] = []
        self.semantic_infer_every_n_frames = max(1, int(cfg["semantic"].get("infer_every_n_frames", 1)))
        self._last_semantic: SemanticFrame | None = None

    def initialize(self) -> None:
        frame_stride = int(self.cfg["system"].get("frame_stride", 1))
        if frame_stride < 1:
            raise RuntimeError("frame_stride must be >= 1.")
        self.slam.initialize()
        self.detector.initialize()
        self.segmenter.initialize()
        self.semantic.initialize()
        self.memory.initialize()
        backend_status = {
            "slam": self.slam.status(),
            "detection": self.detector.status(),
            "segmentation": self.segmenter.status(),
            "semantic": self.semantic.status(),
            "memory": self.memory.status(),
        }
        write_json(self.output_dirs["logs"] / "backend_status.json", backend_status)
        if bool(self.cfg["system"].get("strict_real_models", False)):
            required = ["slam", "detection", "segmentation", "semantic", "memory"]
            missing = [name for name in required if not backend_status[name]["real_backend_active"]]
            if missing:
                raise RuntimeError(
                    "strict_real_models is enabled but these modules are not active with real backends: "
                    + ", ".join(missing)
                )
        self.logger.log('pipeline_initialized', {'ok': True, 'backend_status': backend_status})

    def run(self, video_path: Path, depth_video_path: Path | None = None) -> Dict[str, Any]:
        max_frames = int(self.cfg["system"].get("max_frames", 300))
        frame_stride = int(self.cfg["system"].get("frame_stride", 1))
        packets = iter_video_frames(
            video_path=video_path,
            depth_video_path=depth_video_path,
            frame_stride=frame_stride,
            max_frames=max_frames,
            save_rgb_dir=self.output_dirs['frames'],
            intrinsics=self.cfg["slam"].get("camera_intrinsics"),
        )

        total_frames_in_video = self._read_total_video_frames(video_path)
        expected_processed = max_frames
        if total_frames_in_video is not None:
            expected_processed = min(max_frames, (total_frames_in_video + frame_stride - 1) // frame_stride)

        progress = tqdm(total=expected_processed, desc="Phase1 frames", unit="frame", dynamic_ncols=True)
        for packet in packets:
            slam_result = self.slam.process_frame(packet)
            detections = self.detector.detect(packet)
            segments = self.segmenter.segment(packet, detections)
            if self._last_semantic is None or packet.frame_id % self.semantic_infer_every_n_frames == 0:
                semantics = self.semantic.infer(packet, detections)
                self._last_semantic = semantics
            else:
                semantics = self._reuse_semantic(self._last_semantic, detections)
            if "raw_text" not in semantics.attributes:
                raise RuntimeError("Semantic output is not model-derived (missing raw_text from VLM response).")
            state = FrameWorldState(
                frame_id=packet.frame_id,
                timestamp=packet.timestamp,
                pose=slam_result.pose,
                detections=detections,
                segments=segments,
                semantics=semantics,
            )
            self.states.append(state)
            self._export_frame(packet.frame_id, state)
            render_perception_overlay(
                states=[state],
                frames_dir=self.output_dirs['frames'],
                output_dir=self.output_dirs['viz'] / 'overlays',
                draw_masks=bool(self.cfg['visualization'].get('draw_masks', True)),
                draw_boxes=bool(self.cfg['visualization'].get('draw_boxes', True)),
            )

            # Drop heavy fields (segments, raw_text attributes) from persistent in-memory state.
            compact_state = FrameWorldState(
                frame_id=state.frame_id,
                timestamp=state.timestamp,
                pose=state.pose,
                detections=state.detections,
                segments=[],
                semantics=SemanticFrame(
                    room_label=state.semantics.room_label,
                    caption=state.semantics.caption,
                    attributes={},
                ),
            )
            self.states[-1] = compact_state
            self.memory.add_state(compact_state)
            self.logger.log(
                'frame_processed',
                {
                    'frame_id': compact_state.frame_id,
                    'timestamp': compact_state.timestamp,
                    'pose': to_jsonable(compact_state.pose),
                    'detections_count': len(compact_state.detections),
                    'segments_count': len(segments),
                    'room': compact_state.semantics.room_label,
                },
            )
            processed = packet.frame_id + 1
            remaining = max(expected_processed - processed, 0)
            progress.update(1)
            progress.set_postfix_str(f"processed={processed} remaining={remaining}")

        progress.close()

        slam_summary = self.slam.finalize()
        scene_graph = self.scene_graph_builder.build(self.states)
        self.memory.export_metadata(self.output_dirs['memory'] / 'memory_index_meta.json')
        self._export_global(slam_summary, scene_graph)
        self._export_visuals(scene_graph, slam_summary)
        self.logger.log('pipeline_completed', {'frames': len(self.states)})

        return {'frames': len(self.states), 'scene_graph_nodes': len(scene_graph.nodes), 'scene_graph_edges': len(scene_graph.edges)}

    def _read_total_video_frames(self, video_path: Path) -> int | None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total if total > 0 else None

    def _reuse_semantic(self, previous: SemanticFrame, detections) -> SemanticFrame:
        attrs = dict(previous.attributes)
        attrs["objects"] = [d.label for d in detections]
        attrs["raw_text"] = attrs.get("raw_text", "[reused semantic]")
        attrs["semantic_reused"] = True
        return SemanticFrame(
            room_label=previous.room_label,
            caption=previous.caption,
            attributes=attrs,
        )

    def _export_frame(self, frame_id: int, state: FrameWorldState) -> None:
        write_json(self.output_dirs['detections'] / f'frame_{frame_id:06d}.json', {'detections': to_jsonable(state.detections)})
        write_json(self.output_dirs['segments'] / f'frame_{frame_id:06d}.json', {'segments': to_jsonable(state.segments)})
        write_json(self.output_dirs['semantics'] / f'frame_{frame_id:06d}.json', {'semantic': to_jsonable(state.semantics)})

    def _export_global(self, slam_summary: Dict[str, Any], scene_graph) -> None:
        traj = slam_summary.get('trajectory', [])
        map_export = slam_summary.get("map_export", {})
        pgm_path = map_export.get("occupancy_pgm")
        cloud_path = map_export.get("cloud_ply")

        if not pgm_path or not Path(pgm_path).exists():
            raise RuntimeError("SLAM did not produce a real occupancy map export (PGM).")
        if not cloud_path or not Path(cloud_path).exists():
            raise RuntimeError("SLAM did not produce a real point cloud export (PLY).")

        import cv2

        pgm = cv2.imread(str(pgm_path), cv2.IMREAD_UNCHANGED)
        if pgm is None:
            raise RuntimeError(f"Failed to read occupancy PGM map: {pgm_path}")
        np.save(self.output_dirs['map'] / 'occupancy_grid.npy', pgm)

        cloud_dst = self.output_dirs["map"] / "pointcloud_rgb.ply"
        cloud_dst.write_bytes(Path(cloud_path).read_bytes())

        write_json(self.output_dirs['map'] / 'trajectory.json', {'trajectory': traj})
        write_json(self.output_dirs['map'] / 'pose_graph.json', slam_summary.get('pose_graph', {}))
        write_json(self.output_dirs['scene_graph'] / 'scene_graph.json', {'nodes': scene_graph.nodes, 'edges': scene_graph.edges, 'metadata': scene_graph.metadata})

    def _export_visuals(self, scene_graph, slam_summary: Dict[str, Any]) -> None:
        export_map_visuals(self.output_dirs['map'] / 'occupancy_grid.npy', self.output_dirs['viz'] / 'slam_map.png')
        export_scene_graph_visuals(
            scene_graph_path=self.output_dirs['scene_graph'] / 'scene_graph.json',
            output_png=self.output_dirs['viz'] / 'scene_graph.png',
            enabled=bool(self.cfg['visualization'].get('export_graph_png', True)),
        )
        export_overlay_video(
            overlays_dir=self.output_dirs["viz"] / "overlays",
            output_mp4=self.output_dirs["viz"] / "overlay_preview.mp4",
            fps=int(self.cfg["visualization"].get("video_fps", 10)),
        )
        export_2d_simulation_video(
            occupancy_grid_npy=self.output_dirs["map"] / "occupancy_grid.npy",
            trajectory_json=self.output_dirs["map"] / "trajectory.json",
            output_mp4=self.output_dirs["viz"] / "simulation_2d.mp4",
            fps=int(self.cfg["visualization"].get("video_fps", 10)),
        )
        export_3d_simulation_video(
            cloud_ply=self.output_dirs["map"] / "pointcloud_rgb.ply",
            trajectory_json=self.output_dirs["map"] / "trajectory.json",
            output_mp4=self.output_dirs["viz"] / "simulation_3d.mp4",
            fps=int(self.cfg["visualization"].get("video_fps", 10)),
            max_points=int(self.cfg["visualization"].get("simulation3d_max_points", 18000)),
        )
        profiles = self.cfg["visualization"].get("simulation_profiles", [])
        for prof in profiles:
            if not isinstance(prof, dict):
                continue
            name = str(prof.get("name", "")).strip().lower()
            if not name:
                continue
            fps = int(prof.get("fps", self.cfg["visualization"].get("video_fps", 10)))
            max_points = int(prof.get("max_points", self.cfg["visualization"].get("simulation3d_max_points", 18000)))
            export_2d_simulation_video(
                occupancy_grid_npy=self.output_dirs["map"] / "occupancy_grid.npy",
                trajectory_json=self.output_dirs["map"] / "trajectory.json",
                output_mp4=self.output_dirs["viz"] / f"simulation_2d_{name}.mp4",
                fps=fps,
            )
            export_3d_simulation_video(
                cloud_ply=self.output_dirs["map"] / "pointcloud_rgb.ply",
                trajectory_json=self.output_dirs["map"] / "trajectory.json",
                output_mp4=self.output_dirs["viz"] / f"simulation_3d_{name}.mp4",
                fps=fps,
                max_points=max_points,
            )
        export_pointcloud_simulator_previews(
            cloud_ply=self.output_dirs["map"] / "pointcloud_rgb.ply",
            output_realistic_png=self.output_dirs["viz"] / "simulator_realistic.png",
            output_heatmap_png=self.output_dirs["viz"] / "simulator_heatmap.png",
            point_budget=int(self.cfg["visualization"].get("simulator_point_budget", 250000)),
        )
        export_home_design(
            states=self.states,
            output_png=self.output_dirs["viz"] / "home_design_layout.png",
            output_json=self.output_dirs["viz"] / "home_design_layout.json",
            canvas_size=int(self.cfg["visualization"].get("home_design_canvas_size", 1200)),
        )
        export_memory_demo(self.memory, self.output_dirs['memory'] / 'retrieval_demo.json', [
            'Where is the kitchen?',
            'Where did I see the chair?'
        ])
