from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from common.logging_utils import JsonlLogger, to_jsonable
from common.types import FrameWorldState
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
from visualization.scene_graph_viz import export_scene_graph_visuals


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
        self.states: List[FrameWorldState] = []

    def initialize(self) -> None:
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
        packets = iter_video_frames(
            video_path=video_path,
            depth_video_path=depth_video_path,
            frame_stride=int(self.cfg['system'].get('frame_stride', 1)),
            max_frames=int(self.cfg['system'].get('max_frames', 300)),
            save_rgb_dir=self.output_dirs['frames'],
            intrinsics=self.cfg["slam"].get("camera_intrinsics"),
        )

        for packet in packets:
            slam_result = self.slam.process_frame(packet)
            detections = self.detector.detect(packet)
            segments = self.segmenter.segment(packet, detections)
            semantics = self.semantic.infer(packet, detections)
            state = FrameWorldState(
                frame_id=packet.frame_id,
                timestamp=packet.timestamp,
                pose=slam_result.pose,
                detections=detections,
                segments=segments,
                semantics=semantics,
            )
            self.states.append(state)
            self.memory.add_state(state)
            self._export_frame(packet.frame_id, state)
            self.logger.log('frame_processed', to_jsonable(state))

        slam_summary = self.slam.finalize()
        scene_graph = self.scene_graph_builder.build(self.states)
        self.memory.export_metadata(self.output_dirs['memory'] / 'memory_index_meta.json')
        self._export_global(slam_summary, scene_graph)
        self._export_visuals(scene_graph, slam_summary)
        self.logger.log('pipeline_completed', {'frames': len(self.states)})

        return {'frames': len(self.states), 'scene_graph_nodes': len(scene_graph.nodes), 'scene_graph_edges': len(scene_graph.edges)}

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
        render_perception_overlay(
            states=self.states,
            frames_dir=self.output_dirs['frames'],
            output_dir=self.output_dirs['viz'] / 'overlays',
            draw_masks=bool(self.cfg['visualization'].get('draw_masks', True)),
            draw_boxes=bool(self.cfg['visualization'].get('draw_boxes', True)),
        )
        export_scene_graph_visuals(
            scene_graph_path=self.output_dirs['scene_graph'] / 'scene_graph.json',
            output_png=self.output_dirs['viz'] / 'scene_graph.png',
            enabled=bool(self.cfg['visualization'].get('export_graph_png', True)),
        )
        export_memory_demo(self.memory, self.output_dirs['memory'] / 'retrieval_demo.json', [
            'Where is the kitchen?',
            'Where did I see the chair?'
        ])
