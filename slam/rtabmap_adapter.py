from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import os
import json
from typing import Dict

import cv2
import numpy as np

from common.types import FramePacket, Pose, SlamFrameResult
from slam.base import SlamBackend


class RTABMapBackend(SlamBackend):
    def __init__(
        self,
        executable: str,
        db_path: str,
        resolution: float,
        camera_intrinsics: Dict[str, float] | None = None,
        require_depth: bool = True,
    ):
        self.executable = executable
        self.db_path = Path(db_path)
        self.resolution = resolution
        self.camera_intrinsics = camera_intrinsics or {}
        self.require_depth = require_depth

        self.is_real_backend = False
        self.init_error: str | None = None
        self.sequence_dir = self.db_path.parent / "rtabmap_sequence"
        self.output_dir = self.db_path.parent / "rtabmap_output"
        self.output_name = "rtabmap"
        self.last_poses: list[Dict[str, float]] = []
        self.frames_written = 0
        self._command_log: list[Dict[str, object]] = []

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        exe_ok = Path(self.executable).exists() or shutil.which(self.executable)
        if not exe_ok:
            self.init_error = f"RTAB-Map executable not found: {self.executable}"
            raise RuntimeError(self.init_error)

        self.sequence_dir.mkdir(parents=True, exist_ok=True)
        (self.sequence_dir / "rgb_sync").mkdir(parents=True, exist_ok=True)
        (self.sequence_dir / "depth_sync").mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._clear_sequence_folder()
        self._clear_output_folder()
        self._write_calibration_file()

        self.is_real_backend = True
        self.init_error = None
        self.frames_written = 0
        self.last_poses = []
        self._command_log = []

    def process_frame(self, packet: FramePacket) -> SlamFrameResult:
        if not self.is_real_backend:
            raise RuntimeError("RTAB-Map backend is not initialized.")
        if packet.rgb is None:
            raise RuntimeError("RTAB-Map backend received an empty RGB frame.")
        if self.require_depth and packet.depth is None:
            raise RuntimeError(
                "RTAB-Map backend requires depth input (packet.depth is None). "
                "Provide --depth-video or disable require_depth explicitly."
            )

        if self.frames_written == 0:
            self._sync_intrinsics_with_frame(packet.rgb)
            self._write_calibration_file()

        # Avoid 0.000000 first stamp as CameraImages may fail parsing counts in some builds.
        stamp_seconds = 1.0 + packet.frame_id * 0.1
        stamp = f"{stamp_seconds:.6f}.png"
        rgb_path = self.sequence_dir / "rgb_sync" / stamp
        depth_path = self.sequence_dir / "depth_sync" / stamp
        cv2.imwrite(str(rgb_path), packet.rgb)

        if packet.depth is not None:
            depth_frame = self._prepare_depth_frame(packet.depth)
            cv2.imwrite(str(depth_path), depth_frame)

        self.frames_written += 1
        self._update_trajectory(force=True)

        if not self.last_poses:
            raise RuntimeError(
                "RTAB-Map produced no valid poses. "
                "Check camera intrinsics, frame continuity, and depth consistency."
            )
        last = self.last_poses[-1]
        pose = Pose(
            x=last["x"],
            y=last["y"],
            z=last["z"],
            qx=last["qx"],
            qy=last["qy"],
            qz=last["qz"],
            qw=last["qw"],
        )
        return SlamFrameResult(
            pose=pose,
            keyframe_id=len(self.last_poses) - 1,
            map_update={"backend": "rtabmap", "poses_count": len(self.last_poses), "timestamp": last["timestamp"]},
        )

    def finalize(self) -> Dict:
        if not self.is_real_backend:
            raise RuntimeError("RTAB-Map finalize called before successful initialization.")
        self._update_trajectory(force=True)
        if len(self.last_poses) < 2:
            raise RuntimeError("RTAB-Map finalize called with empty trajectory.")

        db_generated = self.output_dir / f"{self.output_name}.db"
        if not db_generated.exists():
            raise RuntimeError(f"RTAB-Map database not found: {db_generated}")
        if db_generated.resolve() != self.db_path.resolve():
            shutil.copy2(db_generated, self.db_path)

        map_export = self._export_map_artifacts()

        trajectory = [
            {
                "x": p["x"],
                "y": p["y"],
                "z": p["z"],
                "qx": p["qx"],
                "qy": p["qy"],
                "qz": p["qz"],
                "qw": p["qw"],
                "timestamp": p["timestamp"],
            }
            for p in self.last_poses
        ]
        nodes = [{"id": i, "pose": pose} for i, pose in enumerate(trajectory)]
        edges = [{"from": i - 1, "to": i, "type": "odometry"} for i in range(1, len(nodes))]

        return {
            "backend": "rtabmap",
            "real_backend_active": True,
            "trajectory": trajectory,
            "pose_graph": {"nodes": nodes, "edges": edges},
            "occupancy_grid": None,
            "db_path": str(self.db_path),
            "map_export": {
                "occupancy_pgm": map_export["occupancy_pgm"],
                "occupancy_yaml": map_export["occupancy_yaml"],
                "cloud_ply": map_export["cloud_ply"],
            },
        }

    def status(self) -> Dict:
        return {
            "module": "slam",
            "backend": "rtabmap",
            "real_backend_active": self.is_real_backend,
            "executable": self.executable,
            "init_error": self.init_error,
            "db_path": str(self.db_path),
            "require_depth": self.require_depth,
        }

    def _write_calibration_file(self) -> None:
        calib_path = self.sequence_dir / "rtabmap_calib.yaml"
        width = int(self.camera_intrinsics.get("width", 1280))
        height = int(self.camera_intrinsics.get("height", 720))
        fx = float(self.camera_intrinsics.get("fx", 700.0))
        fy = float(self.camera_intrinsics.get("fy", 700.0))
        cx = float(self.camera_intrinsics.get("cx", width / 2.0))
        cy = float(self.camera_intrinsics.get("cy", height / 2.0))

        calib_text = (
            "%YAML:1.0\n"
            "---\n"
            "camera_name: rtabmap_calib\n"
            f"image_width: {width}\n"
            f"image_height: {height}\n"
            "camera_matrix:\n"
            "   rows: 3\n"
            "   cols: 3\n"
            f"   data: [ {fx}, 0., {cx}, 0., {fy}, {cy}, 0., 0., 1. ]\n"
            "distortion_model: plumb_bob\n"
            "distortion_coefficients:\n"
            "   rows: 1\n"
            "   cols: 5\n"
            "   data: [ 0., 0., 0., 0., 0. ]\n"
            "rectification_matrix:\n"
            "   rows: 3\n"
            "   cols: 3\n"
            "   data: [ 1., 0., 0., 0., 1., 0., 0., 0., 1. ]\n"
            "projection_matrix:\n"
            "   rows: 3\n"
            "   cols: 4\n"
            f"   data: [ {fx}, 0., {cx}, 0., 0., {fy}, {cy}, 0., 0., 0., 1., 0. ]\n"
            "local_transform:\n"
            "   rows: 3\n"
            "   cols: 4\n"
            "   data: [ 0., 0., 1., 0., -1., 0., 0., 0., 0., -1., 0., 0. ]\n"
        )
        calib_path.write_text(calib_text, encoding="utf-8")

    def _clear_sequence_folder(self) -> None:
        for sub in ("rgb_sync", "depth_sync"):
            folder = self.sequence_dir / sub
            for fp in folder.glob("*.png"):
                fp.unlink(missing_ok=True)
        (self.sequence_dir / "rtabmap_association.txt").unlink(missing_ok=True)

    def _clear_output_folder(self) -> None:
        for name in (
            f"{self.output_name}.db",
            f"{self.output_name}_poses.txt",
            "slam_cloud.ply",
            "slam_map.pgm",
            "slam_map.yaml",
            "rtabmap_command_log.json",
        ):
            (self.output_dir / name).unlink(missing_ok=True)

    def _prepare_depth_frame(self, depth_frame) -> "cv2.typing.MatLike":
        if depth_frame.ndim == 3:
            depth_frame = cv2.cvtColor(depth_frame, cv2.COLOR_BGR2GRAY)
        depth_u8 = depth_frame.astype("uint8")
        depth_u8 = cv2.GaussianBlur(depth_u8, (5, 5), 0)
        depth_u16 = cv2.normalize(depth_u8, None, 400, 4500, cv2.NORM_MINMAX).astype("uint16")
        return depth_u16

    def _sync_intrinsics_with_frame(self, rgb_frame) -> None:
        h, w = rgb_frame.shape[:2]
        old_w = float(self.camera_intrinsics.get("width", w))
        old_h = float(self.camera_intrinsics.get("height", h))
        fx = float(self.camera_intrinsics.get("fx", max(w, h)))
        fy = float(self.camera_intrinsics.get("fy", max(w, h)))
        cx = float(self.camera_intrinsics.get("cx", old_w / 2.0))
        cy = float(self.camera_intrinsics.get("cy", old_h / 2.0))
        sx = float(w) / max(old_w, 1.0)
        sy = float(h) / max(old_h, 1.0)
        self.camera_intrinsics.update(
            {
                "width": int(w),
                "height": int(h),
                "fx": fx * sx,
                "fy": fy * sy,
                "cx": cx * sx,
                "cy": cy * sy,
            }
        )

    def _update_trajectory(self, force: bool) -> None:
        self._run_rtabmap_dataset()
        poses = self._load_poses_file()
        if poses:
            self.last_poses = poses
            return
        if force:
            raise RuntimeError(
                "RTAB-Map produced no valid poses for the current sequence. "
                "Check camera intrinsics, frame quality, and depth consistency."
            )

    def _run_rtabmap_dataset(self) -> None:
        cmd = [
            self.executable,
            "--output",
            str(self.output_dir),
            "--output_name",
            self.output_name,
            "--Rtabmap/DetectionRate",
            "0",
            "--RGBD/LinearUpdate",
            "0",
            "--RGBD/AngularUpdate",
            "0",
            "--Grid/FromDepth",
            "true",
            "--Grid/3D",
            "false",
            "--Grid/CellSize",
            str(self.resolution),
            "--Vis/MinInliers",
            "10",
            str(self.sequence_dir),
        ]
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        self._record_command(cmd)
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
        if result.returncode != 0:
            raise RuntimeError(
                "RTAB-Map dataset processing failed:\n"
                f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
            )

    def _load_poses_file(self) -> list[Dict[str, float]]:
        poses_path = self.output_dir / f"{self.output_name}_poses.txt"
        if not poses_path.exists():
            return []
        poses: list[Dict[str, float]] = []
        for raw in poses_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            vals = line.split()
            if len(vals) < 8:
                continue
            poses.append(
                {
                    "timestamp": float(vals[0]),
                    "x": float(vals[1]),
                    "y": float(vals[2]),
                    "z": float(vals[3]),
                    "qx": float(vals[4]),
                    "qy": float(vals[5]),
                    "qz": float(vals[6]),
                    "qw": float(vals[7]),
                }
            )
        return poses

    def _export_map_artifacts(self) -> Dict[str, str]:
        db_path = self.output_dir / f"{self.output_name}.db"
        export_bin = str(Path(self.executable).with_name("rtabmap-export"))
        cloud_cmd = [
            export_bin,
            "--cloud",
            "--ascii",
            "--output",
            "slam_cloud",
            "--output_dir",
            str(self.output_dir),
            str(db_path),
        ]
        map_cmd = [
            export_bin,
            "--map",
            "--output",
            "slam_map",
            "--output_dir",
            str(self.output_dir),
            str(db_path),
        ]
        for cmd in (cloud_cmd, map_cmd):
            self._record_command(cmd)
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"RTAB-Map export failed: {result.stderr or result.stdout}")
        self._write_command_log()
        cloud_file = self._pick_first_existing("slam_cloud*.ply")
        map_file = self._pick_first_existing("slam_map*.pgm")
        yaml_file = self._pick_first_existing("slam_map*.yaml")
        if cloud_file is None:
            raise RuntimeError("RTAB-Map export did not produce any cloud PLY file.")
        if map_file is None or yaml_file is None:
            raise RuntimeError(
                "RTAB-Map export did not produce occupancy map files (PGM/YAML). "
                "No synthetic occupancy reconstruction is allowed."
            )
        return {
            "cloud_ply": str(cloud_file),
            "occupancy_pgm": str(map_file),
            "occupancy_yaml": str(yaml_file),
        }

    def _pick_first_existing(self, pattern: str) -> Path | None:
        matches = sorted(self.output_dir.glob(pattern))
        return matches[0] if matches else None

    def _record_command(self, cmd: list[str]) -> None:
        self._command_log.append({"argv": cmd})

    def _write_command_log(self) -> None:
        if not self._command_log:
            return
        out = self.output_dir / "rtabmap_command_log.json"
        out.write_text(json.dumps({"commands": self._command_log}, indent=2), encoding="utf-8")
