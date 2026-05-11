"""JSON serialization for :class:`RigConfig`.

Round-trip-safe: ``load(path_for(rig)) == rig`` for every rig the rest of the
package can produce.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from beartype import beartype
from py123d.datatypes import CameraID, LidarID
from py123d.datatypes.sensors.pinhole_camera import PinholeDistortion, PinholeIntrinsics
from py123d.geometry import PoseSE3

from lead.rig.rig_config import (
    CameraEntry,
    LidarEntry,
    RigConfig,
    WiggleConfig,
)


@beartype
def _camera_to_dict(camera: CameraEntry) -> dict[str, Any]:
    return {
        "camera_id": int(camera.camera_id),
        "camera_name": camera.camera_name,
        "width": camera.width,
        "height": camera.height,
        "intrinsics": camera.intrinsics.tolist(),
        "camera_to_imu_se3": camera.camera_to_imu_se3.tolist(),
        "distortion": camera.distortion.tolist() \
            if camera.distortion is not None else None,
        "is_undistorted": camera.is_undistorted,
    }


@beartype
def _camera_from_dict(data: dict[str, Any]) -> CameraEntry:
    distortion = PinholeDistortion.from_list(data["distortion"]) \
        if data["distortion"] is not None else None
    return CameraEntry(
        camera_id=CameraID(data["camera_id"]),
        camera_name=data["camera_name"],
        width=int(data["width"]),
        height=int(data["height"]),
        intrinsics=PinholeIntrinsics.from_list(data["intrinsics"]),
        camera_to_imu_se3=PoseSE3.from_list(data["camera_to_imu_se3"]),
        distortion=distortion,
        is_undistorted=bool(data["is_undistorted"]),
    )


@beartype
def _lidar_to_dict(lidar: LidarEntry) -> dict[str, Any]:
    return {
        "lidar_id": int(lidar.lidar_id),
        "lidar_name": lidar.lidar_name,
        "lidar_to_imu_se3": lidar.lidar_to_imu_se3.tolist(),
        "range_m": lidar.range_m,
        "rotation_frequency_hz": lidar.rotation_frequency_hz,
        "channels": lidar.channels,
        "points_per_second": lidar.points_per_second,
        "upper_fov_deg": lidar.upper_fov_deg,
        "lower_fov_deg": lidar.lower_fov_deg,
    }


@beartype
def _lidar_from_dict(data: dict[str, Any]) -> LidarEntry:
    return LidarEntry(
        lidar_id=LidarID(data["lidar_id"]),
        lidar_name=data["lidar_name"],
        lidar_to_imu_se3=PoseSE3.from_list(data["lidar_to_imu_se3"]),
        range_m=float(data["range_m"]),
        rotation_frequency_hz=float(data["rotation_frequency_hz"]),
        channels=int(data["channels"]),
        points_per_second=int(data["points_per_second"]),
        upper_fov_deg=float(data["upper_fov_deg"]),
        lower_fov_deg=float(data["lower_fov_deg"]),
    )


@beartype
def to_dict(rig: RigConfig) -> dict[str, Any]:
    """Convert a :class:`RigConfig` to a JSON-serializable dict."""
    return {
        "rig_name": rig.rig_name,
        "source": rig.source,
        "wiggle": {
            "enabled": rig.wiggle.enabled,
            "translation_sigma_m": rig.wiggle.translation_sigma_m,
            "rotation_sigma_deg": rig.wiggle.rotation_sigma_deg,
            "intrinsic_focal_sigma_px": rig.wiggle.intrinsic_focal_sigma_px,
            "intrinsic_cxcy_sigma_px": rig.wiggle.intrinsic_cxcy_sigma_px,
            "seed": rig.wiggle.seed,
        },
        "cameras": [_camera_to_dict(camera) for camera in rig.cameras],
        "lidars": [_lidar_to_dict(lidar) for lidar in rig.lidars],
    }


@beartype
def from_dict(data: dict[str, Any]) -> RigConfig:
    """Construct a :class:`RigConfig` from a JSON-serializable dict."""
    wiggle_data = data.get("wiggle", {})
    wiggle = WiggleConfig(
        enabled=bool(wiggle_data.get("enabled", True)),
        translation_sigma_m=float(
            wiggle_data.get("translation_sigma_m", 0.01)
        ),
        rotation_sigma_deg=float(
            wiggle_data.get("rotation_sigma_deg", 0.5)
        ),
        intrinsic_focal_sigma_px=float(
            wiggle_data.get("intrinsic_focal_sigma_px", 0.1)
        ),
        intrinsic_cxcy_sigma_px=float(
            wiggle_data.get("intrinsic_cxcy_sigma_px", 0.1)
        ),
        seed=wiggle_data.get("seed"),
    )
    return RigConfig(
        rig_name=data["rig_name"],
        cameras=[_camera_from_dict(camera) for camera in data["cameras"]],
        lidars=[_lidar_from_dict(lidar) for lidar in data.get("lidars", [])],
        wiggle=wiggle,
        source=data.get("source", ""),
    )


@beartype
def save(rig: RigConfig, path: Path | str) -> None:
    """Write a rig config to a JSON file (parent dirs are created)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_dict(rig), indent=2))


@beartype
def load(path: Path | str) -> RigConfig:
    """Read a rig config from a JSON file."""
    return from_dict(json.loads(Path(path).read_text()))
