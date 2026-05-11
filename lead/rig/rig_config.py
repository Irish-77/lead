"""Rig configuration dataclasses (py123d / ISO 8855 frame).

A :class:`RigConfig` is a complete sensor mounting specification — name,
camera entries (intrinsics + extrinsics + distortion), lidar entries, and
optional per-frame wiggle parameters. It is the single source of truth that
gets:

- generated either randomly (:mod:`lead.rig.random_rig`) or by extracting
  from a real dataset (:mod:`lead.rig.dataset_rig`),
- converted to a CARLA sensor-config list (:mod:`lead.rig.coord_transform`),
- serialized to JSON for reproducible runs (:mod:`lead.rig.serialization`).

All extrinsics are stored in py123d's rear-axle frame using the OpenCV / pZmYpX
camera convention; lidar entries are in py123d's ISO 8855 vehicle frame
without a camera-convention swap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from py123d.datatypes import CameraID, LidarID
from py123d.datatypes.sensors.pinhole_camera import PinholeDistortion, PinholeIntrinsics
from py123d.geometry import PoseSE3


@dataclass(frozen=True)
class CameraEntry:
    """Single camera in a rig.

    Attributes:
        camera_id: py123d camera identifier (PCAM_F0, PCAM_L0, ...).
        camera_name: Human-readable camera name (used as the dataset name).
        width: Image width in pixels.
        height: Image height in pixels.
        intrinsics: Pinhole intrinsics ``(fx, fy, cx, cy, skew)``.
        camera_to_imu_se3: Camera extrinsic in py123d rear-axle frame.
        distortion: Optional distortion parameters (None for CARLA sensors).
        is_undistorted: Whether the recorded images are already undistorted.
    """

    camera_id: CameraID
    camera_name: str
    width: int
    height: int
    intrinsics: PinholeIntrinsics
    camera_to_imu_se3: PoseSE3
    distortion: Optional[PinholeDistortion] = None
    is_undistorted: bool = True


@dataclass(frozen=True)
class LidarEntry:
    """Single lidar in a rig.

    Attributes:
        lidar_id: py123d lidar identifier.
        lidar_name: Human-readable lidar name.
        lidar_to_imu_se3: Lidar extrinsic in py123d frame.
        range_m: Maximum sensor range in meters (CARLA simulator parameter).
        rotation_frequency_hz: CARLA lidar rotation frequency.
        channels: Number of laser channels.
        points_per_second: Total points emitted per second.
        upper_fov_deg: Upper field of view in degrees.
        lower_fov_deg: Lower field of view in degrees.
    """

    lidar_id: LidarID
    lidar_name: str
    lidar_to_imu_se3: PoseSE3
    range_m: float = 100.0
    rotation_frequency_hz: float = 10.0
    channels: int = 64
    points_per_second: int = 1_200_000
    upper_fov_deg: float = 10.0
    lower_fov_deg: float = -30.0


@dataclass(frozen=True)
class WiggleConfig:
    """Per-frame extrinsic and intrinsic jitter parameters.

    Extrinsic wiggle physically perturbs the sensor on the ego (so the
    rendered frame and the recorded extrinsic stay consistent). Intrinsic
    wiggle is *recorded-only* — CARLA blueprints cannot change FOV per
    tick, so the jitter is applied to the metadata that gets written to the
    Arrow log. Default sigmas are sub-pixel / sub-mm and primarily intended
    to break numerical determinism, not to model real lens vibration.

    Attributes:
        enabled: If False, no wiggle is applied at runtime.
        translation_sigma_m: Per-axis translation jitter standard deviation.
        rotation_sigma_deg: Per-axis rotation jitter standard deviation.
        intrinsic_focal_sigma_px: Per-frame stddev added to fx and fy.
        intrinsic_cxcy_sigma_px: Per-frame stddev added to cx and cy.
        seed: Optional RNG seed for reproducibility.
    """

    enabled: bool = True
    translation_sigma_m: float = 0.01
    rotation_sigma_deg: float = 0.5
    intrinsic_focal_sigma_px: float = 0.1
    intrinsic_cxcy_sigma_px: float = 0.1
    seed: Optional[int] = None


@dataclass(frozen=True)
class RigConfig:
    """A full rig: name, cameras, lidars, optional wiggle.

    Attributes:
        rig_name: Identifier used for output dirs (e.g. ``"carla_nuscenes"``).
        cameras: List of :class:`CameraEntry`.
        lidars: List of :class:`LidarEntry` (empty list = no lidar saved).
        wiggle: Per-frame jitter parameters.
        source: Free-form provenance tag (``"random:seed=42"``, ``"nuscenes"``).
    """

    rig_name: str
    cameras: list[CameraEntry]
    lidars: list[LidarEntry] = field(default_factory=list)
    wiggle: WiggleConfig = field(default_factory=WiggleConfig)
    source: str = ""
