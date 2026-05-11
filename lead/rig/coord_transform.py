"""Bidirectional coordinate conversion between CARLA (Unreal) and py123d
(ISO 8855).

CARLA / Unreal:
    - Translation:  X forward, Y right, Z up (left-handed)
    - Rotation:     roll/pitch/yaw in degrees
    - Camera:       pXpZmY (X forward, Z up, -Y right)

py123d / ISO 8855:
    - Translation:  X forward, Y left, Z up (right-handed)
    - Rotation:     quaternion (qw, qx, qy, qz)
    - Camera:       pZmYpX (Z forward, -Y up, X right)  (OpenCV / COLMAP)

Refer to: https://kesai.eu/py123d/notes/conventions/

The forward path (CARLA -> py123d) is implemented by LEAD in
:mod:`lead.expert.expert_py123d_utils`. This module provides the inverse path
(py123d -> CARLA) used to build CARLA sensor configs from a :class:`RigConfig`
that was either generated randomly or extracted from a real dataset.
"""

from __future__ import annotations

import carla
import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from beartype import beartype
from py123d.datatypes import EgoStateSE3Metadata
from py123d.geometry import EulerAngles, PoseSE3, Quaternion, Vector3D
from py123d.geometry.transform import translate_se3_along_body_frame
from py123d.parser.utils.sensor_utils.camera_conventions import (
    convert_camera_convention,
)


@beartype
def carla_rotation_to_quaternion(
    roll_deg: float,
    pitch_deg: float,
    yaw_deg: float
) -> Quaternion:
    """CARLA Euler (degrees) to py123d Quaternion (ISO 8855).

    Args:
        roll_deg: CARLA roll in degrees.
        pitch_deg: CARLA pitch in degrees.
        yaw_deg: CARLA yaw in degrees.

    Returns:
        Quaternion in ISO 8855 frame (pitch and yaw negated).
    """
    euler = EulerAngles(
        roll=np.deg2rad(roll_deg),
        pitch=-np.deg2rad(pitch_deg),
        yaw=-np.deg2rad(yaw_deg),
    )
    return Quaternion.from_rotation_matrix(euler.rotation_matrix)


@beartype
def quaternion_to_carla_rotation(
    quaternion: Quaternion
) -> tuple[float, float, float]:
    """py123d Quaternion (ISO 8855) to CARLA Euler (degrees).

    Args:
        quaternion: Quaternion in ISO 8855 frame.

    Returns:
        Tuple ``(roll_deg, pitch_deg, yaw_deg)`` in CARLA convention.
    """
    euler = EulerAngles.from_rotation_matrix(quaternion.rotation_matrix)
    return (
        float(np.rad2deg(euler.roll)),
        float(np.rad2deg(-euler.pitch)),
        float(np.rad2deg(-euler.yaw)),
    )


@beartype
def _rear_axle_to_floor_center_translate(
    pose: PoseSE3,
    ego_metadata: EgoStateSE3Metadata,
) -> PoseSE3:
    """Translate a pose from rear-axle frame back to floor-center frame.

    This is the inverse of
    :func:`lead.expert.expert_py123d_utils.floor_center_to_rear_axle_translate`,
    used when going from a py123d rig extrinsic to a CARLA mount point.
    """
    zero_pose = PoseSE3(x=0.0, y=0.0, z=0.0, qw=1.0, qx=0.0, qy=0.0, qz=0.0)
    rear_axle_translate = translate_se3_along_body_frame(
        zero_pose,
        Vector3D(
            x=ego_metadata.rear_axle_to_center_longitudinal,
            y=0.0,
            z=-(ego_metadata.half_height) + \
                ego_metadata.rear_axle_to_center_vertical,
        ),
    )
    new = PoseSE3.from_array(pose.array.copy())
    new.array[:3] -= rear_axle_translate.array[:3]
    return new


@beartype
def carla_camera_extrinsic_to_iso(
    camera_pos: jt.Float[npt.NDArray, "3"] | list[float],
    camera_rot_deg: jt.Float[npt.NDArray, "3"] | list[float],
    ego_metadata: EgoStateSE3Metadata,
) -> PoseSE3:
    """CARLA camera mount (floor-center, pXpZmY) -> py123d extrinsic (rear-axle,
    pZmYpX).

    Mirrors :func:`lead.expert.expert_py123d_utils.get_camera_extrinsic_as_iso`
    so we have a single canonical implementation here. Both directions of the
    transform live in this module to make the round-trip property explicit.

    Args:
        camera_pos: Mount position ``[x, y, z]`` in CARLA (Unreal) coordinates.
        camera_rot_deg: Mount rotation ``[roll, pitch, yaw]`` in CARLA degrees.
        ego_metadata: Ego vehicle metadata (rear-axle offsets).

    Returns:
        PoseSE3 camera extrinsic in py123d (ISO 8855 / OpenCV) convention.
    """
    quaternion = carla_rotation_to_quaternion(
        camera_rot_deg[0], camera_rot_deg[1], camera_rot_deg[2]
    )

    pose = PoseSE3(
        x=float(camera_pos[0]),
        y=-float(camera_pos[1]),
        z=float(camera_pos[2]),
        qw=quaternion.qw,
        qx=quaternion.qx,
        qy=quaternion.qy,
        qz=quaternion.qz,
    )

    from lead.expert.expert_py123d_utils import \
        floor_center_to_rear_axle_translate

    pose = floor_center_to_rear_axle_translate(pose, ego_metadata)
    pose = convert_camera_convention(
        pose, from_convention="pXpZmY", to_convention="pZmYpX"
    )
    return pose


@beartype
def iso_camera_extrinsic_to_carla(
    extrinsic: PoseSE3,
    ego_metadata: EgoStateSE3Metadata,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """py123d camera extrinsic (rear-axle, pZmYpX) -> CARLA mount (floor-center,
    pXpZmY).

    Inverse of :func:`carla_camera_extrinsic_to_iso`.

    Args:
        extrinsic: PoseSE3 in py123d (ISO 8855 / OpenCV) convention.
        ego_metadata: Ego vehicle metadata (rear-axle offsets).

    Returns:
        Tuple ``((x, y, z), (roll_deg, pitch_deg, yaw_deg))`` ready for a CARLA
        sensor config dict.
    """
    pose = convert_camera_convention(
        extrinsic, from_convention="pZmYpX", to_convention="pXpZmY"
    )
    pose = _rear_axle_to_floor_center_translate(pose, ego_metadata)

    quaternion = Quaternion(qw=pose.qw, qx=pose.qx, qy=pose.qy, qz=pose.qz)
    roll, pitch, yaw = quaternion_to_carla_rotation(quaternion)

    return ((float(pose.x), -float(pose.y), float(pose.z)), (roll, pitch, yaw))


@beartype
def carla_lidar_extrinsic_to_iso(
    lidar_pos: jt.Float[npt.NDArray, "3"] | list[float],
    lidar_rot_deg: jt.Float[npt.NDArray, "3"] | list[float],
) -> PoseSE3:
    """CARLA lidar mount -> py123d extrinsic (rear-axle, no convention swap).

    Lidar uses the same body-frame axes as the ego, so no camera-convention
    swap is needed. Position Y is negated and pitch/yaw are negated in the
    quaternion, just like cameras.
    """
    quaternion = carla_rotation_to_quaternion(
        lidar_rot_deg[0], lidar_rot_deg[1], lidar_rot_deg[2]
    )
    return PoseSE3(
        x=float(lidar_pos[0]),
        y=-float(lidar_pos[1]),
        z=float(lidar_pos[2]),
        qw=quaternion.qw,
        qx=quaternion.qx,
        qy=quaternion.qy,
        qz=quaternion.qz,
    )


@beartype
def iso_lidar_extrinsic_to_carla(
    extrinsic: PoseSE3,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """py123d lidar extrinsic -> CARLA mount."""
    quaternion = Quaternion(
        qw=extrinsic.qw, qx=extrinsic.qx, qy=extrinsic.qy, qz=extrinsic.qz
    )
    roll, pitch, yaw = quaternion_to_carla_rotation(quaternion)
    return (
        (float(extrinsic.x), -float(extrinsic.y), float(extrinsic.z)),
        (roll, pitch, yaw)
    )
