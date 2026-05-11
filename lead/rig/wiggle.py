"""Per-frame extrinsic jitter for CARLA sensors.

Each tick we sample a small rigid perturbation and physically move the sensor
on the actor (via :py:meth:`carla.Sensor.set_transform`). The rendered frame
and the recorded extrinsic both reflect the wiggled pose.
"""

from __future__ import annotations

import math

import carla
import numpy as np
from beartype import beartype

from lead.rig.rig_config import WiggleConfig


@beartype
def sample_wiggle_offset(
    wiggle: WiggleConfig,
    rng: np.random.Generator,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Sample a translation/rotation offset from a wiggle config.

    Args:
        wiggle: Per-rig wiggle parameters.
        rng: Source of randomness.

    Returns:
        Tuple ``((dx, dy, dz), (droll, dpitch, dyaw))`` with translation in
        meters (CARLA frame) and rotation deltas in degrees.
    """
    if not wiggle.enabled:
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    sigma_t = wiggle.translation_sigma_m
    sigma_r = wiggle.rotation_sigma_deg
    dx, dy, dz = (float(v) for v in rng.normal(0.0, sigma_t, size=3))
    droll, dpitch, dyaw = (float(v) for v in rng.normal(0.0, sigma_r, size=3))
    return (dx, dy, dz), (droll, dpitch, dyaw)


@beartype
def wiggled_transform(
    base_transform: carla.Transform,
    translation_offset: tuple[float, float, float],
    rotation_offset_deg: tuple[float, float, float],
) -> carla.Transform:
    """Apply a small rigid offset (in vehicle body frame) to a CARLA transform.

    The offset is small (~cm / sub-degree), so we treat the rotation as a
    simple Euler addition rather than constructing a full rotation matrix.
    """
    location = base_transform.location
    rotation = base_transform.rotation

    yaw_rad = math.radians(rotation.yaw)
    cy, sy = math.cos(yaw_rad), math.sin(yaw_rad)
    dx_world = cy * translation_offset[0] - sy * translation_offset[1]
    dy_world = sy * translation_offset[0] + cy * translation_offset[1]
    dz_world = translation_offset[2]

    return carla.Transform(
        carla.Location(
            x=location.x + dx_world,
            y=location.y + dy_world,
            z=location.z + dz_world,
        ),
        carla.Rotation(
            roll=rotation.roll + rotation_offset_deg[0],
            pitch=rotation.pitch + rotation_offset_deg[1],
            yaw=rotation.yaw + rotation_offset_deg[2],
        ),
    )
