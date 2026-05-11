"""Visibility filtering: drop boxes invisible to all cameras of a rig.

For each box and each camera, we project the 8 corners and ask whether the
**2D bounding box of the in-front projected corners overlaps the image
rectangle** ``[0, W] x [0, H]``. Using bbox-overlap rather than "any corner
inside the image" is important because large nearby boxes (e.g. a car a few
metres in front of the ego) routinely have all 8 corners falling outside the
image bounds even though their body fills the centre of the frame.

Boxes are passed in the rig's IMU/rear-axle frame — the same frame the camera
extrinsics ``camera_to_imu_se3`` are stored in.

Context:
- During PETR/BEVFormer training it makes only sense to train on boxes that are
visible in the camera, otherwise the model gets confused by trying to associate
image features to invisible boxes
- However, some randomly generated rigs have larger gaps, so we filter them out
right now
"""

from __future__ import annotations

import jaxtyping as jt
import numpy as np
import numpy.typing as npt
from beartype import beartype
from py123d.geometry import BoundingBoxSE3, PoseSE3
from py123d.geometry.utils.bounding_box_utils import \
    bbse3_array_to_corners_array

from lead.rig.rig_config import CameraEntry, RigConfig


@beartype
def _world_to_camera_matrix(
    camera_to_imu_se3: PoseSE3
) -> jt.Float[npt.NDArray, "4 4"]:
    """Inverse of the camera-to-IMU extrinsic matrix (i.e. world->camera)."""
    return np.linalg.inv(camera_to_imu_se3.transformation_matrix)


@beartype
def _project_corners(
    corners_imu: jt.Float[npt.NDArray, "N 8 3"],
    camera: CameraEntry,
) -> tuple[jt.Float[npt.NDArray, "N 8 2"], jt.Float[npt.NDArray, "N 8"]]:
    """Project corner points (in IMU/rear-axle frame) into a camera's image
    plane.

    Args:
        corners_imu: Box corners in the IMU frame; shape ``(N, 8, 3)``.
        camera: Camera entry with intrinsics and rear-axle extrinsic.

    Returns:
        Tuple ``(uv, depth)`` where ``uv`` is the projected pixel coordinates
        of shape ``(N, 8, 2)`` and ``depth`` is the per-corner depth (Z in
        camera frame, py123d/OpenCV pZmYpX convention) of shape ``(N, 8)``.
    """
    n = corners_imu.shape[0]
    if n == 0:
        return np.zeros((0, 8, 2)), np.zeros((0, 8))

    homogeneous = np.concatenate([corners_imu, np.ones((n, 8, 1))], axis=-1)
    world_to_cam = _world_to_camera_matrix(camera.camera_to_imu_se3)
    camera_frame = (homogeneous @ world_to_cam.T)[..., :3]

    fx, fy, cx, cy = camera.intrinsics.fx, camera.intrinsics.fy, camera.intrinsics.cx, camera.intrinsics.cy
    depth = camera_frame[..., 2]
    safe_depth = np.where(np.abs(depth) < 1e-6, 1e-6, depth)
    u = fx * camera_frame[..., 0] / safe_depth + cx
    v = fy * camera_frame[..., 1] / safe_depth + cy
    return np.stack([u, v], axis=-1), depth


@beartype
def boxes_visible_in_rig(
    boxes_in_imu: list[BoundingBoxSE3],
    rig: RigConfig,
    min_depth_m: float = 0.1,
    edge_margin_px: float = 16.0,
) -> jt.Bool[npt.NDArray, "N"]:
    """Return a boolean mask: True if a box is visible in any rig camera.

    Args:
        boxes_in_imu: Bounding boxes expressed in the IMU/rear-axle frame.
        rig: The rig whose cameras are used for the visibility check.
        min_depth_m: Corners closer than this in camera Z are treated as
            behind the camera (avoids divide-by-zero artefacts at the
            principal plane).
        edge_margin_px: Slack added to the image rectangle when testing
            overlap. Sub-pixel intrinsic wiggle and tick-to-tick ego
            motion can otherwise flip a box that grazes the FOV edge in
            and out of the visibility set between consecutive frames; the
            margin keeps marginal boxes consistently labelled.

    Returns:
        Boolean array of shape ``(len(boxes_in_imu),)``.
    """
    n = len(boxes_in_imu)
    if n == 0:
        return np.zeros((0,), dtype=bool)
    if not rig.cameras:
        return np.zeros((n,), dtype=bool)

    bbse3_array = np.stack([box.array for box in boxes_in_imu], axis=0)
    corners = bbse3_array_to_corners_array(bbse3_array)

    visible = np.zeros((n,), dtype=bool)
    for camera in rig.cameras:
        uv, depth = _project_corners(corners, camera)
        in_front = depth > min_depth_m  # (N, 8)
        any_in_front = in_front.any(axis=-1)  # (N,)

        # Mask behind-camera corners with NaN so they don't poison min/max.
        uv_safe = np.where(in_front[..., None], uv, np.nan)
        with np.errstate(invalid="ignore"):
            u_min = np.nanmin(uv_safe[..., 0], axis=-1)
            u_max = np.nanmax(uv_safe[..., 0], axis=-1)
            v_min = np.nanmin(uv_safe[..., 1], axis=-1)
            v_max = np.nanmax(uv_safe[..., 1], axis=-1)

        overlaps_image = (
            (u_max >= -edge_margin_px)
            & (u_min <= camera.width + edge_margin_px)
            & (v_max >= -edge_margin_px)
            & (v_min <= camera.height + edge_margin_px)
        )
        visible |= any_in_front & overlaps_image
    return visible


@beartype
def filter_boxes_by_rig(
    boxes_in_imu: list[BoundingBoxSE3],
    rig: RigConfig,
) -> list[BoundingBoxSE3]:
    """Return only the boxes visible to at least one camera in ``rig``."""
    mask = boxes_visible_in_rig(boxes_in_imu, rig)
    return [
        box for box, keep in zip(boxes_in_imu, mask, strict=True) if bool(keep)
    ]
