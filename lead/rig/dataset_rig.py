"""Convert real-dataset camera metadata into a :class:`RigConfig`.

Once a dataset has been parsed into py123d's Arrow log format (see
``py123d.parser.*``), every dataset exposes the same scene API regardless of
its source. The cheapest way to import a real rig is therefore to point this
module at one of those Arrow logs and pull
:meth:`SceneAPI.get_camera_metadatas`. That works uniformly for nuScenes,
Waymo, nuPlan, AV2, KITTI360, PandaSet, Physical-AI-AV, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict

from beartype import beartype
from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI
from py123d.datatypes import CameraID
from py123d.datatypes.sensors.pinhole_camera import PinholeCameraMetadata

from lead.rig.rig_config import CameraEntry, RigConfig, WiggleConfig


@beartype
def pinhole_metadata_to_rig(
    metadata: Dict[CameraID, PinholeCameraMetadata],
    rig_name: str,
    source: str = "",
    wiggle: WiggleConfig | None = None,
) -> RigConfig:
    """Wrap a py123d-parsed pinhole rig in a :class:`RigConfig`.

    Args:
        metadata: Mapping from camera id to py123d pinhole metadata.
        rig_name: Output rig name (e.g. ``"carla_nuscenes"``).
        source: Free-form provenance string (e.g. ``"nuscenes:scene_0061"``).
        wiggle: Optional wiggle config; defaults to enabled with seed=0.

    Returns:
        A :class:`RigConfig` with a :class:`CameraEntry` per source camera.
    """
    cameras = [
        CameraEntry(
            camera_id=camera_id,
            camera_name=meta.camera_name,
            width=meta.width,
            height=meta.height,
            intrinsics=meta.intrinsics,
            distortion=meta.distortion,
            camera_to_imu_se3=meta.camera_to_imu_se3,
            is_undistorted=meta.is_undistorted,
        )
        for camera_id, meta in metadata.items()
    ]
    return RigConfig(
        rig_name=rig_name,
        cameras=cameras,
        wiggle=wiggle if wiggle is not None else WiggleConfig(seed=0),
        source=source,
    )


@beartype
def extract_from_log_dir(
    log_dir: Path | str,
    rig_name: str | None = None,
    source: str | None = None,
) -> RigConfig:
    """Extract a :class:`RigConfig` from one py123d Arrow log directory.

    Works for any dataset already parsed into py123d format: nuScenes,
    Waymo, nuPlan, AV2, KITTI360, PandaSet, Physical-AI-AV, ncore. The
    log directory is the leaf folder containing ``*.arrow`` files (e.g.
    ``<py123d_data_root>/logs/<split>/<log_name>/``).

    Only pinhole cameras are exported — fisheye and f-theta entries (if
    any) are skipped because :class:`RigConfig` only models pinhole
    intrinsics today.

    Args:
        log_dir: Path to the leaf py123d log directory.
        rig_name: Output rig name. Defaults to ``carla_<dataset>_<location>``.
        source: Free-form provenance tag. Defaults to ``<dataset>:<log_name>``.

    Returns:
        A :class:`RigConfig` with the dataset's pinhole cameras.

    Raises:
        FileNotFoundError: ``log_dir`` does not exist.
        ValueError: The log has no pinhole cameras.
    """
    log_path = Path(log_dir)
    if not log_path.is_dir():
        raise FileNotFoundError(f"Log dir not found: {log_path}")

    api = ArrowSceneAPI(log_dir=log_path)
    log_meta = api.get_log_metadata()

    all_cameras = api.get_camera_metadatas()
    pinhole_cameras: Dict[CameraID, PinholeCameraMetadata] = {
        camera_id: meta
        for camera_id, meta in all_cameras.items()
        if isinstance(meta, PinholeCameraMetadata)
    }
    if not pinhole_cameras:
        raise ValueError(
            f"Log {log_path} has no pinhole cameras (found {len(all_cameras)} cameras "
            f"of types {sorted({type(m).__name__ for m in all_cameras.values()})})",
        )

    derived_name = rig_name or f"carla_{log_meta.dataset}_{log_meta.location}".lower()
    derived_source = source or f"{log_meta.dataset}:{log_meta.log_name}"

    return pinhole_metadata_to_rig(
        metadata=pinhole_cameras,
        rig_name=derived_name,
        source=derived_source,
    )


@beartype
def extract_from_callable(
    rig_name: str,
    source: str,
    fetch: Callable[[], Dict[CameraID, PinholeCameraMetadata]],
) -> RigConfig:
    """Generic adapter: call ``fetch`` to get camera metadata, wrap in a rig.

    Use only if you need to extract directly from a dataset SDK without
    first parsing it into py123d Arrow format. For everything else, use
    :func:`extract_from_log_dir`.
    """
    metadata = fetch()
    return pinhole_metadata_to_rig(metadata=metadata, rig_name=rig_name, source=source)
