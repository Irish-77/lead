"""Multi-rig variant of :class:`ExpertPy123D`.

Collects data simultaneously for several camera rigs in a single CARLA run.
Each rig gets its own :class:`ArrowLogWriter`, its own per-frame frustum
filter, and (optionally) per-frame extrinsic wiggle. The driving logic and
GT extraction are inherited from :class:`ExpertPy123D` unchanged.

Rigs are loaded from JSON files listed in the ``LEAD_MULTI_RIG_CONFIGS``
environment variable (semicolon-separated paths). The output base directory
defaults to ``data/carla_multi_rig_py123d`` and can be overridden via
``LEAD_MULTI_RIG_OUTPUT_BASE``.

The full-cloud lidar save is disabled: only
``BoxDetectionAttributes.num_lidar_points`` is recorded (the lidar still
runs so CARLA can populate that field, but its point cloud is never written
to disk).
"""

from __future__ import annotations

import logging
import math
import os
import typing
from pathlib import Path

import carla
import numpy as np
from beartype import beartype
from py123d.api.scene.arrow.arrow_log_writer import ArrowLogWriter
from py123d.api.scene.arrow.utils.log_writer_config import LogWriterConfig
from py123d.datatypes import (
    BoxDetectionsSE3,
    Camera,
    CameraID,
    EgoStateSE3,
    Lidar,
    LogMetadata,
    MapMetadata,
    PinholeCameraMetadata,
    Timestamp,
)
from py123d.datatypes.sensors.pinhole_camera import PinholeIntrinsics
from py123d.geometry import PoseSE3
from py123d.geometry.transform import rel_to_abs_se3
from py123d.parser.base_dataset_parser import ModalitiesSync

from lead.common.logging_config import setup_logging
from lead.expert.expert import Expert
from lead.expert.expert_py123d import ExpertPy123D
from lead.rig import serialization
from lead.rig.coord_transform import (
    carla_camera_extrinsic_to_iso,
    iso_camera_extrinsic_to_carla,
)
from lead.rig.frustum_filter import boxes_visible_in_rig
from lead.rig.rig_config import RigConfig
from lead.rig.wiggle import sample_wiggle_offset, wiggled_transform
from py123d.geometry import BoundingBoxSE3
from py123d.geometry.transform import abs_to_rel_se3_array

setup_logging()
LOG = logging.getLogger(__name__)


def get_entry_point() -> str:
    return "ExpertMultiRigPy123D"


@beartype
def _load_rigs_from_env() -> list[RigConfig]:
    """Read ``LEAD_MULTI_RIG_CONFIGS`` (semicolon-separated paths)."""
    raw = os.environ.get("LEAD_MULTI_RIG_CONFIGS", "")
    if not raw.strip():
        raise RuntimeError(
            "LEAD_MULTI_RIG_CONFIGS env var is required for multi-rig agent.",
        )
    paths = [Path(p.strip()) for p in raw.split(";") if p.strip()]
    rigs = [serialization.load(path) for path in paths]
    seen: set[str] = set()
    for rig in rigs:
        if rig.rig_name in seen:
            raise RuntimeError(f"Duplicate rig_name: {rig.rig_name}")
        seen.add(rig.rig_name)
    return rigs


@beartype
def _sensor_id(rig_idx: int, camera_id: CameraID) -> str:
    return f"r{rig_idx}_{camera_id.name.lower()}"


@beartype
def _world_boxes_to_imu(
    box_detections,
    ego_imu_world: PoseSE3,
) -> list[BoundingBoxSE3]:
    """Re-express each box's centre pose in the ego's IMU/rear-axle frame.

    LEAD's :func:`_extract_py123d_box_detections` returns boxes whose
    ``center_se3`` is in py123d global coordinates. The frustum filter projects
    boxes through ``camera.camera_to_imu_se3``, which lives in the IMU frame,
    so we have to transform the boxes from world to IMU before checking
    visibility.
    """
    if not box_detections:
        return []
    pose_array = np.stack(
        [box.bounding_box_se3.center_se3.array for box in box_detections],
        axis=0
    )
    rel_array = abs_to_rel_se3_array(ego_imu_world, pose_array)
    return [
        BoundingBoxSE3(
            center_se3=PoseSE3.from_array(rel_array[i]),
            length=box.bounding_box_se3.length,
            width=box.bounding_box_se3.width,
            height=box.bounding_box_se3.height,
        )
        for i, box in enumerate(box_detections)
    ]


class ExpertMultiRigPy123D(ExpertPy123D):
    """Expert agent that writes one Arrow log per rig in a single CARLA run."""

    @beartype
    def setup(
        self,
        path_to_conf_file: str,
        route_index: str | None = None,
        traffic_manager: carla.TrafficManager | None = None,
    ) -> None:
        self._rigs: list[RigConfig] = _load_rigs_from_env()
        self._output_base: Path = Path(
            os.environ.get(
                "LEAD_MULTI_RIG_OUTPUT_BASE",
                str(Path("data") / "carla_multi_rig_py123d"),
            ),
        )
        self._wiggle_rngs: dict[str, np.random.Generator] = {
            rig.rig_name: np.random.default_rng(rig.wiggle.seed) \
            for rig in self._rigs
        }
        # Each rig camera's local mount in CARLA frame, derived once from the
        # rig config. Wiggle is applied as small offsets ON TOP of this — we
        # never rely on actor.get_transform() (which returns world coords for
        # ego-attached sensors and would not match what set_transform expects).
        self._base_mount_transforms: dict[tuple[str, CameraID], carla.Transform] = {}
        # Last-applied carla.Transform per camera; used to recompute the
        # py123d extrinsic for the next written frame.
        self._wiggled_transforms: dict[tuple[str, CameraID], carla.Transform] = {}
        # Last-applied intrinsics per camera (CARLA can't change FOV per
        # tick, so the jitter is metadata-only — kept tiny by default).
        self._wiggled_intrinsics: dict[tuple[str, CameraID], PinholeIntrinsics] = {}
        self._sensor_handles: dict[tuple[str, CameraID], carla.Actor] = {}
        self._handles_resolved: bool = False

        super().setup(path_to_conf_file, route_index, traffic_manager)

    @beartype
    def _init_py123d_log(self) -> None:
        """Build a separate :class:`ArrowLogWriter` per rig and reset each.

        We deliberately skip the parent's :meth:`_init_py123d_log` to avoid
        creating a redundant single-rig writer at LEAD's default path. We also
        skip building :attr:`_camera_metadatas` / :attr:`_lidar_metadata`
        because our extraction methods use the per-rig variants instead.
        """
        self._py123d_writers: dict[str, ArrowLogWriter] = {}
        self._camera_metadatas_per_rig: dict[str, dict[CameraID, PinholeCameraMetadata]] = {}

        for rig in self._rigs:
            rig_logs_root = self._output_base / rig.rig_name / "logs"
            rig_sensors_root = self._output_base / rig.rig_name / "sensors"
            rig_logs_root.mkdir(parents=True, exist_ok=True)
            rig_sensors_root.mkdir(parents=True, exist_ok=True)
            writer = ArrowLogWriter(
                log_writer_config=LogWriterConfig(
                    force_log_conversion=True,
                    camera_store_option="jpeg_binary",
                    lidar_store_option="binary",
                    lidar_codec="laz",
                ),
                logs_root=rig_logs_root,
                sensors_root=rig_sensors_root,
            )
            log_metadata = LogMetadata(
                dataset=self.config_expert.py123d_dataset,
                split=(
                    f"{self.config_expert.py123d_dataset}_"
                    f"{self.config_expert.py123d_split}",
                ),
                log_name=self._log_name,
                location=self._location,
                map_metadata=MapMetadata(
                    dataset=self.config_expert.py123d_dataset,
                    location=self._location,
                    map_has_z=True,
                    map_is_per_log=False,
                ),
            )
            writer.reset(log_metadata)
            self._py123d_writers[rig.rig_name] = writer
            # last-rig writer also satisfies parent's destroy
            self._py123d_log_writer = writer

            metadatas: dict[CameraID, PinholeCameraMetadata] = {}
            for camera in rig.cameras:
                metadatas[camera.camera_id] = PinholeCameraMetadata(
                    camera_name=camera.camera_name,
                    camera_id=camera.camera_id,
                    intrinsics=camera.intrinsics,
                    distortion=camera.distortion,
                    width=camera.width,
                    height=camera.height,
                    camera_to_imu_se3=camera.camera_to_imu_se3,
                    is_undistorted=camera.is_undistorted,
                )
            self._camera_metadatas_per_rig[rig.rig_name] = metadatas

            LOG.info(
                f"Rig '{rig.rig_name}' initialized with {len(rig.cameras)} "
                f"cameras -> {rig_logs_root.absolute()}",
            )

    def sensors(self) -> list[dict]:
        """LEAD's preset rig (so its tick() pipeline doesn't break) + our rig
        cameras.

        The LEAD-preset cameras are a wasted GPU pass but their data is
        consumed by inherited methods (camera-PC, semantics, depth, BEV ops).
        Choose ``target_dataset=3`` (single-cam preset) at the env level to
        keep that overhead minimal.
        """
        from lead.common.sensor_setup import av_sensor_setup

        result: list[dict] = av_sensor_setup(
            self.config_expert,
            perturbation_rotation=0.0,
            perturbation_translation=0.0,
            lidar=True,
            perturbate=False,
            sensor_agent=False,
            radar=False,
        )

        for rig_idx, rig in enumerate(self._rigs):
            for camera in rig.cameras:
                (pos, rot_deg) = iso_camera_extrinsic_to_carla(
                    extrinsic=camera.camera_to_imu_se3,
                    ego_metadata=self._ego_metadata,
                )
                fov_deg = float(
                    2.0 * math.degrees(
                        math.atan(camera.width / (2.0 * camera.intrinsics.fx))
                    )
                )
                result.append(
                    {
                        "type": "sensor.camera.rgb",
                        "x": pos[0],
                        "y": pos[1],
                        "z": pos[2],
                        "roll": rot_deg[0],
                        "pitch": rot_deg[1],
                        "yaw": rot_deg[2],
                        "width": int(camera.width),
                        "height": int(camera.height),
                        "fov": fov_deg,
                        "id": _sensor_id(rig_idx, camera.camera_id),
                    },
                )
        return result

    @beartype
    def _resolve_sensor_handles(self) -> None:
        """Match spawned RGB sensors to our (rig, camera_id) pairs by spawn
        order.

        The leaderboard wrapper does not propagate our sensor ``id`` to the
        CARLA blueprint's ``role_name``, so we cannot look up by name. Instead
        we rely on the deterministic spawn order matching the order returned
        by :meth:`sensors`.
        """
        if self._handles_resolved:
            return
        world = self.carla_world
        rgb_sensors = list(world.get_actors().filter("sensor.camera.rgb"))
        ego_id = self.ego_vehicle.id
        ours = sorted(
            [s for s in rgb_sensors if s.parent is not None 
            and s.parent.id == ego_id],
            key=lambda actor: actor.id,
        )

        expected: list[tuple[str, CameraID]] = []
        for rig_idx, rig in enumerate(self._rigs):
            for camera in rig.cameras:
                expected.append((rig.rig_name, camera.camera_id))

        # LEAD's preset RGB cameras are spawned first (and a perturbated copy
        # if perturbate_sensors is on); our rig cameras come last. Take the
        # tail-N entries so we ignore the preset.
        if len(ours) < len(expected):
            LOG.warning(
                f"Expected at least {len(expected)} ego-attached RGB sensors, "
                f"found {len(ours)}; wiggle disabled.",
            )
            self._handles_resolved = True
            return
        rig_actors = ours[-len(expected):]

        for (rig_name, camera_id), actor in zip(
            expected, rig_actors, strict=True
        ):
            self._sensor_handles[(rig_name, camera_id)] = actor

        # Cache the per-camera local mount in CARLA frame, derived from the
        # rig config. We never rely on actor.get_transform() because for
        # ego-attached actors that returns world coords, not the local mount.
        for rig in self._rigs:
            for camera in rig.cameras:
                (pos, rot_deg) = iso_camera_extrinsic_to_carla(
                    extrinsic=camera.camera_to_imu_se3,
                    ego_metadata=self._ego_metadata,
                )
                self._base_mount_transforms[
                    (rig.rig_name, camera.camera_id)
                ] = carla.Transform(
                    carla.Location(x=pos[0], y=pos[1], z=pos[2]),
                    carla.Rotation(
                        roll=rot_deg[0], pitch=rot_deg[1], yaw=rot_deg[2]
                    ),
                )

        LOG.info(
            f"Resolved {len(self._sensor_handles)} sensor handles for wiggle."
        )
        self._handles_resolved = True

    @beartype
    def _apply_wiggle(self) -> None:
        """Sample and apply per-frame extrinsic + intrinsic wiggle.

        Extrinsic: the base is the CARLA-frame local mount derived from the
        rig config. ``actor.set_transform`` for an ego-attached sensor
        consumes a transform in the parent's (ego's) local frame, so we feed
        it ``base_local + offset`` directly. The recorded extrinsic for the
        next written frame is read from :attr:`_wiggled_transforms`.

        Intrinsic: CARLA blueprints can't change FOV per tick, so we instead
        add small Gaussian jitter to ``fx``/``fy``/``cx``/``cy`` in the
        recorded :class:`PinholeCameraMetadata` for each frame. Sigmas are
        sub-pixel by default — enough to break numerical determinism in
        downstream evaluation, not enough to introduce a measurable
        calibration mismatch with the rendered image.
        """
        self._resolve_sensor_handles()

        for rig in self._rigs:
            rng = self._wiggle_rngs[rig.rig_name]
            for camera in rig.cameras:
                key = (rig.rig_name, camera.camera_id)
                actor = self._sensor_handles.get(key)
                base = self._base_mount_transforms.get(key)
                if actor is None or base is None:
                    continue

                if not rig.wiggle.enabled:
                    self._wiggled_transforms[key] = base
                    self._wiggled_intrinsics[key] = camera.intrinsics
                    continue

                t_offset, r_offset = sample_wiggle_offset(rig.wiggle, rng)
                wiggled = wiggled_transform(base, t_offset, r_offset)
                actor.set_transform(wiggled)
                self._wiggled_transforms[key] = wiggled

                base_intrinsics = camera.intrinsics
                fx_jitter = float(
                    rng.normal(0.0, rig.wiggle.intrinsic_focal_sigma_px)
                )
                fy_jitter = float(
                    rng.normal(0.0, rig.wiggle.intrinsic_focal_sigma_px)
                )
                cx_jitter = float(
                    rng.normal(0.0, rig.wiggle.intrinsic_cxcy_sigma_px)
                )
                cy_jitter = float(
                    rng.normal(0.0, rig.wiggle.intrinsic_cxcy_sigma_px)
                )
                self._wiggled_intrinsics[key] = PinholeIntrinsics(
                    fx=base_intrinsics.fx + fx_jitter,
                    fy=base_intrinsics.fy + fy_jitter,
                    cx=base_intrinsics.cx + cx_jitter,
                    cy=base_intrinsics.cy + cy_jitter,
                )

    @beartype
    def _ego_is_unstable(
        self,
        vz_threshold_mps: float = 1.0,
        roll_pitch_threshold_deg: float = 20.0
    ) -> bool:
        """True if the ego is free-falling or tumbling.

        Some Town12 routes spawn the ego with a Z that doesn't match the
        actual road surface; CARLA physics drops the vehicle, sometimes
        rolling it on the way down. Cameras attached to a falling/tumbling
        ego produce useless frames (sky-only when the car is below the
        world, smeared sideways views during the tumble), so we skip the
        save instead of writing garbage to the Arrow log.
        """
        try:
            velocity = self.ego_vehicle.get_velocity()
            transform = self.ego_vehicle.get_transform()
        except (AttributeError, RuntimeError):
            return False
        if abs(velocity.z) > vz_threshold_mps:
            return True
        rotation = transform.rotation
        if abs(rotation.roll) > roll_pitch_threshold_deg:
            return True
        # Pitch can be moderately negative on hills; only reject extreme tilts.
        if abs(rotation.pitch) > roll_pitch_threshold_deg:
            return True
        return False

    @beartype
    def _camera_frame_is_uninitialized(
        self,
        rig_idx: int,
        rig: RigConfig,
        input_data: dict,
        mean_threshold: float = 10.0,
        std_threshold: float = 20.0,
    ) -> bool:
        """True if any of this rig's RGB sensors returned a degenerate buffer.

        We rule a frame out if it fails *either* test:

        - ``mean(rgb) < mean_threshold`` — almost-black frames where ~99%
          of pixels are pure zero with a handful of stray spikes (max
          ≈ 50–130). Plain ``max`` won't catch those.
        - ``std(rgb) < std_threshold`` — flat / uniform frames (mean ≈ 180
          but std ≈ 10), which CARLA sometimes hands back when its render
          target wasn't fully populated. Real CARLA daylight scenes have
          std ≈ 50–70; even an overcast or dim scene clears 25.

        Combined cut keeps real footage and rejects "GPU never wrote pixels
        here" buffers regardless of their brightness.
        """
        for camera in rig.cameras:
            sensor_id = _sensor_id(rig_idx, camera.camera_id)
            frame = input_data.get(sensor_id)
            if frame is None:
                continue
            bgra = frame[1] if isinstance(frame, tuple) else frame
            try:
                rgb = bgra[..., :3]
                if float(rgb.mean()) < mean_threshold:
                    return True
                if float(rgb.std()) < std_threshold:
                    return True
            except (ValueError, AttributeError):
                continue
        return False

    @beartype
    def run_step(
        self,
        input_data: dict,
        timestamp: float,
        sensors: list[list[str | typing.Any]] | None,
    ) -> carla.VehicleControl:
        """Per-tick: drive (parent), write per-rig sync, then wiggle for
        next tick."""
        control = super(ExpertPy123D, self).run_step(
            input_data, timestamp, sensors
        )

        save_due = (
            self.config_expert.datagen
            and self.step % self.config_expert.py123d_save_interval == 0
        )
        if save_due and self._ego_is_unstable():
            if self.step % self.config_expert.py123d_log_interval == 0:
                v = self.ego_vehicle.get_velocity()
                r = self.ego_vehicle.get_transform().rotation
                LOG.warning(
                    f"Skipping save at step {self.step}: ego unstable "
                    f"(vz={v.z:.2f} m/s, roll={r.roll:.1f}, "
                    f"pitch={r.pitch:.1f})",
                )
        if save_due and not self._ego_is_unstable():
            # Per-rig black-frame gating: a rig with degenerate cameras at this
            # tick is skipped on its own, but other rigs keep saving. The
            # earlier all-or-nothing version dropped ~38 % of multi-rig saves
            # to iter 0 because at least one rig out of six tended to have a
            # bad render.
            ts = Timestamp.from_s(timestamp)
            ego_state = self._extract_py123d_ego_state(ts)
            boxes_full = self._extract_py123d_box_detections(input_data, ts)
            boxes_imu = _world_boxes_to_imu(
                boxes_full.box_detections, ego_state.imu_se3
            )

            saved_rigs: list[str] = []
            skipped_rigs: list[str] = []
            for rig_idx, rig in enumerate(self._rigs):
                if self._camera_frame_is_uninitialized(
                    rig_idx, rig, input_data
                ):
                    skipped_rigs.append(rig.rig_name)
                    continue
                cameras_for_rig = self._extract_per_rig_cameras(
                    rig, input_data, ts, ego_state
                )
                visible_mask = boxes_visible_in_rig(boxes_imu, rig)
                kept_box_detections = [
                    box
                    for box, keep in zip(
                        boxes_full.box_detections, visible_mask, strict=True
                    )
                    if bool(keep)
                ]
                rig_boxes = BoxDetectionsSE3(
                    box_detections=kept_box_detections,
                    timestamp=ts,
                    metadata=self._box_detections_metadata,
                )
                modalities = [ego_state, rig_boxes, *cameras_for_rig]
                modalities.append(self._extract_py123d_traffic_lights(ts))
                self._py123d_writers[rig.rig_name].write_sync(
                    ModalitiesSync(timestamp=ts, modalities=modalities),
                )
                saved_rigs.append(rig.rig_name)

            if self.step % self.config_expert.py123d_log_interval == 0:
                if skipped_rigs:
                    LOG.warning(
                        f"Step {self.step} (t={timestamp:.2f}s): "
                        f"saved {len(saved_rigs)} rig(s), "
                        f"skipped {len(skipped_rigs)} for black/flat frames "
                        f"({skipped_rigs})",
                    )
                else:
                    LOG.info(
                        f"Saved multi-rig data at step {self.step} "
                        f"(timestamp={timestamp:.2f}s) "
                        f"for {len(self._rigs)} rig(s)",
                    )

        self._apply_wiggle()
        return control

    @beartype
    def _extract_per_rig_cameras(
        self,
        rig: RigConfig,
        input_data: dict,
        timestamp: Timestamp,
        ego_state: EgoStateSE3,
    ) -> list[Camera]:
        """Extract camera frames for one rig from CARLA's per-tick input dict"""
        rig_idx = self._rigs.index(rig)
        cameras: list[Camera] = []
        for camera in rig.cameras:
            sensor_id = _sensor_id(rig_idx, camera.camera_id)
            frame = input_data.get(sensor_id)
            if frame is None:
                LOG.warning(
                    f"Missing frame for sensor {sensor_id}"
                    f"at step {self.step}"
                )
                continue
            bgra = frame[1] if isinstance(frame, tuple) else frame
            rgb = bgra[:, :, :3][:, :, ::-1].copy()

            wiggled = self._wiggled_transforms.get(
                (rig.rig_name, camera.camera_id)
            )
            if wiggled is not None:
                pos = (
                    wiggled.location.x, wiggled.location.y, wiggled.location.z
                )
                rot = (
                    wiggled.rotation.roll,
                    wiggled.rotation.pitch,
                    wiggled.rotation.yaw
                )
                camera_to_imu_se3 = carla_camera_extrinsic_to_iso(
                    list(pos), list(rot), self._ego_metadata
                )
            else:
                camera_to_imu_se3 = camera.camera_to_imu_se3

            metadata = self._camera_metadatas_per_rig[rig.rig_name][
                camera.camera_id
            ]
            wiggled_intrinsics = self._wiggled_intrinsics.get(
                (rig.rig_name, camera.camera_id),
                metadata.intrinsics,
            )
            metadata = PinholeCameraMetadata(
                camera_name=metadata.camera_name,
                camera_id=metadata.camera_id,
                intrinsics=wiggled_intrinsics,
                distortion=metadata.distortion,
                width=metadata.width,
                height=metadata.height,
                camera_to_imu_se3=camera_to_imu_se3,
                is_undistorted=metadata.is_undistorted,
            )
            camera_to_global = rel_to_abs_se3(
                origin=ego_state.imu_se3,
                pose_se3=camera_to_imu_se3,
            )
            cameras.append(
                Camera(
                    metadata=metadata,
                    image=rgb,
                    camera_to_global_se3=camera_to_global,
                    timestamp=timestamp,
                ),
            )
        return cameras

    @beartype
    def _extract_py123d_lidar_data(self, timestamp: Timestamp) -> Lidar | None:
        """Disable lidar point-cloud save in multi-rig mode
        (we keep ``num_points`` only)."""
        return None

    @beartype
    def destroy(self, results: typing.Any = None) -> None:
        for rig_name, writer in getattr(self, "_py123d_writers", {}).items():
            try:
                writer.close()
                LOG.info(f"Closed multi-rig writer for {rig_name}")
            except Exception as e:  # noqa: BLE001
                LOG.error(f"Failed to close writer for {rig_name}: {e}")
        # Skip ExpertPy123D.destroy (which would re-close the now-finalized last
        # rig writer); jump straight to Expert.destroy.
        super(ExpertPy123D, self).destroy(results)
        LOG.info("Multi-rig cleanup complete - data saved to Py123D format")
