"""Slot-based random rig generator.

py123d's :class:`CameraID` already encodes positional semantics (``PCAM_F0``
is front, ``PCAM_L0/L1/L2`` are left front-to-back, etc). We turn each ID
into an explicit **slot** — a CARLA-frame ``(x, y, z, yaw)`` range — and
sample as follows:

1. ``PCAM_F0`` is always included.
2. For every other slot, a Bernoulli draw with the slot's per-ID probability
   decides whether it is part of the rig (or, if the caller passes
   ``num_cameras=N``, ``N-1`` slots are drawn without replacement weighted
   by those probabilities).
3. Only after a slot has been chosen do we sample the camera's mount within
   that slot's ranges and the yaw within its yaw range.

This is what makes ``PCAM_R*`` always end up on the right side of the ego
and ``PCAM_*2`` always near the rear axle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from beartype import beartype
from py123d.datatypes import CameraID
from py123d.datatypes.sensors.pinhole_camera import PinholeIntrinsics

from lead.expert.expert_py123d_utils import get_carla_lincoln_mkz_2020_metadata
from lead.rig.coord_transform import (
    carla_camera_extrinsic_to_iso,
    iso_camera_extrinsic_to_carla,
)
from lead.rig.rig_config import CameraEntry, LidarEntry, RigConfig, WiggleConfig

DEFAULT_RANDOM_RIG_WIDTH = 800
DEFAULT_RANDOM_RIG_HEIGHT = 450
DEFAULT_RANDOM_RIG_FOV_DEG = 70.0
# Per-axis jitter applied to (width, height) when sampling per-camera resolution.
# Width × height is held within ±5% of the base area, so cameras vary in aspect
# ratio (e.g. 800×450 → 720×500 or 920×390) without blowing up the per-camera
# render budget.
DEFAULT_DIM_JITTER_FRAC = 0.20


@dataclass(frozen=True)
class Slot:
    """Allowed CARLA-frame mount, orientation, and intrinsic range for one
    :class:`CameraID`.

    Coordinates are in CARLA convention (X forward, Y right, Z up). ``yaw``
    and ``pitch`` are in CARLA degrees. ``yaw`` may wrap across ``±180``
    (used for the rear camera): if ``yaw_max > 180``, samples land in
    ``[yaw_min, yaw_max]`` and are wrapped into ``[-180, 180]`` on emit.
    Negative ``pitch`` tilts the camera downward (roof-mounted cams looking
    at the road); positive pitch tilts it up.

    ``fov_deg_range`` defines the allowed *horizontal* FOV in degrees from
    which we sample once per generated camera. Front cams tend to be narrower
    (tele-ish, ~50-70 deg) for distant detection; side and rear cams wider
    (~80-110 deg) for blind-spot coverage. ``cxcy_offset_px_max`` is the
    half-width of a uniform jitter applied to the principal point relative
    to the image centre (real cameras have small manufacturing offsets).
    """

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    yaw_range: tuple[float, float]
    pitch_range: tuple[float, float] = (-10.0, 5.0)
    fov_deg_range: tuple[float, float] = (60.0, 90.0)
    cxcy_offset_px_max: float = 5.0


# Lincoln MKZ 2020 (CARLA's default ego) bounding box, in CARLA frame:
# half-width = 0.92 m, half-length = 2.45 m, roof = 1.49 m. Slots are tightened
# so every sampled camera mount lies *outside* the body and oriented away from
# it, which keeps body panels (hood, trunk, fender, mirror) out of frame even
# under wiggle (1 cm translation, 0.5° rotation 1-sigma):
#   * roof slots (F0/B0): z_min = 1.6 (≥ 11 cm above roof)
#   * side slots (L*/R*): |y| ∈ [1.0, 1.2] (≥ 8 cm outside body half-width)
#   * L0/R0 yaw is pulled away from straight-forward so the front-side fender
#     stays outside the camera's horizontal FOV at the widest sampled hfov.
_DEFAULT_SLOTS: dict[CameraID, Slot] = {
    CameraID.PCAM_F0: Slot((1.4, 2.0), (-0.30, 0.30), (1.6, 2.0), (-15.0, 15.0), (-10.0, 5.0), (50.0, 75.0)),
    CameraID.PCAM_B0: Slot((-2.2, -1.6), (-0.30, 0.30), (1.6, 1.9), (165.0, 195.0), (-5.0, 10.0), (70.0, 110.0)),
    CameraID.PCAM_L0: Slot((0.8, 1.8), (-1.20, -1.00), (1.0, 1.8), (-80.0, -50.0), (-15.0, 5.0), (70.0, 100.0)),
    CameraID.PCAM_L1: Slot((-0.5, 0.8), (-1.20, -1.00), (1.4, 2.0), (-110.0, -70.0), (-10.0, 5.0), (80.0, 110.0)),
    CameraID.PCAM_L2: Slot((-1.8, -0.5), (-1.20, -1.00), (1.3, 1.9), (-160.0, -100.0), (-15.0, 5.0), (70.0, 100.0)),
    CameraID.PCAM_R0: Slot((0.8, 1.8), (1.00, 1.20), (1.0, 1.8), (50.0, 80.0), (-15.0, 5.0), (70.0, 100.0)),
    CameraID.PCAM_R1: Slot((-0.5, 0.8), (1.00, 1.20), (1.4, 2.0), (70.0, 110.0), (-10.0, 5.0), (80.0, 110.0)),
    CameraID.PCAM_R2: Slot((-1.8, -0.5), (1.00, 1.20), (1.3, 1.9), (100.0, 160.0), (-15.0, 5.0), (70.0, 100.0)),
}

_DEFAULT_INCLUSION_PROBS: dict[CameraID, float] = {
    CameraID.PCAM_F0: 1.0,
    CameraID.PCAM_L0: 0.85,
    CameraID.PCAM_R0: 0.85,
    CameraID.PCAM_L1: 0.65,
    CameraID.PCAM_R1: 0.65,
    CameraID.PCAM_B0: 0.55,
    CameraID.PCAM_L2: 0.40,
    CameraID.PCAM_R2: 0.40,
}


@beartype
def get_default_slots() -> dict[CameraID, Slot]:
    """Public read-only view of slot table — used by tests + visualisations."""
    return dict(_DEFAULT_SLOTS)


@beartype
def _intrinsics_for(
    width: int,
    height: int,
    fov_deg: float,
    cx_offset_px: float = 0.0,
    cy_offset_px: float = 0.0,
) -> PinholeIntrinsics:
    """CARLA-style pinhole intrinsics derived from horizontal FOV in degrees.

    ``cx_offset_px`` / ``cy_offset_px`` shift the principal point from the
    image centre — both default to 0 (centred) but the slot-based sampler
    feeds in small per-camera offsets to model real-world manufacturing
    tolerances.
    """
    focal = width / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
    cx = width / 2.0 + cx_offset_px
    cy = height / 2.0 + cy_offset_px
    return PinholeIntrinsics(fx=focal, fy=focal, cx=cx, cy=cy)


@beartype
def _wrap_carla_yaw(yaw_deg: float) -> float:
    """Wrap a CARLA yaw into the canonical ``[-180, 180]`` interval."""
    return ((yaw_deg + 180.0) % 360.0) - 180.0


@beartype
def _sample_dimensions(
    rng: np.random.Generator,
    base_width: int,
    base_height: int,
    jitter_frac: float = DEFAULT_DIM_JITTER_FRAC,
) -> tuple[int, int]:
    """Sample ``(width, height)`` keeping area constant within per-axis bounds.

    Each axis is held within ``base * (1 ± jitter_frac)``; total area is held
    constant at ``base_width * base_height``. The width range is intersected
    with the height range so the derived height also stays within the bounds.

    Args:
        rng: Source of randomness.
        base_width: Reference width in pixels.
        base_height: Reference height in pixels.
        jitter_frac: Half-width of the per-axis allowed range, as a fraction.

    Returns:
        A ``(width, height)`` integer pair with ``width × height ≈ base_area``.
    """
    base_area = base_width * base_height
    w_lo = max(
        base_width * (1.0 - jitter_frac),
        base_area / (base_height * (1.0 + jitter_frac)),
    )
    w_hi = min(
        base_width * (1.0 + jitter_frac),
        base_area / (base_height * (1.0 - jitter_frac)),
    )
    width = int(round(float(rng.uniform(w_lo, w_hi))))
    height = int(round(base_area / width))
    return width, height


@dataclass(frozen=True)
class _SampledIntrinsics:
    """Per-camera intrinsic parameters drawn within a slot's ranges."""

    fov_deg: float
    cx_offset_px: float
    cy_offset_px: float


@beartype
def _sample_in_slot(
    slot: Slot,
    rng: np.random.Generator,
) -> tuple[tuple[float, float, float], float, float, _SampledIntrinsics]:
    """Uniformly sample (mount, pitch, yaw, intrinsics) within ``slot``'s
    ranges.

    Returns:
        Tuple ``((x, y, z), pitch_deg, yaw_deg, intrinsics)`` where
        ``intrinsics`` carries the sampled FOV and principal-point offsets.
    """
    x = float(rng.uniform(*slot.x_range))
    y = float(rng.uniform(*slot.y_range))
    z = float(rng.uniform(*slot.z_range))
    pitch = float(rng.uniform(*slot.pitch_range))
    yaw = float(rng.uniform(*slot.yaw_range))
    fov = float(rng.uniform(*slot.fov_deg_range))
    cx_off = float(
        rng.uniform(-slot.cxcy_offset_px_max, slot.cxcy_offset_px_max)
    )
    cy_off = float(
        rng.uniform(-slot.cxcy_offset_px_max, slot.cxcy_offset_px_max)
    )
    return (
        (x, y, z),
        pitch,
        _wrap_carla_yaw(yaw),
        _SampledIntrinsics(
            fov_deg=fov, cx_offset_px=cx_off, cy_offset_px=cy_off
        ),
    )


@beartype
def _resolve_inclusion_probs(
    inclusion_probs: dict[CameraID, float] | None,
) -> dict[CameraID, float]:
    """Merge a partial override onto the default inclusion-probability table."""
    merged = dict(_DEFAULT_INCLUSION_PROBS)
    if inclusion_probs:
        for camera_id, prob in inclusion_probs.items():
            if camera_id not in _DEFAULT_SLOTS:
                raise ValueError(f"Unknown CameraID: {camera_id}")
            if not 0.0 <= prob <= 1.0:
                raise ValueError(
                    f"Probability for {camera_id} out of range: {prob}"
                )
            merged[camera_id] = prob
    return merged


@beartype
def _select_slots_bernoulli(
    rng: np.random.Generator,
    probs: dict[CameraID, float],
) -> list[CameraID]:
    """One Bernoulli draw per non-F0 slot; F0 is always selected."""
    selected: list[CameraID] = [CameraID.PCAM_F0]
    for camera_id, prob in probs.items():
        if camera_id == CameraID.PCAM_F0:
            continue
        if rng.random() < prob:
            selected.append(camera_id)
    return selected


@beartype
def _select_slots_with_target_count(
    rng: np.random.Generator,
    probs: dict[CameraID, float],
    num_cameras: int,
) -> list[CameraID]:
    """Pick exactly ``num_cameras`` slots: F0 always, the rest weighted-sampled
    w/o replacement."""
    if not 1 <= num_cameras <= len(_DEFAULT_SLOTS):
        raise ValueError(
            f"num_cameras must be in [1, {len(_DEFAULT_SLOTS)}],"
            f"got {num_cameras}",
        )
    selected: list[CameraID] = [CameraID.PCAM_F0]
    remaining_ids = [cid for cid in _DEFAULT_SLOTS if cid != CameraID.PCAM_F0]
    remaining_weights = np.array(
        [probs[cid] for cid in remaining_ids], dtype=float
    )

    needed = num_cameras - 1
    while needed > 0 and remaining_ids:
        if remaining_weights.sum() <= 0:
            chosen_idx = int(rng.integers(0, len(remaining_ids)))
        else:
            chosen_idx = int(
                rng.choice(
                    len(remaining_ids),
                    p=remaining_weights / remaining_weights.sum()
                ),
            )
        selected.append(remaining_ids.pop(chosen_idx))
        remaining_weights = np.delete(remaining_weights, chosen_idx)
        needed -= 1
    return selected


@beartype
def generate_random_rig(
    seed: int,
    num_cameras: int | None = None,
    inclusion_probs: dict[CameraID, float] | None = None,
    rig_name: str | None = None,
    width: int = DEFAULT_RANDOM_RIG_WIDTH,
    height: int = DEFAULT_RANDOM_RIG_HEIGHT,
    fov_deg: float | None = None,
    wiggle: WiggleConfig | None = None,
    lidars: Sequence[LidarEntry] = (),
) -> RigConfig:
    """Generate a slot-based random rig with per-camera intrinsic sampling.

    Args:
        seed: RNG seed for reproducibility.
        num_cameras: If ``None`` (default), each non-F0 slot is drawn with its
            Bernoulli probability. If an integer in ``[1, 8]``, the function
            returns a rig with exactly that many cameras (F0 + ``N-1`` slots
            drawn without replacement, weighted by the probabilities).
        inclusion_probs: Optional partial override of the default per-slot
            inclusion probabilities. Missing keys fall back to the defaults.
        rig_name: Optional override for the rig name.
        width: Reference image width in pixels. Each camera's actual width
            is sampled within ±20 % of this, with height adjusted so the
            total pixel count stays at ``width × height``.
        height: Reference image height in pixels (see ``width``).
        fov_deg: If provided, pins every camera's horizontal FOV to this
            value (legacy behaviour). If ``None`` (default), each camera
            samples FOV from its slot's :attr:`Slot.fov_deg_range` and a
            small principal-point offset from :attr:`Slot.cxcy_offset_px_max`.
        wiggle: Optional wiggle configuration.
        lidars: Optional list of lidars (CARLA-side spec).

    Returns:
        A :class:`RigConfig` whose cameras all satisfy slot-membership.

    Raises:
        ValueError: If ``num_cameras`` is out of range or ``inclusion_probs``
            contains an unknown ``CameraID`` or an out-of-range probability.
    """
    rng = np.random.default_rng(seed)
    probs = _resolve_inclusion_probs(inclusion_probs)
    ego_metadata = get_carla_lincoln_mkz_2020_metadata()

    if num_cameras is None:
        selected = _select_slots_bernoulli(rng, probs)
    else:
        selected = _select_slots_with_target_count(rng, probs, num_cameras)

    cameras: list[CameraEntry] = []
    for camera_id in selected:
        slot = _DEFAULT_SLOTS[camera_id]
        (pos, pitch_deg, yaw_deg, sampled_intrinsics) = _sample_in_slot(
            slot, rng
        )
        extrinsic = carla_camera_extrinsic_to_iso(
            camera_pos=list(pos),
            camera_rot_deg=[0.0, pitch_deg, yaw_deg],
            ego_metadata=ego_metadata,
        )
        camera_fov = fov_deg \
            if fov_deg is not None else sampled_intrinsics.fov_deg
        cam_width, cam_height = _sample_dimensions(rng, width, height)
        cameras.append(
            CameraEntry(
                camera_id=camera_id,
                camera_name=str(camera_id),
                width=cam_width,
                height=cam_height,
                intrinsics=_intrinsics_for(
                    width=cam_width,
                    height=cam_height,
                    fov_deg=camera_fov,
                    cx_offset_px=sampled_intrinsics.cx_offset_px,
                    cy_offset_px=sampled_intrinsics.cy_offset_px,
                ),
                camera_to_imu_se3=extrinsic,
            ),
        )

    suffix = f"_n{len(cameras)}" if num_cameras is None else f"_n{num_cameras}"
    return RigConfig(
        rig_name=rig_name or f"carla_random_seed{seed}{suffix}",
        cameras=cameras,
        lidars=list(lidars),
        wiggle=wiggle if wiggle is not None else WiggleConfig(seed=seed),
        source=f"random:seed={seed},n={len(cameras)},fov={fov_deg}",
    )


@beartype
def _carla_pose_for_camera(
    camera: CameraEntry
) -> tuple[float, float, float, float, float]:
    """Recover CARLA-frame ``(x, y, z, pitch_deg, yaw_deg)`` from a stored
    entry."""
    ego_metadata = get_carla_lincoln_mkz_2020_metadata()
    (pos, rot_deg) = iso_camera_extrinsic_to_carla(
        camera.camera_to_imu_se3, ego_metadata
    )
    pitch = rot_deg[1]
    yaw = _wrap_carla_yaw(rot_deg[2])
    return pos[0], pos[1], pos[2], pitch, yaw


@beartype
def _yaw_in_slot(yaw_deg: float, slot: Slot) -> bool:
    """True if ``yaw_deg`` is inside ``slot.yaw_range``
    (with +/-180 wrap support)."""
    low, high = slot.yaw_range
    if high <= 180.0:
        return low <= yaw_deg <= high
    # Wrap range: e.g. [165, 195] is equivalent to [165, 180] ∪ [-180, -165].
    high_wrapped = high - 360.0
    return (low <= yaw_deg <= 180.0) or (-180.0 <= yaw_deg <= high_wrapped)


@beartype
def is_valid(
    rig: RigConfig,
    tolerance_m: float = 1e-6,
    tolerance_deg: float = 1e-6
) -> tuple[bool, str]:
    """Check that every camera's pose lies inside its :class:`CameraID` slot.

    Args:
        rig: The rig to validate.
        tolerance_m: Slack added to position bounds to absorb rounding noise.
        tolerance_deg: Slack added to yaw bounds to absorb rounding noise.

    Returns:
        ``(True, "")`` on success, ``(False, reason)`` otherwise.
    """
    if not rig.cameras:
        return False, "rig has no cameras"

    has_f0 = any(camera.camera_id == CameraID.PCAM_F0 for camera in rig.cameras)
    if not has_f0:
        return False, "PCAM_F0 missing"

    for camera in rig.cameras:
        slot = _DEFAULT_SLOTS.get(camera.camera_id)
        if slot is None:
            return False, f"{camera.camera_id} has no defined slot"

        x, y, _z, pitch_deg, yaw_deg = _carla_pose_for_camera(camera)
        if not (slot.x_range[0] - tolerance_m <= x <= slot.x_range[1] + tolerance_m):
            return False, f"{camera.camera_id} x={x:.3f} outside {slot.x_range}"
        if not (slot.y_range[0] - tolerance_m <= y <= slot.y_range[1] + tolerance_m):
            return False, f"{camera.camera_id} y={y:.3f} outside {slot.y_range}"
        if not _yaw_in_slot(yaw_deg, slot):
            return False, f"{camera.camera_id} yaw={yaw_deg:.2f} outside {slot.yaw_range}"
        if not (slot.pitch_range[0] - tolerance_deg <= pitch_deg <= slot.pitch_range[1] + tolerance_deg):
            return False, f"{camera.camera_id} pitch={pitch_deg:.2f} outside {slot.pitch_range}"

    return True, ""
