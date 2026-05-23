"""Lift real-rig cameras that would be rendered inside the CARLA ego mesh.

Real-dataset rigs (nuScenes, pandaset, nuPlan, etc.) carry the camera positions
of their original physical vehicle. Some of those mount positions fall inside
CARLA's Lincoln MKZ ego body, so the per-rig black/flat-frame gate in
:class:`ExpertMultiRigPy123D` rejects every save after step 0 — even though
the rig itself is healthy in the real world.

This script reads each rig JSON, detects cameras whose
``camera_to_imu_se3`` translation lands inside a conservative ego
bounding box (in IMU/rear-axle ISO 8855 frame), and **lifts them
vertically** to ``LIFT_Z`` so they sit safely above the roof. The
applied offset is stored per camera in a new ``carla_camera_offset``
field so the original digital-twin position can be recovered at
training time by subtracting it.

Choosing vertical lift (vs. horizontal nudge) preserves the camera's
view direction relative to the ego: a camera lifted straight up still
looks at the same headings it would have, just from slightly higher.
For the real rigs in scope (nuScenes / pandaset / nuPlan), the cameras
that fall inside the box are concentrated near the roof line already,
so the lift is small (typically 5–50 cm) and the data shift is minimal.

Usage::

    python scripts/lift_cameras_outside_ego.py
    python scripts/lift_cameras_outside_ego.py \\
        --src rigs/real_rigs_small --dst rigs/real_rigs_small_carla
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# Lincoln MKZ 2020 bounding box in IMU/rear-axle frame
# (x=forward, y=left, z=up; rear-axle on the ground at the origin).
#
# We only lift cameras that fall in the **roof zone** of the ego mesh
# (z roughly 1.3-1.6 m), not the entire body. Cameras at hood level
# (z < 1.3) are typically angled forward and see the road through the
# windshield — they're working as intended and lifting them above the
# roof actually breaks them, because a wide-FOV camera looking forward
# from above the roof ends up framing the hood (a uniform-painted
# surface) and tripping the gate. Cameras at z > 1.6 are already above
# the roof and don't need lifting.
EGO_X_MIN = -1.2      # rear bumper ~ -1.0 m past rear axle, with margin
EGO_X_MAX = +3.5      # front bumper ~ +3.0 m past rear axle, with margin
EGO_Y_HALF = +1.05    # half-width ~ 0.93 m, with margin
# Lift cameras inside the full body height, not just the roof zone.
# The earlier z_min=1.3 cutoff was meant to avoid over-lifting hood-mounted
# cameras (av2 stereo at z=1.20), but those are dropped via CAMERAS_TO_DROP.
# Cameras at low z (e.g., kitti360 at z=0.65) translate to inside-the-cabin
# CARLA positions and would otherwise render the steering wheel / dashboard.
EGO_Z_MIN = 0.0       # entire body vertical extent
EGO_Z_MAX = +1.60     # roof height ~ 1.49 m, with margin

# Where to place a lifted camera in IMU frame. 1.8 m is just above the
# MKZ roof (1.49 m) so the FOV cone barely changes — small data shift,
# small risk of catching the hood.
LIFT_Z = 1.8

# Cameras to drop entirely before lifting. Maps rig filename (without
# .json) to the list of `camera_name` values to remove. The dropped
# cameras don't appear in the resulting rig at all (no records, no
# sensor spawns), so a stuck camera can't bottleneck the per-rig
# black/flat gate. Use sparingly — dropped cameras are lost from the
# digital twin and can't be recovered downstream.
CAMERAS_TO_DROP: dict[str, list[str]] = {
    # av2 stereo pair is redundant with the ring cameras for our use
    # case (the ring already covers the front). Dropping them removes
    # two narrow-FOV hood-zone cameras from the rig.
    "av2-sensor_train": ["stereo_front_left", "stereo_front_right"],
}


def is_inside_ego(x: float, y: float, z: float) -> bool:
    """True if the (x, y, z) IMU-frame position falls inside the ego box."""
    return (
        EGO_X_MIN <= x <= EGO_X_MAX
        and -EGO_Y_HALF <= y <= EGO_Y_HALF
        and EGO_Z_MIN <= z <= EGO_Z_MAX
    )


def nudge_camera(camera_se3: list[float]) -> tuple[list[float], float, bool]:
    """If the camera is inside the ego box, lift Z to ``LIFT_Z``.

    Returns ``(new_se3, z_offset_applied, was_lifted)``. ``new_se3`` is a
    fresh list (the input is not mutated). When the camera is already
    outside the box, the input is returned unchanged with offset 0.
    """
    x, y, z = camera_se3[0], camera_se3[1], camera_se3[2]
    if not is_inside_ego(x, y, z):
        return list(camera_se3), 0.0, False
    z_offset = LIFT_Z - z
    new_se3 = list(camera_se3)
    new_se3[2] = LIFT_Z
    return new_se3, z_offset, True


def process_rig(src: Path, dst: Path) -> dict:
    """Read a rig JSON, drop user-listed cameras, lift problematic cameras,
    write the modified rig.
    """
    rig = json.loads(src.read_text())
    rig_key = src.stem
    drop_names = set(CAMERAS_TO_DROP.get(rig_key, []))
    dropped: list[str] = []

    surviving_cameras = []
    for cam in rig["cameras"]:
        if str(cam["camera_name"]) in drop_names:
            dropped.append(str(cam["camera_name"]))
            continue
        surviving_cameras.append(cam)
    rig["cameras"] = surviving_cameras

    lifted: list[tuple[str, float, list[float], list[float]]] = []
    for cam in rig["cameras"]:
        old_se3 = list(cam["camera_to_imu_se3"])
        new_se3, z_offset, was_lifted = nudge_camera(old_se3)
        if was_lifted:
            cam["camera_to_imu_se3"] = new_se3
            # Record per-camera offset so the original digital-twin position
            # is recoverable. We store a 3-vector for forward-compatibility
            # with future xy nudges.
            cam["carla_camera_offset"] = [0.0, 0.0, float(z_offset)]
            lifted.append((str(cam["camera_name"]), z_offset, old_se3, new_se3))
    dst.write_text(json.dumps(rig, indent=2))
    return {
        "total": len(rig["cameras"]) + len(dropped),
        "dropped": dropped,
        "lifted": len(lifted),
        "details": lifted,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("rigs/real_rigs_small"),
        help="Source rig JSON directory.",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=Path("rigs/real_rigs_small_carla"),
        help="Output directory for the CARLA-collection-ready rigs.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        default=True,
        help="Wipe dst before writing (default true).",
    )
    args = parser.parse_args()

    if args.clean and args.dst.exists():
        shutil.rmtree(args.dst)
    args.dst.mkdir(parents=True, exist_ok=True)

    print(f"Ego box (IMU frame): x in [{EGO_X_MIN}, {EGO_X_MAX}], "
          f"|y| <= {EGO_Y_HALF}, z in [{EGO_Z_MIN}, {EGO_Z_MAX}]")
    print(f"Lift z to {LIFT_Z} m when inside.")
    print()

    src_files = sorted(args.src.glob("*.json"))
    if not src_files:
        print(f"No rig JSONs found in {args.src}")
        return

    total_lifted = 0
    total_dropped = 0
    for src_path in src_files:
        dst_path = args.dst / src_path.name
        stats = process_rig(src_path, dst_path)
        total_lifted += stats["lifted"]
        total_dropped += len(stats["dropped"])
        print(f"=== {src_path.name}: "
              f"dropped {len(stats['dropped'])}, lifted {stats['lifted']} "
              f"of {stats['total']} cameras ===")
        for name in stats["dropped"]:
            print(f"  DROP {name}")
        for cam_name, z_offset, old, new in stats["details"]:
            print(f"  LIFT {cam_name}: "
                  f"({old[0]:+.2f}, {old[1]:+.2f}, {old[2]:+.2f}) -> "
                  f"({new[0]:+.2f}, {new[1]:+.2f}, {new[2]:+.2f}) "
                  f"  (z += {z_offset:+.2f})")
        if stats["lifted"] == 0 and not stats["dropped"]:
            print("  (no changes)")
        print()

    print(f"Done: {total_lifted} lifted, {total_dropped} dropped "
          f"across {len(src_files)} rig(s)")
    print(f"Output: {args.dst}")


if __name__ == "__main__":
    main()
