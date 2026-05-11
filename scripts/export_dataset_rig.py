"""Export a real-dataset rig from a py123d Arrow log directory.

NOTE:
A modified copy of this version exists in my personal repo at:
master-thesis/scripts/export_dataset_rig.py

This works for **any** dataset that has already been parsed into py123d's
Arrow format — nuScenes, Waymo Open Dataset, nuPlan, AV2, KITTI360,
PandaSet, Physical-AI-AV, etc. — because every Arrow log exposes the same
:meth:`SceneAPI.get_camera_metadatas` interface.

You do **not** pass the raw dataset (no nuScenes annotations, no Waymo
TFRecords). You pass the py123d Arrow log directory of one scene.

The leaf log directory is the folder that holds the ``*.arrow`` files for
one scene, e.g.:

    /py123d_data_root/logs/<split>/<log_name>/
    ├── camera.pcam_f0.arrow
    ├── ego_state_se3.arrow
    ├── box_detections_se3.arrow
    ├── ...
    └── sync.arrow

Usage::

    python scripts/export_dataset_rig.py \\
        --log-dir /path/to/py123d_data/logs/nuscenes_train/scene_0061 \\
        --output rigs/nuscenes.json

Or, if you only have the parent ``PY123D_DATA_ROOT`` and want the script to
pick the first log inside it::

    python scripts/export_dataset_rig.py \\
        --data-root /path/to/py123d_data \\
        --output rigs/nuscenes.json

Only pinhole cameras are exported — fisheye and f-theta entries are skipped
because :class:`RigConfig` only models pinhole intrinsics today.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lead.rig import dataset_rig, serialization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--log-dir",
        help="Leaf py123d log directory (the folder holding the *.arrow files).",
    )
    source.add_argument(
        "--data-root",
        help=(
            "PY123D_DATA_ROOT (or any folder containing logs/<split>/<log>); "
            "the script picks the first log it finds."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output JSON path for the RigConfig.",
    )
    parser.add_argument(
        "--rig-name",
        default=None,
        help=(
            "Override the rig name. Defaults to ``carla_<dataset>_<location>`` "
            "derived from the log metadata."
        ),
    )
    return parser.parse_args()


def _first_log_under(data_root: Path) -> Path:
    """Return the first leaf log directory found under ``data_root``."""
    logs_root = data_root / "logs"
    if not logs_root.is_dir():
        # Tolerate either {data_root}/logs/ or {data_root} pointing at a logs dir
        logs_root = data_root
    for arrow_file in sorted(logs_root.rglob("sync.arrow")):
        return arrow_file.parent
    raise FileNotFoundError(
        f"No log dir with sync.arrow found under {data_root}",
    )


def main() -> None:
    args = parse_args()

    if args.log_dir:
        log_dir = Path(args.log_dir).resolve()
    else:
        log_dir = _first_log_under(Path(args.data_root).resolve())
        print(f"[export_dataset_rig] Using first log: {log_dir}")

    rig = dataset_rig.extract_from_log_dir(
        log_dir=log_dir,
        rig_name=args.rig_name,
    )
    output_path = Path(args.output).resolve()
    serialization.save(rig, output_path)
    cam_summary = ", ".join(camera.camera_id.name for camera in rig.cameras)
    print(
        f"[export_dataset_rig] Wrote rig '{rig.rig_name}' with {len(rig.cameras)} "
        f"cameras ({cam_summary}) to {output_path}",
    )


if __name__ == "__main__":
    main()
