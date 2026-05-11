"""Count saved iterations per rig under a dataset root.

Used by the full-dataset collector to decide whether another collection cycle
is needed.

Useful for monitoring progress when process runs in background.

Usage::

    python scripts/dataset_progress.py data/dataset_full/phase_real
    python scripts/dataset_progress.py data/dataset_full/phase_real --target 5000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Phase output directory.")
    parser.add_argument("--target", type=int, default=None)
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"{args.root} does not exist (yet)")
        return 1 if args.target is not None else 0

    rig_totals: dict[str, int] = {}
    for sync in args.root.rglob("sync.arrow"):
        rig = sync.parent.parent.parent.parent.name  # <root>/<rig>/logs/<split>/<log>/sync.arrow
        try:
            api = ArrowSceneAPI(log_dir=sync.parent)
            n = len(api.get_all_iteration_timestamps())
        except Exception as e:  # noqa: BLE001
            print(f"  WARN: {sync.parent.name}: {e}")
            continue
        rig_totals[rig] = rig_totals.get(rig, 0) + n

    if not rig_totals:
        print(f"No data under {args.root}")
        return 1 if args.target is not None else 0

    width = max(len(r) for r in rig_totals)
    for rig in sorted(rig_totals):
        marker = ""
        if args.target is not None:
            marker = " [DONE]" if rig_totals[rig] >= args.target else " [more]"
        print(f"  {rig:<{width}}  {rig_totals[rig]:>6d} iters{marker}")

    if args.target is not None:
        all_done = all(v >= args.target for v in rig_totals.values())
        print(
            f"\nTarget {args.target}:"
            f"{'all rigs done' if all_done else 'more cycles needed'}"
        )
        return 0 if all_done else 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
