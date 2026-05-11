"""Delete log directories under a dataset root that have fewer than N iters.

A "short" log usually means the route went bad immediately after spawn — ego
fell through the world, sensors stayed black, etc. — and the only thing saved
was iter 0. Those logs add noise to the dataset (training would oversample the
spawn moment), so we drop them and let the collection loop re-attempt to fill
the missing budget.

Usage::

    python scripts/prune_short_logs.py data/dataset_full/phase_random_a \
        --min-iters 10
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from py123d.api.scene.arrow.arrow_scene_api import ArrowSceneAPI


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Phase output directory.")
    parser.add_argument(
        "--min-iters",
        type=int,
        default=10,
        help="Logs with fewer than this many saved iters are deleted.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be deleted without removing anything.",
    )
    args = parser.parse_args()

    if not args.root.is_dir():
        print(f"{args.root} does not exist; nothing to prune")
        return 0

    short_logs: list[Path] = []
    for sync in args.root.rglob("sync.arrow"):
        try:
            api = ArrowSceneAPI(log_dir=sync.parent)
            n = len(api.get_all_iteration_timestamps())
        except Exception as e:  # noqa: BLE001
            # Corrupt log — also delete
            print(f"  CORRUPT  {sync.parent}: {e}")
            short_logs.append(sync.parent)
            continue
        if n < args.min_iters:
            short_logs.append(sync.parent)

    if not short_logs:
        print(f"No short logs under {args.root} (min_iters={args.min_iters})")
        return 0

    total = sum(1 for _ in args.root.rglob("sync.arrow"))
    print(
        f"Pruning {len(short_logs)}/{total} short logs "
        f"(<{args.min_iters} iters) under {args.root}",
    )
    for log_dir in short_logs:
        if args.dry_run:
            print(f"  DRY-RUN would delete  {log_dir}")
        else:
            shutil.rmtree(log_dir, ignore_errors=True)
    if not args.dry_run:
        print(f"Deleted {len(short_logs)} short log dir(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
