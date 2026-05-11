"""Top-level entry point for multi-rig, multi-route CARLA data collection.

Examples
--------

Bulk: take every rig in ``rigs/`` and a sample of routes from
``data/data_routes/leaderboard1/`` (default scenario directory)::

    python scripts/collect_carla_data.py \\
        --rigs-dir rigs/ \\
        --routes-dir data/data_routes/leaderboard1 \\
        --max-routes 6 \\
        --per-route-timeout 90 \\
        --output-base data/dataset_v1

Explicit list (still supported)::

    python scripts/collect_carla_data.py \\
        --rig rigs/random_seed42.json --rig rigs/nuscenes.json \\
        --routes data/data_routes/leaderboard1/BlockedIntersection/Town06_13.xml \\
        --routes data/data_routes/leaderboard1/T_Junction/Town04_2.xml \\
        --output-base data/dataset_v1

Each rig produces a complete log under ``<output-base>/<rig_name>/logs/...``;
all rigs see the same world inside one CARLA run per route. Routes are run
sequentially; if any single route exceeds ``--per-route-timeout`` seconds it
is sent ``SIGINT`` (graceful flush) and the loop moves on to the next route.
"""

from __future__ import annotations

import argparse
import os
import random
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

DEFAULT_ROUTES_DIR = Path("data/data_routes/leaderboard1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        "--rig",
        action="append",
        dest="rigs",
        default=[],
        help="Path to a rig JSON. Repeat for multiple rigs.",
    )
    parser.add_argument(
        "--rigs-dir",
        default=None,
        help="Directory containing rig JSONs (every *.json is loaded).",
    )
    parser.add_argument(
        "--routes",
        action="append",
        dest="routes",
        default=[],
        help="Path to a CARLA route XML. Repeat for multiple routes.",
    )
    parser.add_argument(
        "--routes-dir",
        default=None,
        help=(
            f"Directory under which all *.xml route files are discovered "
            f"recursively. Defaults to '{DEFAULT_ROUTES_DIR}' if neither --routes "
            "nor --routes-dir is given."
        ),
    )
    parser.add_argument(
        "--max-routes",
        type=int,
        default=None,
        help="Cap on the number of routes to run (after shuffling).",
    )
    parser.add_argument(
        "--routes-seed",
        type=int,
        default=0,
        help="RNG seed for route shuffling.",
    )
    parser.add_argument(
        "--no-shuffle-routes",
        action="store_true",
        help="Iterate routes in alphabetical order instead of shuffling.",
    )
    parser.add_argument(
        "--output-base",
        default="data/carla_multi_rig_py123d",
        help="Output directory base; per-rig dirs are created beneath this.",
    )
    parser.add_argument("--carla-port", type=int, default=2000)
    parser.add_argument("--traffic-manager-port", type=int, default=8000)
    parser.add_argument("--traffic-manager-seed", type=int, default=0)
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Leaderboard agent watchdog timeout (per-tick), seconds.",
    )
    parser.add_argument(
        "--per-route-timeout",
        type=int,
        default=120,
        help=(
            "Wall-clock budget per route in seconds. After this elapses the "
            "leaderboard process is sent SIGINT and the loop advances."
        ),
    )
    parser.add_argument(
        "--no-wiggle",
        action="store_true",
        help="Disable per-frame extrinsic wiggle (set in env).",
    )
    parser.add_argument(
        "--lead-log-level",
        default="INFO",
        help="Log level for LEAD modules (DEBUG|INFO|WARNING).",
    )
    return parser.parse_args()


def _resolve_rigs(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = [Path(p).resolve() for p in args.rigs]
    if args.rigs_dir:
        rigs_dir = Path(args.rigs_dir).resolve()
        if not rigs_dir.is_dir():
            raise NotADirectoryError(f"--rigs-dir is not a directory: {rigs_dir}")
        paths.extend(sorted(rigs_dir.glob("*.json")))
    if not paths:
        raise SystemExit("No rigs given. Pass --rig <path> or --rigs-dir <dir>.")
    seen: set[Path] = set()
    deduped: list[Path] = []
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"Rig not found: {p}")
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped


def _resolve_routes(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = [Path(p).resolve() for p in args.routes]
    routes_dir: Path | None = None
    if args.routes_dir:
        routes_dir = Path(args.routes_dir).resolve()
    elif not paths:
        # No explicit routes and no dir flag: fall back to the default.
        routes_dir = Path(DEFAULT_ROUTES_DIR).resolve()
        print(
            f"[collect_carla_data] No routes specified; defaulting to {routes_dir}",
        )
    if routes_dir is not None:
        if not routes_dir.is_dir():
            raise NotADirectoryError(f"--routes-dir is not a directory: {routes_dir}")
        paths.extend(sorted(routes_dir.rglob("*.xml")))
    if not paths:
        raise SystemExit("No routes found. Pass --routes <xml> or --routes-dir <dir>.")
    if not args.no_shuffle_routes:
        random.Random(args.routes_seed).shuffle(paths)
    if args.max_routes is not None:
        paths = paths[: args.max_routes]
    return paths


def _build_env(args: argparse.Namespace, rig_paths: list[Path]) -> dict[str, str]:
    env = os.environ.copy()
    env["LEAD_MULTI_RIG_CONFIGS"] = ";".join(str(p) for p in rig_paths)
    output_base = Path(args.output_base).resolve()
    env["LEAD_MULTI_RIG_OUTPUT_BASE"] = str(output_base)
    env["LEAD_LOG_LEVEL"] = args.lead_log_level
    # ExpertPy123D's parent setup demands PY123D_DATA_ROOT even though we
    # bypass its writer; point it at a sibling so the (unused) log lives
    # alongside the per-rig outputs.
    env.setdefault("PY123D_DATA_ROOT", str(output_base / "_lead_default"))
    env["LEAD_EXPERT_CONFIG"] = (
        "target_dataset=6 py123d_data_format=true use_radars=false "
        "use_two_lidars=false "
        "lidar_stack_size=2 save_only_non_ground_lidar=true save_lidar_only_inside_bev=true "
        # Wider lidar save crop (±60 m on both axes) — captures more side / rear
        # vehicles than LEAD's default (-32..64) x (-40..40), so fewer GT boxes
        # end up with num_lidar_points==0.
        "min_x_meter=-60 max_x_meter=60 min_y_meter=-60 max_y_meter=60 "
        "perturbate_sensors=false"
    )
    if args.no_wiggle:
        env["LEAD_MULTI_RIG_WIGGLE_DISABLED"] = "1"
    env["DEBUG_CHALLENGE"] = "0"
    env["DATAGEN"] = "1"
    return env


def _safe_signal(pid: int, sig: int) -> None:
    """Send ``sig`` to ``pid``'s process group, swallowing 'process gone' errors."""
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _reset_carla_world() -> None:
    """Run LEAD's reset script to clear stale actors before a route.

    The leaderboard refuses to spawn the ego if a previous route's actor
    still occupies the spawn point (``Cannot spawn actor vehicle.lincoln...``);
    a hard ``SIGKILL`` between routes leaves stale actors behind. We just call
    the same script LEAD recommends in its error message.
    """
    try:
        subprocess.run(
            [sys.executable, "scripts/reset_carla_world.py"],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass


def _run_single_route(
    args: argparse.Namespace,
    base_env: dict[str, str],
    route_path: Path,
    route_index: int,
    route_total: int,
) -> bool:
    """Run leaderboard for one route. Returns True on natural exit, False on timeout."""
    scenario_name = route_path.parent.name
    route_number = route_path.stem
    env = dict(base_env)
    env["SCENARIO_NAME"] = scenario_name
    env["ROUTE_NUMBER"] = route_number
    env["TEAM_CONFIG"] = str(route_path)
    env["SAVE_PATH"] = f"data/multi_rig_debug/{scenario_name}"
    env["CHECKPOINT_ENDPOINT"] = (
        f"data/multi_rig_debug/results/{scenario_name}_{route_number}_result.json"
    )

    _reset_carla_world()

    cmd = [
        sys.executable,
        "-u",
        "3rd_party/leaderboard_autopilot/leaderboard/leaderboard_evaluator_local.py",
        f"--port={args.carla_port}",
        f"--traffic-manager-port={args.traffic_manager_port}",
        f"--traffic-manager-seed={args.traffic_manager_seed}",
        f"--routes={route_path}",
        "--repetitions=1",
        "--track=MAP",
        f"--checkpoint={env['CHECKPOINT_ENDPOINT']}",
        "--agent=lead/expert/expert_multi_rig_py123d.py",
        f"--agent-config={route_path}",
        "--debug=0",
        "--resume=1",
        f"--timeout={args.timeout}",
    ]

    print(
        f"\n[collect_carla_data] === route {route_index}/{route_total}: "
        f"{scenario_name}/{route_number} ===",
    )
    print("  " + " ".join(shlex.quote(c) for c in cmd))

    proc = subprocess.Popen(cmd, env=env, start_new_session=True)
    deadline = time.monotonic() + args.per_route_timeout
    sigint_sent = False
    sigterm_sent = False
    sigterm_deadline = 0.0
    sigkill_deadline = 0.0
    while True:
        try:
            ret = proc.wait(timeout=2.0)
            return ret == 0 and not sigint_sent
        except subprocess.TimeoutExpired:
            now = time.monotonic()
            if not sigint_sent and now >= deadline:
                sigint_sent = True
                sigterm_deadline = now + 60.0  # grace for ArrowLogWriter flush
                print(
                    f"[collect_carla_data] per-route-timeout ({args.per_route_timeout}s) "
                    f"reached; sending SIGINT to flush",
                )
                _safe_signal(proc.pid, signal.SIGINT)
            elif sigint_sent and not sigterm_sent and now >= sigterm_deadline:
                sigterm_sent = True
                sigkill_deadline = now + 30.0
                print(
                    "[collect_carla_data] SIGINT did not exit; escalating to SIGTERM",
                )
                _safe_signal(proc.pid, signal.SIGTERM)
            elif sigterm_sent and now >= sigkill_deadline:
                print(
                    "[collect_carla_data] SIGTERM did not exit; escalating to SIGKILL",
                )
                _safe_signal(proc.pid, signal.SIGKILL)
                sigkill_deadline = now + 1e9


def main() -> None:
    args = parse_args()
    rig_paths = _resolve_rigs(args)
    route_paths = _resolve_routes(args)
    base_env = _build_env(args, rig_paths)

    output_base = Path(args.output_base).resolve()
    print(f"[collect_carla_data] Output base: {output_base}")
    print(f"[collect_carla_data] Rigs ({len(rig_paths)}): {[p.name for p in rig_paths]}")
    print(f"[collect_carla_data] Routes ({len(route_paths)}):")
    for i, route in enumerate(route_paths, 1):
        print(f"  {i:>3}. {route.parent.name}/{route.stem}")
    print(f"[collect_carla_data] Per-route wall-clock budget: {args.per_route_timeout}s")

    natural_exit = 0
    timeouts = 0
    for i, route in enumerate(route_paths, 1):
        ok = _run_single_route(args, base_env, route, i, len(route_paths))
        if ok:
            natural_exit += 1
        else:
            timeouts += 1

    print(
        f"\n[collect_carla_data] Done. {natural_exit} natural exit(s), "
        f"{timeouts} timeout/error(s) across {len(route_paths)} route(s).",
    )


if __name__ == "__main__":
    main()
