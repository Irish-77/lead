#!/usr/bin/env python3
"""
Hotkey script to start Viser viewer for visualizing 123D scenes.
By default, you can open viewer via: http://localhost:8080
"""

import argparse
import os
import sys
import time

try:
    from watchdog.events import FileSystemEventHandler
    from watchdog.observers import Observer

    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False

import numpy as np

from py123d.api.scene.arrow.arrow_scene_builder import ArrowSceneBuilder
from py123d.api.scene.scene_filter import SceneFilter
from py123d.common.execution.thread_pool_executor import ThreadPoolExecutor
from py123d.datatypes.sensors.lidar import LidarID
import py123d.visualization.viser.elements.box_detections_se3_element as _box_element_module
from py123d.visualization.viser.viser_config import ViserConfig
from py123d.visualization.viser.viser_viewer import ViserViewer

# Recolour 3D box detections with no lidar points (or `num_lidar_points`
# unset) to red. These boxes would be filtered out of lidar-conditioned
# training; highlighting them helps spot frustum-filter or num_lidar_points
# wiring issues during CARLA-collection validation.
_NO_LIDAR_COLOR = np.array([1.0, 0.2, 0.2], dtype=np.float32)
_orig_get_outlines = _box_element_module._get_bounding_box_outlines


def _outlines_with_no_lidar_highlight(scene, iteration, initial_ego_state):
    box_outlines, box_colors, box_se3_array = _orig_get_outlines(scene, iteration, initial_ego_state)
    if box_outlines.size == 0:
        return box_outlines, box_colors, box_se3_array

    box_detections = scene.get_box_detections_se3_at_iteration(iteration)
    bd_list = box_detections.box_detections if box_detections is not None else []
    if not bd_list:
        return box_outlines, box_colors, box_se3_array

    edges_per_box = box_colors.shape[0] // len(bd_list)
    for i, bd in enumerate(bd_list):
        npts = bd.attributes.num_lidar_points
        if npts is None or npts == 0:
            box_colors[i * edges_per_box : (i + 1) * edges_per_box, ...] = _NO_LIDAR_COLOR
    return box_outlines, box_colors, box_se3_array


_box_element_module._get_bounding_box_outlines = _outlines_with_no_lidar_highlight


def parse_args():
    parser = argparse.ArgumentParser(
        description="Start Viser viewer to visualize 123D scenes in 3D",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--split-names",
        type=str,
        nargs="+",
        default=None,
        help="Dataset split names to filter (e.g., 'train', 'val', 'test')",
    )
    parser.add_argument(
        "--future-duration-s",
        type=float,
        default=None,
        help="Future duration in seconds for each scene (None = complete log)",
    )
    parser.add_argument(
        "--timestamp-threshold-s",
        type=float,
        default=0.0,
        help="Timestamp threshold in seconds",
    )
    parser.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle scenes (default: shuffle enabled)",
    )
    parser.add_argument(
        "--require-map",
        action="store_true",
        help="Only include scenes/logs with an available map",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for Viser viewer server",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for Viser viewer server",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # --- Auto-restart logic ---
    # Only activate if watchdog is available and PY123D_DATA_ROOT is set
    data_root = os.environ.get("PY123D_DATA_ROOT", None)
    if WATCHDOG_AVAILABLE and data_root:

        class SceneChangeHandler(FileSystemEventHandler):
            def __init__(self, restart_callback):
                super().__init__()
                self.restart_callback = restart_callback

            def on_created(self, event):
                # Only restart for new files (not directories)
                if not event.is_directory:
                    print(
                        f"\n[Viser] New scene file detected: {event.src_path}. Restarting viewer...",
                    )
                    self.restart_callback()

        def restart_script():
            # Flush output, then exec self
            sys.stdout.flush()
            sys.stderr.flush()
            os.execv(sys.executable, [sys.executable] + sys.argv)

        observer = Observer()
        event_handler = SceneChangeHandler(restart_script)
        observer.schedule(event_handler, data_root, recursive=True)
        observer.start()
        print(f"[Viser] Watching for new scenes in: {data_root}")
        # Give watchdog a moment to start
        time.sleep(0.5)

    # Create scene filter
    scene_filter = SceneFilter(
        split_names=args.split_names,
        future_duration_s=args.future_duration_s,
        timestamp_threshold_s=args.timestamp_threshold_s,
        shuffle=not args.no_shuffle,
        has_map=True if args.require_map else None,
    )

    # Build scenes
    print("Building scenes from dataset...")
    executor = ThreadPoolExecutor()
    scenes = ArrowSceneBuilder().get_scenes(scene_filter, executor)

    dataset_splits = set(scene.log_metadata.split for scene in scenes)
    print(f"\nFound {len(scenes)} scenes from {len(dataset_splits)} dataset splits:")
    for split in dataset_splits:
        print(f" - {split}")

    if len(scenes) == 0:
        print("\nWarning: No scenes found with the given filter criteria!")
        print("Try adjusting your filter parameters or check your PY123D_DATA_ROOT.")
        return

    # Start Viser viewer
    print(f"\nStarting Viser viewer on http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop the viewer.")
    print("-" * 60)

    viser_config = ViserConfig(
        server={"host": args.host, "port": args.port},
        lidar={"ids": [LidarID.LIDAR_TOP]},
    )

    try:
        viewer = ViserViewer(scenes, viser_config=viser_config)
        viewer.server.gui.configure_theme(dark_mode=False)
    finally:
        # Stop watchdog observer on exit
        if WATCHDOG_AVAILABLE and data_root:
            observer.stop()
            observer.join()


if __name__ == "__main__":
    main()
