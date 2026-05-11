"""Generate many random rigs and visualize them as a BEV GIF.

Each frame in the GIF is one randomly seeded rig: the ego rectangle is drawn
with the camera mount points and horizontal FOV wedges in py123d's body frame
(X forward = right of plot, Y left = up of plot — the rear axle is at origin).

Run from the LEAD repo root::

    python scripts/visualize_random_rigs.py --num-rigs 24 \
        --output rigs/random_rigs_bev.gif

Pass ``--show-slots`` to overlay the per-:class:`CameraID` slot rectangles
the random generator samples within. Pass ``--num-cameras N`` for a fixed-size
rig; omit to use the default Bernoulli-per-slot sampling.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon, Rectangle

from lead.expert.expert_py123d_utils import get_carla_lincoln_mkz_2020_metadata
from lead.rig.random_rig import Slot, generate_random_rig, get_default_slots
from lead.rig.rig_config import CameraEntry, RigConfig

EGO = get_carla_lincoln_mkz_2020_metadata()


def _camera_yaw_in_body_frame(camera: CameraEntry) -> float:
    """Recover the optical-axis yaw (radians, py123d body frame) of a camera.

    The camera's optical axis is +Z in pZmYpX convention. Mapping that
    direction through the camera-to-IMU rotation gives the optical axis in
    body frame; we project it to the X/Y plane.
    """
    rotation = camera.camera_to_imu_se3.rotation_matrix
    optical_axis_body = rotation @ np.array([0.0, 0.0, 1.0])
    return math.atan2(optical_axis_body[1], optical_axis_body[0])


def _camera_hfov_rad(camera: CameraEntry) -> float:
    """Horizontal field of view in radians from intrinsics."""
    return 2.0 * math.atan(camera.width / (2.0 * camera.intrinsics.fx))


def _wedge_polygon(
    centre_xy: tuple[float, float],
    yaw_rad: float,
    hfov_rad: float,
    length_m: float,
) -> np.ndarray:
    """Triangular FOV wedge anchored at ``centre_xy`` pointing along
    ``yaw_rad``."""
    half = hfov_rad / 2.0
    cx, cy = centre_xy
    left = (
        cx + length_m * math.cos(yaw_rad + half),
        cy + length_m * math.sin(yaw_rad + half),
    )
    right = (
        cx + length_m * math.cos(yaw_rad - half),
        cy + length_m * math.sin(yaw_rad - half),
    )
    return np.array([centre_xy, left, right])


def _carla_to_py123d_xy(x_carla: float, y_carla: float) -> tuple[float, float]:
    """Map a CARLA-frame (x_carla, y_carla) onto the py123d BEV plot."""
    return (
        x_carla + EGO.rear_axle_to_center_longitudinal,
        -y_carla,
    )


def _draw_ego(ax: plt.Axes) -> None:
    """Draw the Lincoln MKZ outline in py123d body frame.

    The ego's geometric centre sits ``rear_axle_to_center_longitudinal`` ahead 
    of the rear axle (which is at the BEV origin). Length extends along +X,
    width along +/- Y.
    """
    length = EGO.length
    width = EGO.width
    rear_x = EGO.rear_axle_to_center_longitudinal - length / 2.0
    ego = Rectangle(
        xy=(rear_x, -width / 2.0),
        width=length,
        height=width,
        linewidth=1.0,
        edgecolor="black",
        facecolor="0.85",
        zorder=1,
    )
    ax.add_patch(ego)
    ax.plot(0.0, 0.0, marker="x", color="black", markersize=8, zorder=2)
    ax.text(0.15, 0.15, "rear axle", fontsize=7, color="black", zorder=3)
    nose_x = rear_x + length
    ax.annotate(
        "",
        xy=(nose_x + 1.0, 0.0),
        xytext=(nose_x, 0.0),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.2},
        zorder=3,
    )


def _draw_slot_overlay(ax: plt.Axes, slots: dict) -> None:
    """Draw each slot's allowed mount-XY region as a faint dashed rectangle."""
    cmap = plt.get_cmap("tab10", max(len(slots), 1))
    for idx, (camera_id, slot) in enumerate(
        sorted(slots.items(), key=lambda kv: kv[0].name)
    ):
        slot_box = _slot_to_py123d_box(slot)
        x_min, y_min, x_max, y_max = slot_box
        rect = Rectangle(
            xy=(x_min, y_min),
            width=x_max - x_min,
            height=y_max - y_min,
            linewidth=0.8,
            edgecolor=cmap(idx),
            facecolor="none",
            linestyle="--",
            alpha=0.45,
            zorder=0,
        )
        ax.add_patch(rect)
        ax.text(
            x_min,
            y_max + 0.05,
            camera_id.name,
            fontsize=5,
            color=cmap(idx),
            alpha=0.7,
            zorder=0,
        )


def _slot_to_py123d_box(slot: Slot) -> tuple[float, float, float, float]:
    """Convert a CARLA-frame slot to a py123d BEV (x_min,y_min,x_max,y_max)."""
    x_lo, y_lo = _carla_to_py123d_xy(slot.x_range[0], slot.y_range[1])
    x_hi, y_hi = _carla_to_py123d_xy(slot.x_range[1], slot.y_range[0])
    return (min(x_lo, x_hi), min(y_lo, y_hi), max(x_lo, x_hi), max(y_lo, y_hi))


def _draw_rig(ax: plt.Axes, rig: RigConfig, wedge_length: float = 5.0) -> None:
    """Draw cameras + FOV wedges of a single rig onto ``ax``."""
    cmap = plt.get_cmap("tab10", max(len(rig.cameras), 1))
    for idx, camera in enumerate(rig.cameras):
        x = camera.camera_to_imu_se3.x
        y = camera.camera_to_imu_se3.y
        yaw = _camera_yaw_in_body_frame(camera)
        hfov = _camera_hfov_rad(camera)
        colour = cmap(idx)

        wedge = _wedge_polygon((x, y), yaw, hfov, wedge_length)
        ax.add_patch(Polygon(wedge, alpha=0.20, color=colour, zorder=2))
        ax.plot(x, y, marker="o", color=colour, markersize=6, zorder=3)
        ax.text(
            x + 0.15,
            y + 0.15,
            camera.camera_id.name,
            fontsize=6,
            color=colour,
            zorder=4,
        )


def _build_rigs(num_rigs: int, num_cameras: int | None) -> list[RigConfig]:
    return [
        generate_random_rig(seed=seed, num_cameras=num_cameras) \
            for seed in range(num_rigs)
        ]


def _set_axes(ax: plt.Axes, view_radius: float) -> None:
    ax.set_xlim(-view_radius, view_radius)
    ax.set_ylim(-view_radius, view_radius)
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlabel("X (forward) [m]")
    ax.set_ylabel("Y (left) [m]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-rigs", type=int, default=24)
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=None,
        help="Force exactly N cameras per rig. Default: Bernoulli per slot.",
    )
    parser.add_argument("--output", default="rigs/random_rigs_bev.gif")
    parser.add_argument("--fps", type=int, default=2, help="GIF frame rate.")
    parser.add_argument(
        "--view-radius", type=float, default=8.0,
        help="Half side of the BEV plot in metres."
    )
    parser.add_argument(
        "--wedge-length", type=float, default=5.0,
        help="FOV wedge length in metres."
    )
    parser.add_argument(
        "--show-slots",
        action="store_true",
        help="Overlay each CameraID's allowed mount-XY rectangle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rigs = _build_rigs(args.num_rigs, args.num_cameras)
    slots = get_default_slots()

    fig, ax = plt.subplots(figsize=(7, 7))

    def render(frame_idx: int) -> None:
        ax.clear()
        rig = rigs[frame_idx]
        _set_axes(ax, args.view_radius)
        if args.show_slots:
            _draw_slot_overlay(ax, slots)
        _draw_ego(ax)
        _draw_rig(ax, rig, wedge_length=args.wedge_length)
        ax.set_title(
            f"{rig.rig_name}  ({len(rig.cameras)} cameras)\n"
            f"frame {frame_idx + 1} / {len(rigs)}",
            fontsize=10,
        )

    animation = FuncAnimation(
        fig, render, frames=len(rigs), interval=1000 // max(args.fps, 1)
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"Wrote {len(rigs)} rigs to {output_path}")


if __name__ == "__main__":
    main()
