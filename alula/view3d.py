from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D, Poly3DCollection
from numpy.typing import ArrayLike, NDArray

from alula.dynamics import POSITION_NED, QUATERNION
from alula.plotting import GRIDLINE, PRIMARY_INK, SECONDARY_INK, SERIES_COLORS, SURFACE
from alula.rotations import body_to_ned_from_quaternion
from alula.simulation import SimulationResult

AIRCRAFT_BODY_POLYGONS = {
    "fuselage": np.array(
        [[0.50, 0.0, 0.0], [0.30, 0.0, -0.06], [-0.50, 0.0, -0.03], [-0.50, 0.0, 0.03], [0.30, 0.0, 0.07]]
    ),
    "wing": np.array([[0.12, -0.55, 0.0], [0.12, 0.55, 0.0], [-0.08, 0.55, 0.0], [-0.08, -0.55, 0.0]]),
    "tailplane": np.array([[-0.38, -0.20, 0.0], [-0.38, 0.20, 0.0], [-0.50, 0.20, 0.0], [-0.50, -0.20, 0.0]]),
    "fin": np.array([[-0.34, 0.0, 0.0], [-0.50, 0.0, 0.0], [-0.52, 0.0, -0.20], [-0.44, 0.0, -0.20]]),
}
NOSE_BODY = np.array([0.50, 0.0, 0.0])
RIGHT_WINGTIP_BODY = np.array([0.0, 0.55, 0.0])
LEFT_WINGTIP_BODY = np.array([0.0, -0.55, 0.0])
FIN_TIP_BODY = np.array([-0.52, 0.0, -0.20])
TAIL_BODY = np.array([-0.50, 0.0, 0.0])

AIRCRAFT_COLOR = SERIES_COLORS[1]
TRAJECTORY_COLOR = SERIES_COLORS[0]
SHADOW_COLOR = "#b5b3aa"
MINIMUM_AIRCRAFT_SCALE = 20.0
AIRCRAFT_SCALE_FRACTION = 0.06
MINIMUM_VERTICAL_BOX_RATIO = 0.15


@dataclass
class FlightAnimation:
    figure: Figure
    animation: FuncAnimation
    frame_indices: list[int]
    aircraft: Poly3DCollection
    trail: Line3D
    show_frame: Callable[[int], dict[str, NDArray[np.float64]]]

    def save_gif(self, path: str | Path, frames_per_second: int = 10) -> None:
        self.animation.save(str(path), writer=PillowWriter(fps=frames_per_second))


def ned_to_plot(points_ned: ArrayLike) -> NDArray[np.float64]:
    points = np.atleast_2d(np.asarray(points_ned, dtype=float))
    return np.column_stack([points[:, 1], points[:, 0], -points[:, 2]])


def body_points_to_plot(
    points_body: ArrayLike, position_ned: ArrayLike, quaternion: ArrayLike, scale: float
) -> NDArray[np.float64]:
    body_to_ned = body_to_ned_from_quaternion(quaternion)
    points_ned = np.asarray(position_ned, dtype=float) + scale * np.atleast_2d(points_body) @ body_to_ned.T
    return ned_to_plot(points_ned)


def aircraft_polygons(
    position_ned: ArrayLike, quaternion: ArrayLike, scale: float
) -> dict[str, NDArray[np.float64]]:
    return {
        name: body_points_to_plot(polygon, position_ned, quaternion, scale)
        for name, polygon in AIRCRAFT_BODY_POLYGONS.items()
    }


def default_aircraft_scale(result: SimulationResult) -> float:
    positions = ned_to_plot(result.states[:, POSITION_NED])
    extent = float(np.ptp(positions[:, :2], axis=0).max())
    return max(MINIMUM_AIRCRAFT_SCALE, AIRCRAFT_SCALE_FRACTION * extent)


def snapshot_indices(times: ArrayLike, interval: float) -> list[int]:
    if interval <= 0.0:
        raise ValueError(f"Snapshot interval must be positive, got {interval} s")
    times = np.asarray(times, dtype=float)
    marks = np.arange(times[0], times[-1] + 1e-9, interval)
    return sorted({int(np.argmin(np.abs(times - mark))) for mark in marks})


def plot_flight_3d(
    result: SimulationResult,
    snapshot_interval: float = 15.0,
    aircraft_scale: float | None = None,
) -> Figure:
    scale = aircraft_scale or default_aircraft_scale(result)
    positions = ned_to_plot(result.states[:, POSITION_NED])
    figure, axes = _make_axes()

    axes.plot(*positions.T, color=TRAJECTORY_COLOR, linewidth=1.5, label="trajectory")
    axes.plot(
        positions[:, 0], positions[:, 1], np.zeros(len(positions)),
        color=SHADOW_COLOR, linewidth=1.0, label="ground track",
    )
    for index in snapshot_indices(result.times, snapshot_interval):
        state = result.states[index]
        axes.add_collection3d(_aircraft_artist(aircraft_polygons(state[POSITION_NED], state[QUATERNION], scale)))
    axes.plot([], [], [], color=AIRCRAFT_COLOR, linewidth=6.0, label="aircraft (enlarged)")

    exaggeration = _frame_axes(axes, positions, scale)
    axes.set_title(_title("Flight in 3D", exaggeration), loc="left", color=PRIMARY_INK)
    axes.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=SECONDARY_INK)
    return figure


def animate_flight(
    result: SimulationResult,
    seconds_per_frame: float = 1.0,
    playback_speed: float = 10.0,
    aircraft_scale: float | None = None,
) -> FlightAnimation:
    if playback_speed <= 0.0:
        raise ValueError(f"Playback speed must be positive, got {playback_speed}")
    scale = aircraft_scale or default_aircraft_scale(result)
    positions = ned_to_plot(result.states[:, POSITION_NED])
    frames = snapshot_indices(result.times, seconds_per_frame)
    figure, axes = _make_axes()

    axes.plot(
        positions[:, 0], positions[:, 1], np.zeros(len(positions)),
        color=SHADOW_COLOR, linewidth=1.0, label="ground track",
    )
    (trail,) = axes.plot([], [], [], color=TRAJECTORY_COLOR, linewidth=1.5, label="trajectory")
    first = result.states[frames[0]]
    aircraft = _aircraft_artist(aircraft_polygons(first[POSITION_NED], first[QUATERNION], scale))
    axes.add_collection3d(aircraft)
    axes.plot([], [], [], color=AIRCRAFT_COLOR, linewidth=6.0, label="aircraft (enlarged)")
    title = axes.set_title("", loc="left", color=PRIMARY_INK)
    exaggeration = _frame_axes(axes, positions, scale)
    axes.legend(loc="upper left", frameon=False, fontsize=8, labelcolor=SECONDARY_INK)

    def show_frame(frame: int) -> dict[str, NDArray[np.float64]]:
        index = frames[frame]
        state = result.states[index]
        polygons = aircraft_polygons(state[POSITION_NED], state[QUATERNION], scale)
        aircraft.set_verts(list(polygons.values()))
        trail.set_data_3d(*positions[: index + 1].T)
        title.set_text(_title(f"t = {result.times[index]:.0f} s", exaggeration))
        return polygons

    animation = FuncAnimation(
        figure,
        show_frame,
        frames=len(frames),
        interval=1000.0 * seconds_per_frame / playback_speed,
        blit=False,
    )
    return FlightAnimation(figure, animation, frames, aircraft, trail, show_frame)


def _make_axes() -> tuple[Figure, Axes3D]:
    figure = plt.figure(figsize=(9.0, 7.0), facecolor=SURFACE)
    axes = figure.add_subplot(projection="3d")
    axes.set_facecolor(SURFACE)
    for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        axis.set_pane_color(SURFACE)
        axis._axinfo["grid"]["color"] = GRIDLINE
    axes.tick_params(labelsize=8, labelcolor=SECONDARY_INK)
    axes.set_xlabel("east [m]", color=SECONDARY_INK)
    axes.set_ylabel("north [m]", color=SECONDARY_INK)
    axes.set_zlabel("altitude [m]", color=SECONDARY_INK)
    return figure, axes


def _aircraft_artist(polygons: dict[str, NDArray[np.float64]]) -> Poly3DCollection:
    return Poly3DCollection(
        list(polygons.values()),
        facecolors=AIRCRAFT_COLOR,
        edgecolors=PRIMARY_INK,
        linewidths=0.4,
        alpha=0.9,
    )


def _title(text: str, vertical_exaggeration: float) -> str:
    if vertical_exaggeration > 1.0 + 1e-9:
        return f"{text}   (vertical scale exaggerated x{vertical_exaggeration:.1f})"
    return text


def _frame_axes(axes: Axes3D, positions: NDArray[np.float64], scale: float) -> float:
    low = positions.min(axis=0) - scale
    high = positions.max(axis=0) + scale
    centre = 0.5 * (low + high)
    half_span = 0.5 * float((high[:2] - low[:2]).max())
    axes.set_xlim(centre[0] - half_span, centre[0] + half_span)
    axes.set_ylim(centre[1] - half_span, centre[1] + half_span)
    axes.set_zlim(0.0, high[2])
    true_ratio = high[2] / (2.0 * half_span)
    box_ratio = max(MINIMUM_VERTICAL_BOX_RATIO, true_ratio)
    axes.set_box_aspect((1.0, 1.0, box_ratio), zoom=1.25)
    return box_ratio / true_ratio
