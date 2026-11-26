import math
from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray

from alula.linear_analysis import AircraftModes, Mode
from alula.simulation import FlightHistory, SimulationResult

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
UNSTABLE_WASH = "#d03b3b"

LINE_WIDTH = 1.5
MARKER_SIZE = 8.0

FLIGHT_PANELS = (
    "Altitude",
    "Airspeed",
    "Heading",
    "Attitude",
    "Aerodynamic angles",
    "Control surfaces",
    "Throttle",
)


def plot_flight(result: SimulationResult) -> Figure:
    history = result.history()
    figure, axes = plt.subplots(
        len(FLIGHT_PANELS), 1, sharex=True, figsize=(9.0, 13.0), facecolor=SURFACE
    )
    altitude_axes, airspeed_axes, heading_axes, attitude_axes, angle_axes, surface_axes, throttle_axes = axes

    _single_series(altitude_axes, history.time, history.altitude, "Altitude", "m")
    _single_series(airspeed_axes, history.time, history.airspeed, "Airspeed", "m/s")
    _single_series(heading_axes, history.time, continuous_heading_degrees(history.yaw), "Heading", "deg")
    _multi_series(
        attitude_axes,
        history.time,
        {"pitch": np.degrees(history.pitch), "roll": np.degrees(history.roll)},
        "Attitude",
        "deg",
    )
    _multi_series(
        angle_axes,
        history.time,
        {"alpha": np.degrees(history.alpha), "beta": np.degrees(history.beta)},
        "Aerodynamic angles",
        "deg",
    )

    control_time = history.time[:-1]
    _multi_series(
        surface_axes,
        control_time,
        {
            "elevator": np.degrees([controls.elevator for controls in result.controls]),
            "aileron": np.degrees([controls.aileron for controls in result.controls]),
            "rudder": np.degrees([controls.rudder for controls in result.controls]),
        },
        "Control surfaces",
        "deg",
        drawstyle="steps-post",
    )
    _single_series(
        throttle_axes,
        control_time,
        100.0 * np.array([controls.throttle for controls in result.controls]),
        "Throttle",
        "%",
        drawstyle="steps-post",
    )

    throttle_axes.set_xlabel("time [s]", color=SECONDARY_INK)
    figure.align_ylabels(axes)
    figure.tight_layout()
    return figure


def plot_ground_track(history: FlightHistory) -> Figure:
    figure, axes = plt.subplots(figsize=(7.0, 7.0), facecolor=SURFACE)
    _style(axes)
    axes.plot(history.east, history.north, color=SERIES_COLORS[0], linewidth=LINE_WIDTH)
    for index, label in ((0, "start"), (-1, "end")):
        axes.plot(
            history.east[index],
            history.north[index],
            marker="o",
            markersize=MARKER_SIZE,
            color=SERIES_COLORS[0],
            markeredgecolor=SURFACE,
            markeredgewidth=2.0,
            linestyle="none",
        )
        axes.annotate(
            label,
            (history.east[index], history.north[index]),
            xytext=(8, 6),
            textcoords="offset points",
            color=SECONDARY_INK,
        )
    axes.set_aspect("equal", adjustable="datalim")
    axes.set_xlabel("east [m]", color=SECONDARY_INK)
    axes.set_ylabel("north [m]", color=SECONDARY_INK)
    axes.set_title("Ground track", loc="left", color=PRIMARY_INK)
    figure.tight_layout()
    return figure


def plot_modes(modes: AircraftModes) -> Figure:
    named_modes = {
        "short period": modes.short_period,
        "phugoid": modes.phugoid,
        "roll subsidence": modes.roll_subsidence,
        "Dutch roll": modes.dutch_roll,
        "spiral": modes.spiral,
    }
    eigenvalues = _eigenvalues_with_conjugates(named_modes.values())

    figure, axes = plt.subplots(figsize=(9.0, 5.5), facecolor=SURFACE)
    _style(axes)
    axes.set_xscale("symlog", linthresh=0.05)

    largest_real = max(abs(value.real) for value in eigenvalues)
    largest_imaginary = max(abs(value.imag) for value in eigenvalues)
    right_edge = 0.2
    axes.set_xlim(-6.0 * largest_real, right_edge)
    axes.set_ylim(-1.3 * largest_imaginary, 1.3 * largest_imaginary)
    axes.axvspan(0.0, right_edge, color=UNSTABLE_WASH, alpha=0.08, linewidth=0.0)
    axes.text(0.01, 1.2 * largest_imaginary, "unstable", color=SECONDARY_INK, va="top")
    axes.axvline(0.0, color=BASELINE, linewidth=1.0)
    axes.axhline(0.0, color=BASELINE, linewidth=1.0)

    axes.plot(
        [value.real for value in eigenvalues],
        [value.imag for value in eigenvalues],
        marker="o",
        markersize=MARKER_SIZE,
        color=SERIES_COLORS[0],
        markeredgecolor=SURFACE,
        markeredgewidth=2.0,
        linestyle="none",
    )
    for name, mode in named_modes.items():
        axes.annotate(
            name,
            (mode.eigenvalue.real, abs(mode.eigenvalue.imag)),
            xytext=(-6, 9),
            textcoords="offset points",
            ha="right",
            color=PRIMARY_INK,
        )

    axes.set_xlabel("real part [1/s]   (left: decays faster)", color=SECONDARY_INK)
    axes.set_ylabel("imaginary part [rad/s]   (higher: oscillates faster)", color=SECONDARY_INK)
    axes.set_title("Dynamic modes: eigenvalues of the linearised aircraft", loc="left", color=PRIMARY_INK)
    figure.tight_layout()
    return figure


def continuous_heading_degrees(yaw: Sequence[float]) -> NDArray[np.float64]:
    heading = np.degrees(np.unwrap(np.asarray(yaw, dtype=float)))
    return heading + (heading[0] % 360.0 - heading[0])


def separated_label_positions(values: Sequence[float], minimum_gap: float) -> list[float]:
    order = np.argsort(values)
    positions = np.asarray(values, dtype=float).copy()
    for previous, current in zip(order, order[1:]):
        positions[current] = max(positions[current], positions[previous] + minimum_gap)
    return positions.tolist()


def _eigenvalues_with_conjugates(modes: Sequence[Mode]) -> list[complex]:
    values = []
    for mode in modes:
        values.append(mode.eigenvalue)
        if mode.is_oscillatory:
            values.append(mode.eigenvalue.conjugate())
    return values


def _single_series(
    axes: Axes,
    time: NDArray[np.float64],
    values: Sequence[float],
    title: str,
    unit: str,
    drawstyle: str = "default",
) -> None:
    _style(axes)
    axes.plot(time, values, color=SERIES_COLORS[0], linewidth=LINE_WIDTH, drawstyle=drawstyle)
    axes.set_title(title, loc="left", color=PRIMARY_INK, fontsize=10)
    axes.set_ylabel(unit, color=SECONDARY_INK)
    _avoid_flat_axis(axes, values)


def _multi_series(
    axes: Axes,
    time: NDArray[np.float64],
    series: dict[str, Sequence[float]],
    title: str,
    unit: str,
    drawstyle: str = "default",
) -> None:
    _style(axes)
    for color, (label, values) in zip(SERIES_COLORS, series.items()):
        axes.plot(time, values, color=color, linewidth=LINE_WIDTH, drawstyle=drawstyle, label=label)
    axes.set_title(title, loc="left", color=PRIMARY_INK, fontsize=10)
    axes.set_ylabel(unit, color=SECONDARY_INK)
    axes.legend(loc="upper right", frameon=False, ncols=len(series), fontsize=8, labelcolor=SECONDARY_INK)
    _avoid_flat_axis(axes, np.concatenate([np.asarray(values) for values in series.values()]))

    low, high = axes.get_ylim()
    end_values = [float(values[-1]) for values in series.values()]
    for label, position in zip(series, separated_label_positions(end_values, 0.12 * (high - low))):
        axes.annotate(
            label,
            (time[-1], position),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=SECONDARY_INK,
            fontsize=8,
            annotation_clip=False,
        )


def _avoid_flat_axis(axes: Axes, values: Sequence[float]) -> None:
    low, high = float(np.min(values)), float(np.max(values))
    if math.isclose(low, high, abs_tol=1e-9):
        middle = 0.5 * (low + high)
        margin = max(1.0, abs(middle) * 0.01)
        axes.set_ylim(middle - margin, middle + margin)


def _style(axes: Axes) -> None:
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRIDLINE, linewidth=0.6, linestyle="-")
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(BASELINE)
    axes.tick_params(colors=MUTED_INK, labelcolor=SECONDARY_INK, labelsize=8)
