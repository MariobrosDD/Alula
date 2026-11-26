import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from alula.autopilot import Autopilot, AutopilotTargets
from alula.linear_analysis import aircraft_modes, linearize
from alula.plotting import (
    FLIGHT_PANELS,
    continuous_heading_degrees,
    plot_flight,
    plot_ground_track,
    plot_modes,
    separated_label_positions,
)
from alula.simulation import simulate
from alula.trim import trim

TIME_STEP = 0.05


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def trimmed(light_aircraft):
    return trim(light_aircraft, 1000.0, 57.7)


@pytest.fixture
def turning_flight(light_aircraft, trimmed):
    pilot = Autopilot(AutopilotTargets(1050.0, math.radians(60.0), 55.0), trimmed, TIME_STEP)
    return simulate(light_aircraft, trimmed.state, pilot, 30.0, TIME_STEP)


def panel(figure, title):
    return next(axes for axes in figure.axes if axes.get_title(loc="left") == title)


def line_labelled(axes, label):
    return next(line for line in axes.get_lines() if line.get_label() == label)


def eigenvalue_markers(axes):
    return next(line for line in axes.get_lines() if line.get_marker() == "o")


def test_flight_figure_has_one_panel_per_quantity(turning_flight):
    figure = plot_flight(turning_flight)
    assert [axes.get_title(loc="left") for axes in figure.axes] == list(FLIGHT_PANELS)


def test_all_flight_panels_share_the_time_axis(turning_flight):
    figure = plot_flight(turning_flight)
    limits = {axes.get_xlim() for axes in figure.axes}
    assert len(limits) == 1


def test_altitude_panel_draws_the_simulated_altitude(turning_flight):
    history = turning_flight.history()
    line = panel(plot_flight(turning_flight), "Altitude").get_lines()[0]
    assert line.get_xdata() == pytest.approx(history.time)
    assert line.get_ydata() == pytest.approx(history.altitude)


def test_angles_are_drawn_in_degrees(turning_flight):
    history = turning_flight.history()
    figure = plot_flight(turning_flight)
    attitude = panel(figure, "Attitude")
    assert line_labelled(attitude, "roll").get_ydata() == pytest.approx(np.degrees(history.roll))
    assert line_labelled(attitude, "pitch").get_ydata() == pytest.approx(np.degrees(history.pitch))
    assert attitude.get_ylabel() == "deg"
    assert line_labelled(panel(figure, "Aerodynamic angles"), "alpha").get_ydata() == pytest.approx(
        np.degrees(history.alpha)
    )


def test_controls_are_drawn_as_held_steps_in_degrees_and_percent(turning_flight):
    figure = plot_flight(turning_flight)
    surfaces = panel(figure, "Control surfaces")
    aileron = line_labelled(surfaces, "aileron")
    assert aileron.get_drawstyle() == "steps-post"
    assert aileron.get_xdata() == pytest.approx(turning_flight.times[:-1])
    assert aileron.get_ydata() == pytest.approx(
        np.degrees([controls.aileron for controls in turning_flight.controls])
    )
    throttle = panel(figure, "Throttle").get_lines()[0]
    assert throttle.get_ydata() == pytest.approx(
        [100.0 * controls.throttle for controls in turning_flight.controls]
    )


def test_multi_series_panels_have_a_legend_and_single_series_panels_do_not(turning_flight):
    figure = plot_flight(turning_flight)
    for title in ("Attitude", "Aerodynamic angles", "Control surfaces"):
        assert panel(figure, title).get_legend() is not None
    for title in ("Altitude", "Airspeed", "Heading", "Throttle"):
        assert panel(figure, title).get_legend() is None


def test_heading_panel_is_continuous_and_reaches_the_target(turning_flight):
    heading = panel(plot_flight(turning_flight), "Heading").get_lines()[0].get_ydata()
    assert np.abs(np.diff(heading)).max() < 5.0
    assert heading[-1] == pytest.approx(60.0, abs=2.0)


def test_continuous_heading_does_not_jump_across_north():
    yaw = np.radians([-10.0, -5.0, 0.0, 5.0, 10.0])
    assert continuous_heading_degrees(yaw) == pytest.approx([350.0, 355.0, 360.0, 365.0, 370.0])


def test_continuous_heading_starts_within_one_turn():
    heading = continuous_heading_degrees(np.radians([-170.0, 170.0]))
    assert 0.0 <= heading[0] < 360.0
    assert heading[1] - heading[0] == pytest.approx(-20.0)


@pytest.mark.parametrize(
    "values, expected",
    [
        ([0.0, 0.0, 5.0], [0.0, 1.0, 5.0]),
        ([0.0, 3.0, 6.0], [0.0, 3.0, 6.0]),
        ([5.0, 0.0, 0.5], [5.0, 0.0, 1.0]),
    ],
)
def test_direct_labels_are_pushed_apart(values, expected):
    assert separated_label_positions(values, minimum_gap=1.0) == pytest.approx(expected)


def test_ground_track_plots_east_horizontally_and_north_vertically(turning_flight):
    history = turning_flight.history()
    axes = plot_ground_track(history).axes[0]
    track = axes.get_lines()[0]
    assert track.get_xdata() == pytest.approx(history.east)
    assert track.get_ydata() == pytest.approx(history.north)
    assert axes.get_xlabel() == "east [m]"
    assert axes.get_ylabel() == "north [m]"


def test_ground_track_keeps_the_same_scale_on_both_axes(turning_flight):
    axes = plot_ground_track(turning_flight.history()).axes[0]
    assert axes.get_aspect() == 1.0


def test_ground_track_marks_start_and_end(turning_flight):
    axes = plot_ground_track(turning_flight.history()).axes[0]
    assert {text.get_text() for text in axes.texts} == {"start", "end"}


def test_modes_plot_shows_every_eigenvalue_and_names_each_mode(light_aircraft, trimmed):
    modes = aircraft_modes(linearize(light_aircraft, trimmed))
    axes = plot_modes(modes).axes[0]
    points = eigenvalue_markers(axes)
    drawn = sorted(zip(points.get_xdata(), points.get_ydata()))
    expected = []
    for mode in (modes.short_period, modes.phugoid, modes.dutch_roll):
        expected += [(mode.eigenvalue.real, mode.eigenvalue.imag), (mode.eigenvalue.real, -mode.eigenvalue.imag)]
    for mode in (modes.roll_subsidence, modes.spiral):
        expected.append((mode.eigenvalue.real, 0.0))
    assert np.array(drawn) == pytest.approx(np.array(sorted(expected)))
    labels = {text.get_text() for text in axes.texts}
    assert {"short period", "phugoid", "roll subsidence", "Dutch roll", "spiral", "unstable"} <= labels


def test_modes_plot_places_stable_modes_left_of_the_unstable_region(light_aircraft, trimmed):
    axes = plot_modes(aircraft_modes(linearize(light_aircraft, trimmed))).axes[0]
    assert max(eigenvalue_markers(axes).get_xdata()) < 0.0
    assert axes.get_xlim()[1] > 0.0
    assert axes.get_xscale() == "symlog"


@pytest.mark.parametrize("make_figure", ["flight", "track", "modes"])
def test_figures_can_be_saved_as_png(tmp_path, light_aircraft, trimmed, turning_flight, make_figure):
    figures = {
        "flight": lambda: plot_flight(turning_flight),
        "track": lambda: plot_ground_track(turning_flight.history()),
        "modes": lambda: plot_modes(aircraft_modes(linearize(light_aircraft, trimmed))),
    }
    path = tmp_path / f"{make_figure}.png"
    figures[make_figure]().savefig(path)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
