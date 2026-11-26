import math

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from alula.autopilot import Autopilot, AutopilotTargets
from alula.dynamics import POSITION_NED
from alula.rotations import quaternion_from_euler
from alula.simulation import simulate
from alula.trim import trim
from alula.view3d import (
    FIN_TIP_BODY,
    LEFT_WINGTIP_BODY,
    MINIMUM_AIRCRAFT_SCALE,
    NOSE_BODY,
    RIGHT_WINGTIP_BODY,
    TAIL_BODY,
    animate_flight,
    body_points_to_plot,
    default_aircraft_scale,
    ned_to_plot,
    plot_flight_3d,
    snapshot_indices,
)

pytestmark = pytest.mark.filterwarnings("ignore:Animation was deleted without rendering anything")

POSITION = np.array([0.0, 0.0, -1000.0])
SCALE = 10.0


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


@pytest.fixture
def turning_flight(light_aircraft):
    trimmed = trim(light_aircraft, 1000.0, 57.7)
    pilot = Autopilot(AutopilotTargets(1050.0, math.radians(90.0), 57.7), trimmed, 0.05)
    return simulate(light_aircraft, trimmed.state, pilot, 30.0, 0.05)


def point(body_point, roll=0.0, pitch=0.0, yaw=0.0):
    return body_points_to_plot(body_point, POSITION, quaternion_from_euler(roll, pitch, yaw), SCALE)[0]


def test_plot_axes_are_east_north_and_altitude():
    assert ned_to_plot([100.0, 20.0, -1200.0])[0] == pytest.approx([20.0, 100.0, 1200.0])


def test_higher_aircraft_is_drawn_higher():
    low, high = ned_to_plot([[0.0, 0.0, -1000.0], [0.0, 0.0, -1200.0]])
    assert high[2] > low[2]


def test_level_aircraft_heading_north():
    assert point(NOSE_BODY) == pytest.approx([0.0, 5.0, 1000.0])
    assert point(RIGHT_WINGTIP_BODY) == pytest.approx([5.5, 0.0, 1000.0])
    assert point(FIN_TIP_BODY)[2] > 1000.0


def test_heading_east_points_nose_east_and_right_wing_south():
    yaw = math.radians(90.0)
    assert point(NOSE_BODY, yaw=yaw) == pytest.approx([5.0, 0.0, 1000.0], abs=1e-12)
    assert point(RIGHT_WINGTIP_BODY, yaw=yaw)[1] < 0.0


def test_rolling_right_lowers_the_right_wing():
    roll = math.radians(30.0)
    assert point(RIGHT_WINGTIP_BODY, roll=roll)[2] < point(LEFT_WINGTIP_BODY, roll=roll)[2]


def test_pitching_up_raises_the_nose_above_the_tail():
    pitch = math.radians(10.0)
    assert point(NOSE_BODY, pitch=pitch)[2] > point(TAIL_BODY, pitch=pitch)[2]


def test_snapshot_indices_follow_the_interval():
    times = np.arange(0.0, 10.01, 0.5)
    assert snapshot_indices(times, 2.0) == [0, 4, 8, 12, 16, 20]


def test_snapshot_indices_reject_non_positive_interval():
    with pytest.raises(ValueError, match="interval"):
        snapshot_indices(np.arange(0.0, 10.0, 1.0), 0.0)


def test_aircraft_scale_grows_with_the_flight_but_never_vanishes(light_aircraft, turning_flight):
    trimmed = trim(light_aircraft, 1000.0, 57.7)
    short = simulate(light_aircraft, trimmed.state, trimmed.controls, 0.1, 0.05)
    assert default_aircraft_scale(short) == MINIMUM_AIRCRAFT_SCALE
    assert default_aircraft_scale(turning_flight) > MINIMUM_AIRCRAFT_SCALE


def trajectory_lines(axes):
    return {line.get_label(): line for line in axes.get_lines()}


def test_3d_figure_draws_trajectory_and_ground_shadow(turning_flight):
    axes = plot_flight_3d(turning_flight).axes[0]
    lines = trajectory_lines(axes)
    expected = ned_to_plot(turning_flight.states[:, POSITION_NED])
    drawn = np.column_stack(lines["trajectory"].get_data_3d())
    assert drawn == pytest.approx(expected)
    shadow = np.column_stack(lines["ground track"].get_data_3d())
    assert shadow[:, :2] == pytest.approx(expected[:, :2])
    assert np.all(shadow[:, 2] == 0.0)


def test_3d_figure_draws_one_aircraft_per_snapshot(turning_flight):
    axes = plot_flight_3d(turning_flight, snapshot_interval=10.0).axes[0]
    aircraft = [artist for artist in axes.collections if isinstance(artist, Poly3DCollection)]
    assert len(aircraft) == len(snapshot_indices(turning_flight.times, 10.0)) == 4


def test_3d_figure_axes_start_at_the_ground(turning_flight):
    axes = plot_flight_3d(turning_flight).axes[0]
    assert axes.get_zlim()[0] == 0.0
    assert (axes.get_xlabel(), axes.get_ylabel(), axes.get_zlabel()) == ("east [m]", "north [m]", "altitude [m]")


def test_flat_long_flight_warns_about_vertical_exaggeration(light_aircraft):
    trimmed = trim(light_aircraft, 300.0, 57.7)
    flight = simulate(light_aircraft, trimmed.state, trimmed.controls, 120.0, 0.1)
    title = plot_flight_3d(flight).axes[0].get_title(loc="left")
    assert "exaggerated" in title


def test_animation_frames_follow_the_requested_interval(turning_flight):
    animation = animate_flight(turning_flight, seconds_per_frame=5.0)
    assert animation.frame_indices == snapshot_indices(turning_flight.times, 5.0)


def test_animation_frame_places_aircraft_and_trail(turning_flight):
    animation = animate_flight(turning_flight, seconds_per_frame=5.0)
    frame = 3
    index = animation.frame_indices[frame]
    polygons = animation.show_frame(frame)
    centre = np.vstack(list(polygons.values())).mean(axis=0)
    expected = ned_to_plot(turning_flight.states[index, POSITION_NED])[0]
    assert np.linalg.norm(centre - expected) < default_aircraft_scale(turning_flight)
    assert len(animation.trail.get_data_3d()[0]) == index + 1
    assert animation.figure.axes[0].get_title(loc="left").startswith(f"t = {turning_flight.times[index]:.0f} s")


def test_animation_rejects_non_positive_playback_speed(turning_flight):
    with pytest.raises(ValueError, match="Playback speed"):
        animate_flight(turning_flight, playback_speed=0.0)


def test_animation_can_be_saved_as_gif(tmp_path, light_aircraft):
    trimmed = trim(light_aircraft, 1000.0, 57.7)
    flight = simulate(light_aircraft, trimmed.state, trimmed.controls, 6.0, 0.1)
    path = tmp_path / "flight.gif"
    animate_flight(flight, seconds_per_frame=2.0).save_gif(path, frames_per_second=5)
    assert path.read_bytes()[:6] in (b"GIF89a", b"GIF87a")
