import math

import numpy as np
import pytest

from alula.atmosphere import STANDARD_GRAVITY
from alula.controls import Controls
from alula.dynamics import (
    ANGULAR_VELOCITY_BODY,
    POSITION_NED,
    QUATERNION,
    STATE_SIZE,
    VELOCITY_BODY,
    State,
    aircraft_derivatives,
    aircraft_loads,
    rigid_body_derivatives,
)
from alula.rotations import quaternion_derivative, quaternion_from_euler

MASS = 1000.0
INERTIA = np.diag([1285.0, 1825.0, 2667.0])
NO_LOAD = np.zeros(3)

CRUISE_AIRSPEED = 55.0
TRIM_ALPHA = 0.019909
TRIM_ELEVATOR = -0.05057
TRIM_THROTTLE = 0.7711


def make_state(
    position=(0.0, 0.0, 0.0),
    velocity=(0.0, 0.0, 0.0),
    roll=0.0,
    pitch=0.0,
    yaw=0.0,
    angular_velocity=(0.0, 0.0, 0.0),
):
    return State(
        position_ned=position,
        velocity_body=velocity,
        quaternion=quaternion_from_euler(roll, pitch, yaw),
        angular_velocity_body=angular_velocity,
    )


def rigid_body(state, force=NO_LOAD, moment=NO_LOAD, inertia=INERTIA):
    return rigid_body_derivatives(state, force, moment, MASS, inertia)


def trimmed_cruise_state(altitude=0.0):
    return make_state(
        position=(0.0, 0.0, -altitude),
        velocity=(
            CRUISE_AIRSPEED * math.cos(TRIM_ALPHA),
            0.0,
            CRUISE_AIRSPEED * math.sin(TRIM_ALPHA),
        ),
        pitch=TRIM_ALPHA,
    )


TRIM_CONTROLS = Controls(elevator=TRIM_ELEVATOR, throttle=TRIM_THROTTLE)


def test_state_survives_round_trip_through_array():
    values = np.arange(STATE_SIZE, dtype=float)
    assert State.from_array(values).to_array() == pytest.approx(values)


def test_state_array_layout():
    state = make_state(position=(1.0, 2.0, -3.0), velocity=(4.0, 5.0, 6.0), angular_velocity=(7.0, 8.0, 9.0))
    values = state.to_array()
    assert values[POSITION_NED] == pytest.approx([1.0, 2.0, -3.0])
    assert values[VELOCITY_BODY] == pytest.approx([4.0, 5.0, 6.0])
    assert values[QUATERNION] == pytest.approx([1.0, 0.0, 0.0, 0.0])
    assert values[ANGULAR_VELOCITY_BODY] == pytest.approx([7.0, 8.0, 9.0])


def test_altitude_is_minus_down_coordinate():
    assert make_state(position=(0.0, 0.0, -1500.0)).altitude == 1500.0


def test_state_keeps_its_own_copy_of_the_data():
    position = np.array([0.0, 0.0, -100.0])
    state = make_state(position=position)
    position[2] = 0.0
    assert state.altitude == 100.0


def test_state_rejects_wrong_component_sizes():
    with pytest.raises(ValueError, match="quaternion"):
        State([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0])


def test_state_rejects_wrong_array_length():
    with pytest.raises(ValueError):
        State.from_array(np.zeros(12))


def test_free_fall_accelerates_down_at_g():
    derivatives = rigid_body(make_state())
    assert derivatives[VELOCITY_BODY] == pytest.approx([0.0, 0.0, STANDARD_GRAVITY])
    assert derivatives[POSITION_NED] == pytest.approx(np.zeros(3))
    assert derivatives[QUATERNION] == pytest.approx(np.zeros(4))
    assert derivatives[ANGULAR_VELOCITY_BODY] == pytest.approx(np.zeros(3))


def test_gravity_in_body_axes_with_nose_up():
    derivatives = rigid_body(make_state(pitch=math.radians(10.0)))
    assert derivatives[VELOCITY_BODY] == pytest.approx([-1.7029, 0.0, 9.6577], abs=1e-4)


def test_upward_force_equal_to_weight_cancels_gravity():
    derivatives = rigid_body(make_state(velocity=(50.0, 0.0, 0.0)), force=[0.0, 0.0, -MASS * STANDARD_GRAVITY])
    assert derivatives[VELOCITY_BODY] == pytest.approx(np.zeros(3), abs=1e-12)


def test_centripetal_force_keeps_body_velocity_constant_in_a_turn():
    state = make_state(velocity=(50.0, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.2))
    weight_support = [0.0, 0.0, -MASS * STANDARD_GRAVITY]
    centripetal = [0.0, MASS * 0.2 * 50.0, -MASS * STANDARD_GRAVITY]
    assert rigid_body(state, force=weight_support)[VELOCITY_BODY] == pytest.approx([0.0, -10.0, 0.0])
    assert rigid_body(state, force=centripetal)[VELOCITY_BODY] == pytest.approx(np.zeros(3), abs=1e-12)


def test_pitching_moment_gives_pitch_acceleration():
    derivatives = rigid_body(make_state(), moment=[0.0, 2.0 * 1825.0, 0.0])
    assert derivatives[ANGULAR_VELOCITY_BODY] == pytest.approx([0.0, 2.0, 0.0])


def test_spin_about_principal_axis_needs_no_moment():
    derivatives = rigid_body(make_state(angular_velocity=(0.5, 0.0, 0.0)))
    assert derivatives[ANGULAR_VELOCITY_BODY] == pytest.approx(np.zeros(3), abs=1e-12)


def test_rolling_and_yawing_together_couple_into_pitch():
    derivatives = rigid_body(make_state(angular_velocity=(1.0, 0.0, 1.0)))
    expected_pitch_acceleration = -(1285.0 - 2667.0) / 1825.0
    assert derivatives[ANGULAR_VELOCITY_BODY] == pytest.approx([0.0, expected_pitch_acceleration, 0.0])


def test_product_of_inertia_couples_roll_into_yaw():
    inertia = np.array([[1285.0, 0.0, -50.0], [0.0, 1825.0, 0.0], [-50.0, 0.0, 2667.0]])
    derivatives = rigid_body(make_state(), moment=[1000.0, 0.0, 0.0], inertia=inertia)
    roll_acceleration, _, yaw_acceleration = derivatives[ANGULAR_VELOCITY_BODY]
    assert roll_acceleration > 0.0
    assert yaw_acceleration == pytest.approx(50.0 * roll_acceleration / 2667.0)


def test_heading_east_moves_east():
    derivatives = rigid_body(make_state(velocity=(10.0, 0.0, 0.0), yaw=math.radians(90.0)))
    assert derivatives[POSITION_NED] == pytest.approx([0.0, 10.0, 0.0], abs=1e-12)


def test_nose_up_climbs():
    derivatives = rigid_body(make_state(velocity=(10.0, 0.0, 0.0), pitch=math.radians(30.0)))
    assert derivatives[POSITION_NED] == pytest.approx([10.0 * math.cos(math.radians(30.0)), 0.0, -5.0])


def test_attitude_rate_comes_from_quaternion_kinematics():
    state = make_state(roll=0.2, pitch=0.1, yaw=1.0, angular_velocity=(0.3, -0.2, 0.1))
    expected = quaternion_derivative(state.quaternion, state.angular_velocity_body)
    assert rigid_body(state)[QUATERNION] == pytest.approx(expected)


def test_rejects_non_positive_mass():
    with pytest.raises(ValueError, match="Mass"):
        rigid_body_derivatives(make_state(), NO_LOAD, NO_LOAD, 0.0, INERTIA)


def test_rejects_zero_quaternion():
    state = State([0.0, 0.0, 0.0], [50.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        rigid_body(state)


def test_throttle_only_changes_axial_force(light_aircraft):
    state = trimmed_cruise_state()
    idle = aircraft_loads(state, Controls(elevator=TRIM_ELEVATOR), light_aircraft)
    powered = aircraft_loads(state, TRIM_CONTROLS, light_aircraft)
    difference = powered.force_body - idle.force_body
    assert difference[0] == pytest.approx(981.4, rel=1e-3)
    assert difference[1:] == pytest.approx([0.0, 0.0])
    assert powered.moment_body == pytest.approx(idle.moment_body)


def test_trimmed_cruise_from_primer_is_in_equilibrium(light_aircraft):
    derivatives = aircraft_derivatives(trimmed_cruise_state(), TRIM_CONTROLS, light_aircraft)
    assert derivatives[VELOCITY_BODY] == pytest.approx(np.zeros(3), abs=0.05)
    assert derivatives[ANGULAR_VELOCITY_BODY] == pytest.approx(np.zeros(3), abs=1e-3)
    assert derivatives[QUATERNION] == pytest.approx(np.zeros(4))
    assert derivatives[POSITION_NED] == pytest.approx([CRUISE_AIRSPEED, 0.0, 0.0], abs=1e-9)


def test_symmetric_flight_has_no_lateral_accelerations(light_aircraft):
    derivatives = aircraft_derivatives(trimmed_cruise_state(), TRIM_CONTROLS, light_aircraft)
    velocity_rate = derivatives[VELOCITY_BODY]
    angular_acceleration = derivatives[ANGULAR_VELOCITY_BODY]
    assert velocity_rate[1] == 0.0
    assert angular_acceleration[0] == 0.0
    assert angular_acceleration[2] == 0.0


def test_cutting_the_engine_decelerates_the_aircraft(light_aircraft):
    derivatives = aircraft_derivatives(trimmed_cruise_state(), Controls(elevator=TRIM_ELEVATOR), light_aircraft)
    assert derivatives[VELOCITY_BODY][0] < -0.9


def test_aircraft_below_ground_is_rejected(light_aircraft):
    with pytest.raises(ValueError, match="ISA"):
        aircraft_derivatives(trimmed_cruise_state(altitude=-10.0), TRIM_CONTROLS, light_aircraft)
