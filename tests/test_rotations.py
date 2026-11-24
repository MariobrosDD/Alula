import numpy as np
import pytest

from alula.rotations import (
    body_to_ned_from_euler,
    body_to_ned_from_quaternion,
    euler_from_body_to_ned,
    euler_from_quaternion,
    normalize_quaternion,
    quaternion_derivative,
    quaternion_from_euler,
)

NOSE = np.array([1.0, 0.0, 0.0])
RIGHT_WING = np.array([0.0, 1.0, 0.0])
NORTH = np.array([1.0, 0.0, 0.0])
EAST = np.array([0.0, 1.0, 0.0])
DOWN = np.array([0.0, 0.0, 1.0])

GENERIC_ATTITUDES = [
    (0.3, 0.2, 0.1),
    (-1.2, 0.7, 2.5),
    (2.9, -1.1, -3.0),
    (0.0, -0.4, 1.6),
]


def test_zero_angles_give_identity():
    assert body_to_ned_from_euler(0.0, 0.0, 0.0) == pytest.approx(np.eye(3))


def test_yaw_of_90_degrees_points_nose_east():
    rotation = body_to_ned_from_euler(roll=0.0, pitch=0.0, yaw=np.radians(90))
    assert rotation @ NOSE == pytest.approx(EAST, abs=1e-12)


def test_positive_pitch_raises_the_nose():
    pitch = np.radians(30)
    rotation = body_to_ned_from_euler(roll=0.0, pitch=pitch, yaw=0.0)
    expected = np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    assert rotation @ NOSE == pytest.approx(expected, abs=1e-12)


def test_roll_of_90_degrees_points_right_wing_down():
    rotation = body_to_ned_from_euler(roll=np.radians(90), pitch=0.0, yaw=0.0)
    assert rotation @ RIGHT_WING == pytest.approx(DOWN, abs=1e-12)


def test_gravity_in_body_axes_with_nose_up():
    gravity_ned = np.array([0.0, 0.0, 9.81])
    rotation = body_to_ned_from_euler(roll=0.0, pitch=np.radians(10), yaw=0.0)
    gravity_body = rotation.T @ gravity_ned
    assert gravity_body == pytest.approx([-1.7035, 0.0, 9.6610], abs=1e-4)


@pytest.mark.parametrize("roll, pitch, yaw", GENERIC_ATTITUDES)
def test_rotation_matrix_is_proper_orthogonal(roll, pitch, yaw):
    rotation = body_to_ned_from_euler(roll, pitch, yaw)
    assert rotation @ rotation.T == pytest.approx(np.eye(3), abs=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


@pytest.mark.parametrize("roll, pitch, yaw", GENERIC_ATTITUDES)
def test_euler_survives_round_trip_through_matrix(roll, pitch, yaw):
    recovered = euler_from_body_to_ned(body_to_ned_from_euler(roll, pitch, yaw))
    assert recovered == pytest.approx((roll, pitch, yaw), abs=1e-12)


@pytest.mark.parametrize("roll, pitch, yaw", GENERIC_ATTITUDES)
def test_euler_survives_round_trip_through_quaternion(roll, pitch, yaw):
    recovered = euler_from_quaternion(quaternion_from_euler(roll, pitch, yaw))
    assert recovered == pytest.approx((roll, pitch, yaw), abs=1e-12)


@pytest.mark.parametrize("roll, pitch, yaw", GENERIC_ATTITUDES)
def test_quaternion_and_euler_describe_same_rotation(roll, pitch, yaw):
    from_quaternion = body_to_ned_from_quaternion(quaternion_from_euler(roll, pitch, yaw))
    from_euler = body_to_ned_from_euler(roll, pitch, yaw)
    assert from_quaternion == pytest.approx(from_euler, abs=1e-12)


@pytest.mark.parametrize("roll, pitch, yaw", GENERIC_ATTITUDES)
def test_quaternion_from_euler_has_unit_norm(roll, pitch, yaw):
    assert np.linalg.norm(quaternion_from_euler(roll, pitch, yaw)) == pytest.approx(1.0)


@pytest.mark.parametrize("pitch", [np.pi / 2, -np.pi / 2])
def test_gimbal_lock_still_returns_an_equivalent_attitude(pitch):
    rotation = body_to_ned_from_euler(roll=0.3, pitch=pitch, yaw=0.5)
    recovered = euler_from_body_to_ned(rotation)
    assert recovered.pitch == pytest.approx(pitch)
    assert recovered.roll == 0.0
    assert body_to_ned_from_euler(*recovered) == pytest.approx(rotation, abs=1e-12)


def test_normalize_quaternion_gives_unit_norm():
    assert normalize_quaternion([2.0, 0.0, 0.0, 0.0]) == pytest.approx([1.0, 0.0, 0.0, 0.0])


def test_normalize_rejects_zero_quaternion():
    with pytest.raises(ValueError):
        normalize_quaternion([0.0, 0.0, 0.0, 0.0])


def test_quaternion_derivative_at_identity():
    derivative = quaternion_derivative([1.0, 0.0, 0.0, 0.0], [0.1, 0.2, 0.3])
    assert derivative == pytest.approx([0.0, 0.05, 0.1, 0.15])


@pytest.mark.parametrize("roll, pitch, yaw", GENERIC_ATTITUDES)
def test_quaternion_derivative_preserves_norm(roll, pitch, yaw):
    quaternion = quaternion_from_euler(roll, pitch, yaw)
    derivative = quaternion_derivative(quaternion, [0.4, -0.7, 1.1])
    assert np.dot(quaternion, derivative) == pytest.approx(0.0, abs=1e-15)


def test_constant_yaw_rate_turns_heading_linearly():
    yaw_rate = 0.5
    time_step = 1e-3
    duration = 1.0
    quaternion = np.array([1.0, 0.0, 0.0, 0.0])
    for _ in range(round(duration / time_step)):
        quaternion = normalize_quaternion(
            quaternion + time_step * quaternion_derivative(quaternion, [0.0, 0.0, yaw_rate])
        )
    attitude = euler_from_quaternion(quaternion)
    assert attitude.yaw == pytest.approx(yaw_rate * duration, abs=1e-6)
    assert attitude.roll == pytest.approx(0.0, abs=1e-12)
    assert attitude.pitch == pytest.approx(0.0, abs=1e-12)
