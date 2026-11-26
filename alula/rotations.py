from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

GIMBAL_LOCK_TOLERANCE = 1e-9


class EulerAngles(NamedTuple):
    roll: float
    pitch: float
    yaw: float


def body_to_ned_from_euler(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [
                cos_pitch * cos_yaw,
                sin_roll * sin_pitch * cos_yaw - cos_roll * sin_yaw,
                cos_roll * sin_pitch * cos_yaw + sin_roll * sin_yaw,
            ],
            [
                cos_pitch * sin_yaw,
                sin_roll * sin_pitch * sin_yaw + cos_roll * cos_yaw,
                cos_roll * sin_pitch * sin_yaw - sin_roll * cos_yaw,
            ],
            [
                -sin_pitch,
                sin_roll * cos_pitch,
                cos_roll * cos_pitch,
            ],
        ]
    )


def euler_from_body_to_ned(body_to_ned: ArrayLike) -> EulerAngles:
    matrix = np.asarray(body_to_ned, dtype=float)
    cos_pitch = np.hypot(matrix[0, 0], matrix[1, 0])
    pitch = np.arctan2(-matrix[2, 0], cos_pitch)

    if cos_pitch < GIMBAL_LOCK_TOLERANCE:
        roll = 0.0
        yaw = np.arctan2(-matrix[0, 1], matrix[1, 1])
    else:
        roll = np.arctan2(matrix[2, 1], matrix[2, 2])
        yaw = np.arctan2(matrix[1, 0], matrix[0, 0])

    return EulerAngles(float(roll), float(pitch), float(yaw))


def quaternion_from_euler(roll: float, pitch: float, yaw: float) -> NDArray[np.float64]:
    cos_half_roll, sin_half_roll = np.cos(roll / 2), np.sin(roll / 2)
    cos_half_pitch, sin_half_pitch = np.cos(pitch / 2), np.sin(pitch / 2)
    cos_half_yaw, sin_half_yaw = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.array(
        [
            cos_half_roll * cos_half_pitch * cos_half_yaw
            + sin_half_roll * sin_half_pitch * sin_half_yaw,
            sin_half_roll * cos_half_pitch * cos_half_yaw
            - cos_half_roll * sin_half_pitch * sin_half_yaw,
            cos_half_roll * sin_half_pitch * cos_half_yaw
            + sin_half_roll * cos_half_pitch * sin_half_yaw,
            cos_half_roll * cos_half_pitch * sin_half_yaw
            - sin_half_roll * sin_half_pitch * cos_half_yaw,
        ]
    )


def body_to_ned_from_quaternion(quaternion: ArrayLike) -> NDArray[np.float64]:
    q0, q1, q2, q3 = normalize_quaternion(quaternion)
    return np.array(
        [
            [
                q0**2 + q1**2 - q2**2 - q3**2,
                2 * (q1 * q2 - q0 * q3),
                2 * (q1 * q3 + q0 * q2),
            ],
            [
                2 * (q1 * q2 + q0 * q3),
                q0**2 - q1**2 + q2**2 - q3**2,
                2 * (q2 * q3 - q0 * q1),
            ],
            [
                2 * (q1 * q3 - q0 * q2),
                2 * (q2 * q3 + q0 * q1),
                q0**2 - q1**2 - q2**2 + q3**2,
            ],
        ]
    )


def euler_from_quaternion(quaternion: ArrayLike) -> EulerAngles:
    return euler_from_body_to_ned(body_to_ned_from_quaternion(quaternion))


def normalize_quaternion(quaternion: ArrayLike) -> NDArray[np.float64]:
    components = np.asarray(quaternion, dtype=float)
    norm = np.linalg.norm(components)
    if norm == 0.0:
        raise ValueError("A zero quaternion does not represent any orientation")
    return components / norm


def euler_angle_rates(
    roll: float, pitch: float, angular_velocity: ArrayLike
) -> NDArray[np.float64]:
    cos_pitch = np.cos(pitch)
    if abs(cos_pitch) < GIMBAL_LOCK_TOLERANCE:
        raise ValueError("Euler angle rates are undefined at gimbal lock (pitch of +-90 deg)")
    p, q, r = np.asarray(angular_velocity, dtype=float)
    sin_roll, cos_roll = np.sin(roll), np.cos(roll)
    off_axis_rate = q * sin_roll + r * cos_roll
    return np.array(
        [
            p + off_axis_rate * np.tan(pitch),
            q * cos_roll - r * sin_roll,
            off_axis_rate / cos_pitch,
        ]
    )


def quaternion_derivative(
    quaternion: ArrayLike, angular_velocity: ArrayLike
) -> NDArray[np.float64]:
    p, q, r = np.asarray(angular_velocity, dtype=float)
    rate_matrix = np.array(
        [
            [0.0, -p, -q, -r],
            [p, 0.0, r, -q],
            [q, -r, 0.0, p],
            [r, q, -p, 0.0],
        ]
    )
    return 0.5 * rate_matrix @ np.asarray(quaternion, dtype=float)
