import math
from dataclasses import dataclass, field

from alula.aerodynamics import air_data_from_body_velocity
from alula.controls import Controls
from alula.dynamics import State
from alula.pid import PID
from alula.rotations import euler_from_quaternion
from alula.trim import TrimResult

ELEVATOR_LIMITS = (math.radians(-25.0), math.radians(20.0))
AILERON_LIMIT = math.radians(20.0)
RUDDER_LIMIT = math.radians(25.0)
TIME_TOLERANCE = 1e-9


@dataclass(frozen=True)
class AutopilotTargets:
    altitude: float
    heading: float
    airspeed: float

    def __post_init__(self) -> None:
        if self.altitude < 0.0:
            raise ValueError(f"Target altitude cannot be below sea level, got {self.altitude} m")
        if self.airspeed <= 0.0:
            raise ValueError(f"Target airspeed must be positive, got {self.airspeed} m/s")


@dataclass(frozen=True)
class LoopGains:
    proportional: float
    integral: float = 0.0
    derivative: float = 0.0


@dataclass(frozen=True)
class AutopilotGains:
    altitude_to_pitch: LoopGains = field(default_factory=lambda: LoopGains(0.003, 0.0001, 0.01))
    pitch_to_elevator: LoopGains = field(default_factory=lambda: LoopGains(2.0, 0.5, 0.5))
    heading_to_bank: LoopGains = field(default_factory=lambda: LoopGains(1.0))
    bank_to_aileron: LoopGains = field(default_factory=lambda: LoopGains(2.0, 0.0, 0.3))
    sideslip_to_rudder: LoopGains = field(default_factory=lambda: LoopGains(10.0, 4.0))
    airspeed_to_throttle: LoopGains = field(default_factory=lambda: LoopGains(0.1, 0.02))
    max_pitch_offset: float = math.radians(6.0)
    max_bank: float = math.radians(25.0)


def angle_difference(target: float, current: float) -> float:
    return (target - current + math.pi) % (2.0 * math.pi) - math.pi


class Autopilot:
    def __init__(
        self,
        targets: AutopilotTargets,
        trimmed: TrimResult,
        time_step: float,
        gains: AutopilotGains | None = None,
    ) -> None:
        if time_step <= 0.0:
            raise ValueError(f"Time step must be positive, got {time_step} s")
        gains = gains or AutopilotGains()
        trim_controls = trimmed.controls
        self.targets = targets
        self.time_step = time_step
        self.trim_controls = trim_controls
        self.trim_pitch = euler_from_quaternion(trimmed.state.quaternion).pitch
        self._last_time: float | None = None

        self.altitude_loop = _pid(gains.altitude_to_pitch, -gains.max_pitch_offset, gains.max_pitch_offset)
        self.pitch_loop = _pid(
            gains.pitch_to_elevator,
            trim_controls.elevator - ELEVATOR_LIMITS[1],
            trim_controls.elevator - ELEVATOR_LIMITS[0],
        )
        self.heading_loop = _pid(gains.heading_to_bank, -gains.max_bank, gains.max_bank)
        self.bank_loop = _pid(gains.bank_to_aileron, -AILERON_LIMIT, AILERON_LIMIT)
        self.sideslip_loop = _pid(gains.sideslip_to_rudder, -RUDDER_LIMIT, RUDDER_LIMIT)
        self.airspeed_loop = _pid(
            gains.airspeed_to_throttle, -trim_controls.throttle, 1.0 - trim_controls.throttle
        )

    def __call__(self, time: float, state: State) -> Controls:
        self._check_timing(time)
        step = self.time_step
        targets = self.targets
        attitude = euler_from_quaternion(state.quaternion)
        air_data = air_data_from_body_velocity(state.velocity_body)

        pitch_reference = self.trim_pitch + self.altitude_loop.update(targets.altitude, state.altitude, step)
        nose_up_demand = self.pitch_loop.update(pitch_reference, attitude.pitch, step)

        heading_error = angle_difference(targets.heading, attitude.yaw)
        bank_reference = self.heading_loop.update(heading_error, 0.0, step)
        aileron = self.bank_loop.update(bank_reference, attitude.roll, step)

        rudder = self.sideslip_loop.update(0.0, air_data.beta, step)
        extra_throttle = self.airspeed_loop.update(targets.airspeed, air_data.airspeed, step)

        return Controls(
            elevator=self.trim_controls.elevator - nose_up_demand,
            aileron=aileron,
            rudder=rudder,
            throttle=min(max(self.trim_controls.throttle + extra_throttle, 0.0), 1.0),
        )

    def _check_timing(self, time: float) -> None:
        if self._last_time is not None and not math.isclose(
            time - self._last_time, self.time_step, abs_tol=TIME_TOLERANCE
        ):
            raise ValueError(
                f"Autopilot expects calls every {self.time_step} s, "
                f"got {time - self._last_time} s since the previous call"
            )
        self._last_time = time


def _pid(gains: LoopGains, output_min: float, output_max: float) -> PID:
    return PID(
        proportional_gain=gains.proportional,
        integral_gain=gains.integral,
        derivative_gain=gains.derivative,
        output_min=output_min,
        output_max=output_max,
    )
