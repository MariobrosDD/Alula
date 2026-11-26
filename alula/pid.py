import math
from dataclasses import dataclass, field


@dataclass
class PID:
    proportional_gain: float
    integral_gain: float = 0.0
    derivative_gain: float = 0.0
    output_min: float = -math.inf
    output_max: float = math.inf
    integral: float = field(default=0.0, init=False)
    previous_measurement: float | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.output_min >= self.output_max:
            raise ValueError(
                f"output_min ({self.output_min}) must be smaller than output_max ({self.output_max})"
            )

    def update(self, reference: float, measurement: float, time_step: float) -> float:
        if time_step <= 0.0:
            raise ValueError(f"Time step must be positive, got {time_step} s")

        error = reference - measurement
        measurement_rate = (
            0.0
            if self.previous_measurement is None
            else (measurement - self.previous_measurement) / time_step
        )
        self.previous_measurement = measurement

        candidate_integral = self.integral + error * time_step
        unlimited_output = (
            self.proportional_gain * error
            + self.integral_gain * candidate_integral
            - self.derivative_gain * measurement_rate
        )
        output = min(max(unlimited_output, self.output_min), self.output_max)

        integral_push = self.integral_gain * error
        winding_up = (unlimited_output > self.output_max and integral_push > 0.0) or (
            unlimited_output < self.output_min and integral_push < 0.0
        )
        if not winding_up:
            self.integral = candidate_integral

        return output

    def reset(self) -> None:
        self.integral = 0.0
        self.previous_measurement = None
