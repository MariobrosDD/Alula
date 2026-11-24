from dataclasses import dataclass


@dataclass(frozen=True)
class Controls:
    elevator: float = 0.0
    aileron: float = 0.0
    rudder: float = 0.0
    throttle: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.throttle <= 1.0:
            raise ValueError(f"Throttle must be between 0 and 1, got {self.throttle}")
