import pytest

from alula.controls import Controls


def test_default_controls_are_neutral_and_idle():
    controls = Controls()
    assert (controls.elevator, controls.aileron, controls.rudder, controls.throttle) == (
        0.0,
        0.0,
        0.0,
        0.0,
    )


@pytest.mark.parametrize("throttle", [0.0, 0.5, 1.0])
def test_accepts_throttle_within_range(throttle):
    assert Controls(throttle=throttle).throttle == throttle


@pytest.mark.parametrize("throttle", [-0.01, 1.01])
def test_rejects_throttle_outside_range(throttle):
    with pytest.raises(ValueError):
        Controls(throttle=throttle)
