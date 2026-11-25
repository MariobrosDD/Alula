import numpy as np
import pytest

from alula.atmosphere import SEA_LEVEL_DENSITY, isa
from alula.propulsion import altitude_power_fraction, shaft_power, thrust

CROSSOVER_AIRSPEED = 28.0


@pytest.fixture
def engine(light_aircraft):
    return light_aircraft.propulsion


def test_full_power_at_sea_level(engine):
    assert altitude_power_fraction(SEA_LEVEL_DENSITY) == pytest.approx(1.0)
    assert shaft_power(engine, throttle=1.0, density=SEA_LEVEL_DENSITY) == pytest.approx(100_000.0)


def test_engine_keeps_71_percent_of_power_at_3000_m():
    assert altitude_power_fraction(isa(3_000.0).density) == pytest.approx(0.708, abs=1e-3)


def test_full_throttle_cruise_thrust_matches_primer(engine):
    assert thrust(engine, 1.0, 55.0, SEA_LEVEL_DENSITY) == pytest.approx(1272.7, abs=0.1)


def test_trim_throttle_balances_cruise_drag(engine):
    assert thrust(engine, 0.77, 55.0, SEA_LEVEL_DENSITY) == pytest.approx(980.0, abs=0.1)


@pytest.mark.parametrize("throttle", [0.25, 0.5, 1.0])
def test_standing_still_gives_static_thrust(engine, throttle):
    assert thrust(engine, throttle, 0.0, SEA_LEVEL_DENSITY) == pytest.approx(2500.0 * throttle)


def test_low_airspeed_is_limited_to_static_thrust(engine):
    assert thrust(engine, 1.0, 10.0, SEA_LEVEL_DENSITY) == pytest.approx(2500.0)


@pytest.mark.parametrize("airspeed", [0.0, 20.0, 55.0])
def test_idle_produces_no_thrust(engine, airspeed):
    assert thrust(engine, 0.0, airspeed, SEA_LEVEL_DENSITY) == 0.0


@pytest.mark.parametrize("throttle", [0.3, 1.0])
@pytest.mark.parametrize("density", [SEA_LEVEL_DENSITY, isa(3_000.0).density])
def test_thrust_is_continuous_at_crossover_airspeed(engine, throttle, density):
    below = thrust(engine, throttle, CROSSOVER_AIRSPEED - 1e-6, density)
    above = thrust(engine, throttle, CROSSOVER_AIRSPEED + 1e-6, density)
    assert above == pytest.approx(below, rel=1e-6)


def test_thrust_falls_with_airspeed_above_crossover(engine):
    airspeeds = np.linspace(CROSSOVER_AIRSPEED + 1.0, 80.0, 20)
    thrusts = [thrust(engine, 1.0, airspeed, SEA_LEVEL_DENSITY) for airspeed in airspeeds]
    assert all(faster < slower for slower, faster in zip(thrusts, thrusts[1:]))


def test_thrust_grows_with_throttle(engine):
    throttles = np.linspace(0.0, 1.0, 11)
    thrusts = [thrust(engine, throttle, 40.0, SEA_LEVEL_DENSITY) for throttle in throttles]
    assert all(more > less for less, more in zip(thrusts, thrusts[1:]))


def test_thrust_falls_with_altitude(engine):
    thrusts = [thrust(engine, 1.0, 55.0, isa(altitude).density) for altitude in range(0, 12_001, 1_000)]
    assert all(higher < lower for lower, higher in zip(thrusts, thrusts[1:]))


def test_rejects_negative_airspeed(engine):
    with pytest.raises(ValueError, match="Airspeed"):
        thrust(engine, 1.0, -1.0, SEA_LEVEL_DENSITY)


@pytest.mark.parametrize("throttle", [-0.1, 1.1])
def test_rejects_throttle_outside_range(engine, throttle):
    with pytest.raises(ValueError, match="Throttle"):
        thrust(engine, throttle, 55.0, SEA_LEVEL_DENSITY)


def test_rejects_air_too_thin_for_piston_engine(engine):
    with pytest.raises(ValueError, match="too low"):
        thrust(engine, 1.0, 55.0, isa(20_000.0).density)
