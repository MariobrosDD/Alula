import pytest

from alula.atmosphere import (
    GAS_CONSTANT_AIR,
    TROPOPAUSE_ALTITUDE,
    dynamic_pressure,
    isa,
)

ICAO_REFERENCE = [
    (0.0, 288.15, 101_325.0, 1.22500, 340.294),
    (3_000.0, 268.65, 70_108.5, 0.909122, 328.584),
    (11_000.0, 216.65, 22_632.0, 0.363918, 295.070),
    (20_000.0, 216.65, 5_474.89, 0.0880349, 295.070),
]


@pytest.mark.parametrize(
    "altitude, temperature, pressure, density, speed_of_sound", ICAO_REFERENCE
)
def test_matches_icao_standard_atmosphere(
    altitude, temperature, pressure, density, speed_of_sound
):
    air = isa(altitude)
    assert air.temperature == pytest.approx(temperature, rel=1e-6)
    assert air.pressure == pytest.approx(pressure, rel=1e-4)
    assert air.density == pytest.approx(density, rel=1e-4)
    assert air.speed_of_sound == pytest.approx(speed_of_sound, rel=1e-4)


@pytest.mark.parametrize("altitude", [0.0, 1_500.0, 9_000.0, 15_000.0, 20_000.0])
def test_obeys_ideal_gas_law(altitude):
    air = isa(altitude)
    assert air.pressure == pytest.approx(air.density * GAS_CONSTANT_AIR * air.temperature)


def test_is_continuous_at_tropopause():
    below = isa(TROPOPAUSE_ALTITUDE - 1e-6)
    above = isa(TROPOPAUSE_ALTITUDE + 1e-6)
    assert above.temperature == pytest.approx(below.temperature)
    assert above.pressure == pytest.approx(below.pressure)
    assert above.density == pytest.approx(below.density)


def test_density_decreases_with_altitude():
    densities = [isa(h).density for h in range(0, 20_001, 500)]
    assert all(upper < lower for lower, upper in zip(densities, densities[1:]))


@pytest.mark.parametrize("altitude", [-1.0, 20_000.1])
def test_rejects_altitudes_outside_model(altitude):
    with pytest.raises(ValueError):
        isa(altitude)


def test_dynamic_pressure_of_cruising_light_aircraft():
    assert dynamic_pressure(density=1.225, airspeed=55.0) == pytest.approx(1852.81, rel=1e-5)
