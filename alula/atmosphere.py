import math
from dataclasses import dataclass

GAS_CONSTANT_AIR = 287.05287
HEAT_CAPACITY_RATIO = 1.4
STANDARD_GRAVITY = 9.80665

SEA_LEVEL_TEMPERATURE = 288.15
SEA_LEVEL_PRESSURE = 101_325.0
SEA_LEVEL_DENSITY = SEA_LEVEL_PRESSURE / (GAS_CONSTANT_AIR * SEA_LEVEL_TEMPERATURE)

TROPOSPHERE_LAPSE_RATE = 0.0065
TROPOPAUSE_ALTITUDE = 11_000.0
MAX_ALTITUDE = 20_000.0

TROPOPAUSE_TEMPERATURE = SEA_LEVEL_TEMPERATURE - TROPOSPHERE_LAPSE_RATE * TROPOPAUSE_ALTITUDE
TROPOSPHERE_PRESSURE_EXPONENT = STANDARD_GRAVITY / (TROPOSPHERE_LAPSE_RATE * GAS_CONSTANT_AIR)
TROPOPAUSE_PRESSURE = SEA_LEVEL_PRESSURE * (
    TROPOPAUSE_TEMPERATURE / SEA_LEVEL_TEMPERATURE
) ** TROPOSPHERE_PRESSURE_EXPONENT


@dataclass(frozen=True)
class AtmosphereState:
    temperature: float
    pressure: float
    density: float
    speed_of_sound: float


def isa(altitude: float) -> AtmosphereState:
    if not 0.0 <= altitude <= MAX_ALTITUDE:
        raise ValueError(
            f"ISA model is valid between 0 and {MAX_ALTITUDE:.0f} m, got {altitude} m"
        )

    if altitude <= TROPOPAUSE_ALTITUDE:
        temperature = SEA_LEVEL_TEMPERATURE - TROPOSPHERE_LAPSE_RATE * altitude
        pressure = SEA_LEVEL_PRESSURE * (
            temperature / SEA_LEVEL_TEMPERATURE
        ) ** TROPOSPHERE_PRESSURE_EXPONENT
    else:
        temperature = TROPOPAUSE_TEMPERATURE
        pressure = TROPOPAUSE_PRESSURE * math.exp(
            -STANDARD_GRAVITY
            * (altitude - TROPOPAUSE_ALTITUDE)
            / (GAS_CONSTANT_AIR * temperature)
        )

    density = pressure / (GAS_CONSTANT_AIR * temperature)
    speed_of_sound = math.sqrt(HEAT_CAPACITY_RATIO * GAS_CONSTANT_AIR * temperature)
    return AtmosphereState(temperature, pressure, density, speed_of_sound)


def dynamic_pressure(density: float, airspeed: float) -> float:
    return 0.5 * density * airspeed**2
