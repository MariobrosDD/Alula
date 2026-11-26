import math
from pathlib import Path

from alula.aircraft import load_aircraft
from alula.atmosphere import STANDARD_GRAVITY
from alula.linear_analysis import aircraft_modes, linearize
from alula.trim import trim

AIRCRAFT_FILE = Path(__file__).resolve().parents[1] / "aircraft" / "light_aircraft.toml"

ALTITUDE = 1000.0
AIRSPEED = 57.7


def describe(name: str, mode) -> str:
    stability = "stable" if mode.is_stable else "UNSTABLE"
    if mode.is_oscillatory:
        timing = f"period {mode.period:7.2f} s   damping ratio {mode.damping_ratio:5.3f}"
    else:
        timing = f"time constant {mode.time_constant:7.2f} s   (no oscillation)"
    return f"{name:<18} {timing:<48} {stability}"


def main() -> None:
    aircraft = load_aircraft(AIRCRAFT_FILE)
    trimmed = trim(aircraft, ALTITUDE, AIRSPEED)
    modes = aircraft_modes(linearize(aircraft, trimmed))

    print(f"Dynamic modes of the {aircraft.name.lower()} at {ALTITUDE:.0f} m and {AIRSPEED:.1f} m/s\n")
    print("Longitudinal")
    print("  " + describe("Short period", modes.short_period))
    print("  " + describe("Phugoid", modes.phugoid))
    print("Lateral-directional")
    print("  " + describe("Roll subsidence", modes.roll_subsidence))
    print("  " + describe("Dutch roll", modes.dutch_roll))
    print("  " + describe("Spiral", modes.spiral))

    lanchester = math.sqrt(2.0) * math.pi * AIRSPEED / STANDARD_GRAVITY
    print(f"\nLanchester's phugoid estimate: {lanchester:.1f} s")


if __name__ == "__main__":
    main()
