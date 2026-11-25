import math
from pathlib import Path

from alula.aircraft import load_aircraft
from alula.atmosphere import SEA_LEVEL_DENSITY, isa
from alula.controls import Controls
from alula.propulsion import altitude_power_fraction
from alula.simulation import initial_state, simulate

AIRCRAFT_FILE = Path(__file__).resolve().parents[1] / "aircraft" / "light_aircraft.toml"

ALTITUDE = 1000.0
SEA_LEVEL_TRIM_AIRSPEED = 55.0
TRIM_ALPHA = 0.019909
TRIM_ELEVATOR = -0.05057
TRIM_DRAG = 981.4

PULSE_START = 5.0
PULSE_DURATION = 1.0
PULSE_ELEVATOR = math.radians(-2.0)

DURATION = 120.0
TIME_STEP = 0.01
PRINT_EVERY = 4.0

METRES_PER_SECOND_TO_KNOTS = 1.0 / 0.514444


def main() -> None:
    aircraft = load_aircraft(AIRCRAFT_FILE)
    density = isa(ALTITUDE).density
    airspeed = SEA_LEVEL_TRIM_AIRSPEED * math.sqrt(SEA_LEVEL_DENSITY / density)
    engine = aircraft.propulsion
    throttle = TRIM_DRAG * airspeed / (
        engine.propeller_efficiency * engine.max_shaft_power * altitude_power_fraction(density)
    )

    def pilot(time: float, state) -> Controls:
        pulling = PULSE_START <= time < PULSE_START + PULSE_DURATION
        elevator = TRIM_ELEVATOR + (PULSE_ELEVATOR if pulling else 0.0)
        return Controls(elevator=elevator, throttle=throttle)

    print(f"{aircraft.name} trimmed at {ALTITUDE:.0f} m, {airspeed:.1f} m/s, throttle {throttle:.0%}")
    print(f"Stick pulled back {abs(math.degrees(PULSE_ELEVATOR)):.0f} deg at t = {PULSE_START:.0f} s "
          f"for {PULSE_DURATION:.0f} s\n")

    result = simulate(aircraft, initial_state(ALTITUDE, airspeed, TRIM_ALPHA), pilot, DURATION, TIME_STEP)
    history = result.history()

    print(f"{'time [s]':>9} {'altitude [m]':>13} {'airspeed [kt]':>14} {'alpha [deg]':>12} {'pitch [deg]':>12}")
    stride = round(PRINT_EVERY / TIME_STEP)
    for index in range(0, len(history.time), stride):
        print(
            f"{history.time[index]:9.0f}"
            f"{history.altitude[index]:14.1f}"
            f"{history.airspeed[index] * METRES_PER_SECOND_TO_KNOTS:15.1f}"
            f"{math.degrees(history.alpha[index]):13.2f}"
            f"{math.degrees(history.pitch[index]):13.2f}"
        )

    print(f"\nSimulation ended at t = {result.final_time:.1f} s: {result.termination.value}")
    if result.termination_message:
        print(result.termination_message)


if __name__ == "__main__":
    main()
