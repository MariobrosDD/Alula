import argparse
import math
from pathlib import Path

from alula.aircraft import load_aircraft
from alula.controls import Controls
from alula.simulation import simulate
from alula.trim import trim

AIRCRAFT_FILE = Path(__file__).resolve().parents[1] / "aircraft" / "light_aircraft.toml"

ALTITUDE = 1000.0
AIRSPEED = 57.7

PULSE_START = 5.0
PULSE_DURATION = 1.0
PULSE_ELEVATOR = math.radians(-2.0)

DURATION = 120.0
TIME_STEP = 0.01
PRINT_EVERY = 4.0

METRES_PER_SECOND_TO_KNOTS = 1.0 / 0.514444


def main() -> None:
    parser = argparse.ArgumentParser(description="Trimmed flight with a short elevator pulse.")
    parser.add_argument("--plot", action="store_true", help="show the time histories in a window")
    arguments = parser.parse_args()

    aircraft = load_aircraft(AIRCRAFT_FILE)
    trimmed = trim(aircraft, ALTITUDE, AIRSPEED)
    trim_controls = trimmed.controls

    def pilot(time: float, state) -> Controls:
        pulling = PULSE_START <= time < PULSE_START + PULSE_DURATION
        elevator = trim_controls.elevator + (PULSE_ELEVATOR if pulling else 0.0)
        return Controls(elevator=elevator, throttle=trim_controls.throttle)

    print(
        f"{aircraft.name} trimmed at {ALTITUDE:.0f} m and {AIRSPEED:.1f} m/s: "
        f"alpha {math.degrees(trimmed.alpha):.2f} deg, "
        f"elevator {math.degrees(trim_controls.elevator):.2f} deg, "
        f"throttle {trim_controls.throttle:.0%}"
    )
    print(
        f"Stick pulled back {abs(math.degrees(PULSE_ELEVATOR)):.0f} deg at t = {PULSE_START:.0f} s "
        f"for {PULSE_DURATION:.0f} s\n"
    )

    result = simulate(aircraft, trimmed.state, pilot, DURATION, TIME_STEP)
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

    if arguments.plot:
        import matplotlib.pyplot as plt

        from alula.plotting import plot_flight

        plot_flight(result)
        plt.show()


if __name__ == "__main__":
    main()
