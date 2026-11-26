import argparse
import math
from pathlib import Path

from alula.aircraft import load_aircraft
from alula.autopilot import Autopilot, AutopilotTargets
from alula.simulation import simulate
from alula.trim import trim

AIRCRAFT_FILE = Path(__file__).resolve().parents[1] / "aircraft" / "light_aircraft.toml"

START_ALTITUDE = 1000.0
START_AIRSPEED = 57.7
TARGETS = AutopilotTargets(altitude=1200.0, heading=math.radians(120.0), airspeed=50.0)

DURATION = 240.0
TIME_STEP = 0.02
PRINT_EVERY = 10.0

METRES_PER_SECOND_TO_KNOTS = 1.0 / 0.514444


def compass_heading(yaw: float) -> float:
    return math.degrees(yaw) % 360.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Autopilot climb, turn and speed change.")
    parser.add_argument("--plot", action="store_true", help="show time histories and ground track")
    parser.add_argument("--3d", dest="view_3d", action="store_true", help="show the flight in 3D and animate it")
    parser.add_argument("--gif", metavar="PATH", help="save the 3D animation as an animated GIF")
    arguments = parser.parse_args()

    aircraft = load_aircraft(AIRCRAFT_FILE)
    trimmed = trim(aircraft, START_ALTITUDE, START_AIRSPEED)
    autopilot = Autopilot(TARGETS, trimmed, TIME_STEP)

    print(f"{aircraft.name} trimmed at {START_ALTITUDE:.0f} m, heading 000, {START_AIRSPEED:.1f} m/s")
    print(
        f"Autopilot engaged: climb to {TARGETS.altitude:.0f} m, "
        f"turn to heading {compass_heading(TARGETS.heading):03.0f}, "
        f"slow down to {TARGETS.airspeed:.0f} m/s\n"
    )

    result = simulate(aircraft, trimmed.state, autopilot, DURATION, TIME_STEP)
    history = result.history()

    print(
        f"{'time [s]':>9} {'altitude [m]':>13} {'airspeed [kt]':>14} "
        f"{'heading':>8} {'bank [deg]':>11} {'throttle':>9}"
    )
    stride = round(PRINT_EVERY / TIME_STEP)
    for index in range(0, len(history.time), stride):
        throttle = result.controls[min(index, len(result.controls) - 1)].throttle
        print(
            f"{history.time[index]:9.0f}"
            f"{history.altitude[index]:14.1f}"
            f"{history.airspeed[index] * METRES_PER_SECOND_TO_KNOTS:15.1f}"
            f"{compass_heading(history.yaw[index]):9.0f}"
            f"{math.degrees(history.roll[index]):12.1f}"
            f"{throttle:10.0%}"
        )

    print(f"\nSimulation ended at t = {result.final_time:.1f} s: {result.termination.value}")

    if not (arguments.plot or arguments.view_3d or arguments.gif):
        return

    import matplotlib.pyplot as plt

    from alula.plotting import plot_flight, plot_ground_track
    from alula.view3d import animate_flight, plot_flight_3d

    if arguments.plot:
        plot_flight(result)
        plot_ground_track(history)
    animation = None
    if arguments.view_3d or arguments.gif:
        animation = animate_flight(result, seconds_per_frame=2.0)
    if arguments.gif:
        animation.save_gif(arguments.gif)
        print(f"Animation saved to {arguments.gif}")
    if arguments.view_3d:
        plot_flight_3d(result)
    if arguments.plot or arguments.view_3d:
        plt.show()


if __name__ == "__main__":
    main()
