# Alula

A six-degree-of-freedom (6-DOF) flight dynamics simulator, written in Python and built from first principles.

The goal of Alula is to model a rigid aircraft flying through the atmosphere: aerodynamic forces and moments,
propulsion, the Newton-Euler equations of motion, trim, linear stability analysis and automatic flight control.
Every module is backed by tests and validated against published reference data.

> The *alula* is a small group of feathers on the leading edge of a bird's wing. Birds deploy it at low speed to
> delay the stall, just like the leading-edge slats of an aircraft.

## Roadmap

- [x] **Foundations**: International Standard Atmosphere, rotations (Euler angles, quaternions)
- [x] **First flight**: linear aerodynamic model, propulsion, 6-DOF equations of motion, RK4 integrator
- [x] **Equilibrium and analysis**: trim, linearisation, dynamic modes
- [ ] **Control**: PID controllers and an altitude/heading autopilot
- [ ] **Visualisation**: time-history plots and a 3D view of the flight

## Getting started

Requires Python 3.11 or newer.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -e ".[dev]"
pytest
```

## First flight

```bash
python examples/first_flight.py
```

Trims the light aircraft at 1000 m, pulls the stick back for one second and prints how the aircraft climbs,
descends and slowly settles again: the phugoid mode, with a period of about 32 seconds.

## Dynamic modes

```bash
python examples/aircraft_modes.py
```

Linearises the trimmed aircraft and computes its five classic dynamic modes (short period, phugoid,
roll subsidence, Dutch roll and spiral) from the eigenvalues of the state matrix. The phugoid period obtained
this way agrees with the one measured in the nonlinear simulation to within 2 %.

## Project layout

```
alula/          Simulator source code
aircraft/       Aircraft data files (geometry, mass, stability derivatives)
examples/       Runnable scripts that show the simulator in action
tests/          Unit tests, one file per module
```
