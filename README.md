# Alula

A six-degree-of-freedom (6-DOF) flight dynamics simulator, written in Python and built from first principles.

The goal of Alula is to model a rigid aircraft flying through the atmosphere: aerodynamic forces and moments,
propulsion, the Newton-Euler equations of motion, trim, linear stability analysis and automatic flight control.
Every module is backed by tests and validated against published reference data.

> The *alula* is a small group of feathers on the leading edge of a bird's wing. Birds deploy it at low speed to
> delay the stall, just like the leading-edge slats of an aircraft.

## Roadmap

- [ ] **Foundations**: International Standard Atmosphere, rotations (Euler angles, quaternions)
- [ ] **First flight**: linear aerodynamic model, propulsion, 6-DOF equations of motion, RK4 integrator
- [ ] **Equilibrium and analysis**: trim, linearisation, dynamic modes
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

## Project layout

```
alula/          Simulator source code
tests/          Unit tests, one file per module
```
