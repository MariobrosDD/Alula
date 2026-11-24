from pathlib import Path

import pytest

from alula.aircraft import Aircraft, load_aircraft

LIGHT_AIRCRAFT_FILE = Path(__file__).resolve().parents[1] / "aircraft" / "light_aircraft.toml"


@pytest.fixture
def light_aircraft_file() -> Path:
    return LIGHT_AIRCRAFT_FILE


@pytest.fixture
def light_aircraft() -> Aircraft:
    return load_aircraft(LIGHT_AIRCRAFT_FILE)
