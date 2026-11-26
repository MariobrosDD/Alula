from pathlib import Path

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QFileDialog, QMessageBox

from alula.gui.aircraft_summary import aircraft_summary
from alula.gui.aircraft_tab import NO_AIRCRAFT_TEXT
from alula.gui.main_window import MainWindow


@pytest.fixture
def window(qtbot, light_aircraft_file):
    main_window = MainWindow(aircraft_file=light_aircraft_file)
    qtbot.addWidget(main_window)
    return main_window


@pytest.fixture
def empty_window(qtbot):
    main_window = MainWindow(aircraft_file=None)
    qtbot.addWidget(main_window)
    return main_window


@pytest.fixture
def broken_aircraft_file(tmp_path, light_aircraft_file):
    text = light_aircraft_file.read_text(encoding="utf-8").replace("mass_kg = 1000.0", "mass_kg = -5.0")
    path = tmp_path / "broken.toml"
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def captured_errors(monkeypatch):
    messages = []
    monkeypatch.setattr(QMessageBox, "critical", lambda parent, title, text: messages.append((title, text)))
    return messages


def cell(window, section, row, column):
    return window.aircraft_tab.tables[section].item(row, column).text()


def test_starts_with_the_default_aircraft_loaded(window, light_aircraft):
    assert window.aircraft == light_aircraft
    assert window.windowTitle() == "Alula - Light aircraft"
    assert window.aircraft_tab.title.text() == "Light aircraft"


def test_starts_empty_without_an_aircraft_file(empty_window):
    assert empty_window.aircraft is None
    assert empty_window.windowTitle() == "Alula"
    assert empty_window.aircraft_tab.title.text() == NO_AIRCRAFT_TEXT
    assert empty_window.aircraft_tab.tables == {}


def test_aircraft_tab_shows_one_table_per_summary_section(window, light_aircraft):
    sections = aircraft_summary(light_aircraft)
    assert list(window.aircraft_tab.tables) == [section.title for section in sections]
    for section in sections:
        assert window.aircraft_tab.tables[section.title].rowCount() == len(section.rows)


def test_aircraft_tab_shows_values_with_units(window):
    assert (cell(window, "Geometry", 0, 0), cell(window, "Geometry", 0, 1), cell(window, "Geometry", 0, 2)) == (
        "Wing area",
        "16.2",
        "m²",
    )


def test_numbers_are_right_aligned_and_text_sections_hide_the_unit_column(window):
    numbers = window.aircraft_tab.tables["Geometry"]
    assert numbers.item(0, 1).textAlignment() & Qt.AlignmentFlag.AlignRight
    assert not numbers.isColumnHidden(2)
    text = window.aircraft_tab.tables["Control sign conventions"]
    assert not text.item(0, 1).textAlignment() & Qt.AlignmentFlag.AlignRight
    assert text.isColumnHidden(2)


def test_tables_are_read_only(window):
    table = window.aircraft_tab.tables["Geometry"]
    assert table.editTriggers() == table.EditTrigger.NoEditTriggers


def test_loading_emits_the_aircraft_loaded_signal(qtbot, empty_window, light_aircraft_file):
    with qtbot.waitSignal(empty_window.aircraft_loaded, timeout=1000) as blocker:
        assert empty_window.load_aircraft(light_aircraft_file)
    assert blocker.args[0].name == "Light aircraft"


def test_status_bar_shows_the_loaded_file(window, light_aircraft_file):
    assert str(light_aircraft_file) in window.statusBar().currentMessage()


def test_invalid_file_shows_an_error_and_keeps_the_current_aircraft(
    window, broken_aircraft_file, captured_errors, light_aircraft
):
    assert not window.load_aircraft(broken_aircraft_file)
    assert window.aircraft == light_aircraft
    assert len(captured_errors) == 1
    title, text = captured_errors[0]
    assert title == "Cannot open aircraft"
    assert "mass_kg" in text


def test_missing_file_shows_an_error(empty_window, tmp_path, captured_errors):
    assert not empty_window.load_aircraft(tmp_path / "does_not_exist.toml")
    assert len(captured_errors) == 1


def test_open_dialog_loads_the_chosen_file(empty_window, light_aircraft_file, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: (str(light_aircraft_file), ""))
    empty_window.open_action.trigger()
    assert empty_window.aircraft is not None
    assert empty_window.aircraft_file == Path(light_aircraft_file)


def test_cancelling_the_open_dialog_changes_nothing(window, light_aircraft, monkeypatch):
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: ("", ""))
    window.open_action.trigger()
    assert window.aircraft == light_aircraft


def test_menu_actions_use_standard_shortcuts(window):
    assert window.open_action.shortcut() == QKeySequence(QKeySequence.StandardKey.Open)
    assert window.quit_action.shortcut() == QKeySequence(QKeySequence.StandardKey.Quit)


def test_quit_action_closes_the_window(qtbot, window):
    window.show()
    qtbot.waitExposed(window)
    window.quit_action.trigger()
    assert not window.isVisible()


def test_about_dialog_names_the_application(window, monkeypatch):
    shown = []
    monkeypatch.setattr(QMessageBox, "about", lambda parent, title, text: shown.append((title, text)))
    window.about_action.trigger()
    assert shown and shown[0][0] == "About Alula"
    assert "six-degree-of-freedom" in shown[0][1]
