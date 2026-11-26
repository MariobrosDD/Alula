import re
from pathlib import Path

import pytest
from PySide6.QtGui import QKeySequence
from PySide6.QtWidgets import QFileDialog, QLabel, QMessageBox

from alula.gui.aircraft_page import NO_AIRCRAFT_TEXT
from alula.gui.aircraft_summary import aircraft_summary, key_figures
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


def test_starts_with_the_default_aircraft_loaded(window, light_aircraft):
    assert window.aircraft == light_aircraft
    assert window.windowTitle() == "Alula - Light aircraft"
    assert window.aircraft_page.title.text() == "Light aircraft"


def test_starts_empty_without_an_aircraft_file(empty_window):
    assert empty_window.aircraft is None
    assert empty_window.windowTitle() == "Alula"
    assert empty_window.aircraft_page.title.text() == NO_AIRCRAFT_TEXT
    assert empty_window.aircraft_page.cards == {}
    assert empty_window.aircraft_page.stat_tiles == []


def test_sidebar_starts_on_the_aircraft_page(window):
    assert list(window.navigation_buttons) == ["Aircraft"]
    assert window.current_page_name() == "Aircraft"
    assert window.pages.currentWidget() is window.aircraft_page


def test_stat_tiles_show_the_key_figures(window, light_aircraft):
    shown = [(tile.value.text(), tile.unit.text(), tile.label.text()) for tile in window.aircraft_page.stat_tiles]
    expected = [(figure.value, figure.unit, figure.label.upper()) for figure in key_figures(light_aircraft)]
    assert shown == expected


def test_only_the_first_key_figure_has_no_divider(window):
    assert [tile.property("first") for tile in window.aircraft_page.stat_tiles] == [True, False, False, False, False]


def test_window_and_sidebar_show_the_logo(window):
    assert not window.windowIcon().isNull()
    assert not window.logo.pixmap().isNull()


def test_sidebar_entries_are_numbered(window):
    assert window.navigation_buttons["Aircraft"].text().split() == ["01", "Aircraft"]


def test_style_sheet_has_no_rounded_corners_or_colour(window):
    style = window.styleSheet()
    radii = re.findall(r"border-radius:\s*(\d+)px", style)
    assert radii and set(radii) == {"0"}
    for colour in re.findall(r"#([0-9a-fA-F]{6})\b", style):
        red, green, blue = (int(colour[i : i + 2], 16) for i in (0, 2, 4))
        assert red == green == blue, f"#{colour} is not a shade of grey"


def test_one_card_per_summary_section_with_its_rows(window, light_aircraft):
    sections = aircraft_summary(light_aircraft)
    assert list(window.aircraft_page.cards) == [section.title for section in sections]
    for section in sections:
        expected = [(row.quantity, row.value, row.unit) for row in section.rows]
        assert window.aircraft_page.cards[section.title].rows() == expected


def test_number_cards_show_units_and_text_cards_do_not(window):
    geometry_units = window.aircraft_page.cards["Geometry"].findChildren(QLabel, "RowUnit")
    assert [label.text() for label in geometry_units] == ["m²", "m", "m", "-"]
    sign_card = window.aircraft_page.cards["Control sign conventions"]
    assert sign_card.findChildren(QLabel, "RowUnit") == []
    descriptions = sign_card.findChildren(QLabel, "RowText")
    assert len(descriptions) == 4
    assert all(label.wordWrap() for label in descriptions)


def test_top_view_draws_the_loaded_wing(window, light_aircraft):
    planform = window.aircraft_page.planform
    assert planform.geometry_data == light_aircraft.geometry
    planform.resize(600, 320)
    assert not planform.grab().isNull()


def test_open_button_opens_the_file_dialog(window, monkeypatch):
    opened = []
    monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *args: opened.append(True) or ("", ""))
    window.aircraft_page.open_button.click()
    assert opened == [True]


def test_loading_emits_the_aircraft_loaded_signal(qtbot, empty_window, light_aircraft_file):
    with qtbot.waitSignal(empty_window.aircraft_loaded, timeout=1000) as blocker:
        assert empty_window.load_aircraft(light_aircraft_file)
    assert blocker.args[0].name == "Light aircraft"


def test_reloading_replaces_cards_instead_of_duplicating_them(window, light_aircraft_file):
    window.load_aircraft(light_aircraft_file)
    assert len(window.aircraft_page.cards) == len(aircraft_summary(window.aircraft))
    assert len(window.aircraft_page.stat_tiles) == 5


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
