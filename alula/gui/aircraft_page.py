from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLayout,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from alula.aircraft import Aircraft
from alula.gui.aircraft_summary import aircraft_summary, key_figures
from alula.gui.widgets import KeyValueCard, PlanformView, Section, StatTile

NO_AIRCRAFT_TEXT = "No aircraft loaded"
EMPTY_HINT = "Open a TOML aircraft file to see its geometry, mass, engine and stability derivatives."
GRID_COLUMNS = 3
TRAILING_SECTION = "Control sign conventions"
SIDE_OF_PLANFORM_SECTION = "Geometry"
PLANFORM_CAPTION = "Wing to scale from the data file. Fuselage and tailplane are schematic."
COLUMN_GAP = 44
ROW_GAP = 40


class AircraftPage(QWidget):
    open_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Page")
        self.cards: dict[str, KeyValueCard] = {}
        self.stat_tiles: list[StatTile] = []
        self.planform: PlanformView | None = None

        self.title = QLabel(NO_AIRCRAFT_TEXT)
        self.title.setObjectName("PageTitle")
        self.source = QLabel(EMPTY_HINT)
        self.source.setObjectName("Path")
        self.source.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.open_button = QPushButton("OPEN AIRCRAFT")
        self.open_button.setObjectName("OutlineButton")
        self.open_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.open_button.clicked.connect(self.open_requested.emit)

        heading = QVBoxLayout()
        heading.setSpacing(6)
        heading.addWidget(self.title)
        heading.addWidget(self.source)
        header = QHBoxLayout()
        header.addLayout(heading, stretch=1)
        header.addWidget(self.open_button, alignment=Qt.AlignmentFlag.AlignBottom)

        self.figure_strip = QFrame()
        self.figure_strip.setObjectName("FigureStrip")
        self.stats_row = QHBoxLayout(self.figure_strip)
        self.stats_row.setContentsMargins(0, 0, 0, 0)
        self.stats_row.setSpacing(0)
        self.grid = QGridLayout()
        self.grid.setHorizontalSpacing(COLUMN_GAP)
        self.grid.setVerticalSpacing(ROW_GAP)
        for column in range(GRID_COLUMNS):
            self.grid.setColumnStretch(column, 1)

        content = QWidget()
        content.setObjectName("PageContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 24, 40)
        content_layout.setSpacing(ROW_GAP)
        content_layout.addWidget(self.figure_strip)
        content_layout.addLayout(self.grid)
        content_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(content)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(48, 40, 24, 0)
        layout.setSpacing(32)
        layout.addLayout(header)
        layout.addWidget(scroll, stretch=1)

    def show_aircraft(self, aircraft: Aircraft, source: Path | None = None) -> None:
        self.title.setText(aircraft.name)
        self.source.setText(str(source) if source else "")
        _clear(self.stats_row)
        _clear(self.grid)
        self.cards.clear()

        self.stat_tiles = [
            StatTile(figure, first=index == 0) for index, figure in enumerate(key_figures(aircraft))
        ]
        for tile in self.stat_tiles:
            self.stats_row.addWidget(tile, stretch=1)

        sections = {section.title: KeyValueCard(section) for section in aircraft_summary(aircraft)}
        self.cards.update(sections)

        self.planform = PlanformView()
        self.planform.set_aircraft_geometry(aircraft.geometry)
        planform_card = Section("Top view")
        planform_card.body.addWidget(self.planform, stretch=1)
        caption = QLabel(PLANFORM_CAPTION)
        caption.setObjectName("Muted")
        planform_card.body.addWidget(caption)
        self.grid.addWidget(planform_card, 0, 0, 1, GRID_COLUMNS - 1)
        self.grid.addWidget(sections[SIDE_OF_PLANFORM_SECTION], 0, GRID_COLUMNS - 1)

        flowing = [
            card for title, card in sections.items() if title not in (SIDE_OF_PLANFORM_SECTION, TRAILING_SECTION)
        ]
        for position, card in enumerate(flowing):
            row, column = divmod(position, GRID_COLUMNS)
            self.grid.addWidget(card, row + 1, column)
        row, column = divmod(len(flowing), GRID_COLUMNS)
        self.grid.addWidget(sections[TRAILING_SECTION], row + 1, column, 1, GRID_COLUMNS - column)


def _clear(layout: QLayout) -> None:
    while layout.count():
        widget = layout.takeAt(0).widget()
        if widget is not None:
            widget.deleteLater()
