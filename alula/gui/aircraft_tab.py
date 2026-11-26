from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from alula.aircraft import Aircraft
from alula.gui.aircraft_summary import SummarySection, aircraft_summary

NO_AIRCRAFT_TEXT = "No aircraft loaded"
SECTION_COLUMNS = 2
TABLE_HEADERS = ("Quantity", "Value", "Unit")


class AircraftTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.tables: dict[str, QTableWidget] = {}

        self.title = QLabel(NO_AIRCRAFT_TEXT)
        title_font = QFont(self.title.font())
        title_font.setPointSize(title_font.pointSize() + 6)
        title_font.setBold(True)
        self.title.setFont(title_font)

        self.source = QLabel("Use File > Open aircraft... to load a TOML aircraft file.")
        self.source.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)

        self.sections = QWidget()
        self.sections_layout = QGridLayout(self.sections)
        self.sections_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.sections)

        layout = QVBoxLayout(self)
        layout.addWidget(self.title)
        layout.addWidget(self.source)
        layout.addWidget(scroll, stretch=1)

    def show_aircraft(self, aircraft: Aircraft, source: Path | None = None) -> None:
        self.title.setText(aircraft.name)
        self.source.setText(str(source) if source else "")
        self._clear_sections()
        for position, section in enumerate(aircraft_summary(aircraft)):
            row, column = divmod(position, SECTION_COLUMNS)
            self.sections_layout.addWidget(self._section_box(section), row, column)

    def _clear_sections(self) -> None:
        self.tables.clear()
        while self.sections_layout.count():
            widget = self.sections_layout.takeAt(0).widget()
            if widget is not None:
                widget.deleteLater()

    def _section_box(self, section: SummarySection) -> QGroupBox:
        is_text_section = all(not entry.unit for entry in section.rows)
        table = QTableWidget(len(section.rows), len(TABLE_HEADERS))
        table.setHorizontalHeaderLabels(TABLE_HEADERS)
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        table.setWordWrap(True)
        for row, entry in enumerate(section.rows):
            for column, text in enumerate((entry.quantity, entry.value, entry.unit)):
                item = QTableWidgetItem(text)
                if column == 1 and not is_text_section:
                    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                table.setItem(row, column, item)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        if is_text_section:
            table.setColumnHidden(2, True)
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        else:
            header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
            header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        table.resizeRowsToContents()
        table.setFixedHeight(
            header.height() + sum(table.rowHeight(row) for row in range(table.rowCount())) + 4
        )
        self.tables[section.title] = table

        box = QGroupBox(section.title)
        box_layout = QVBoxLayout(box)
        box_layout.addWidget(table)
        box_layout.addStretch(1)
        return box
