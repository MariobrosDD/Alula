from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from alula.aircraft import Geometry
from alula.gui import theme
from alula.gui.aircraft_summary import KeyFigure, SummarySection

FUSELAGE_LENGTH_PER_SPAN = 0.72
FUSELAGE_WIDTH_PER_SPAN = 0.085
WING_LEADING_EDGE_PER_LENGTH = 0.28
TAIL_SPAN_PER_SPAN = 0.32
TAIL_CHORD_PER_CHORD = 0.62
DRAWING_MARGIN = 36.0
DIMENSION_BAND = 52.0
CHORD_DIMENSION_BAND = 90.0
UNIT_COLUMN_WIDTH = 56
TEXT_LABEL_WIDTH = 96


class Section(QFrame):
    def __init__(self, title: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Section")
        self.body = QVBoxLayout(self)
        self.body.setContentsMargins(0, 12, 0, 0)
        self.body.setSpacing(0)
        heading = QLabel(title.upper())
        heading.setObjectName("SectionTitle")
        heading.setContentsMargins(0, 0, 0, 10)
        self.body.addWidget(heading)


class StatTile(QFrame):
    def __init__(self, figure: KeyFigure, first: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Figure")
        self.setProperty("first", first)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0 if first else 28, 18, 16, 20)
        layout.setSpacing(4)
        self.label = QLabel(figure.label.upper())
        self.label.setObjectName("FigureLabel")
        self.value = QLabel(figure.value)
        self.value.setObjectName("FigureValue")
        self.unit = QLabel(figure.unit)
        self.unit.setObjectName("FigureUnit")
        value_row = QHBoxLayout()
        value_row.setSpacing(8)
        value_row.addWidget(self.value, alignment=Qt.AlignmentFlag.AlignBaseline)
        value_row.addWidget(self.unit, alignment=Qt.AlignmentFlag.AlignBaseline)
        value_row.addStretch(1)
        layout.addWidget(self.label)
        layout.addLayout(value_row)


class KeyValueCard(Section):
    def __init__(self, section: SummarySection, parent: QWidget | None = None) -> None:
        super().__init__(section.title, parent)
        self.section = section
        is_text = all(not row.unit for row in section.rows)
        for row in section.rows:
            self.body.addWidget(_text_row(row.quantity, row.value) if is_text else _number_row(row.quantity, row.value, row.unit))
        self.body.addStretch(1)

    def rows(self) -> list[tuple[str, str, str]]:
        return [(row.quantity, row.value, row.unit) for row in self.section.rows]


def _spec_row() -> tuple[QFrame, QHBoxLayout]:
    frame = QFrame()
    frame.setObjectName("SpecRow")
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(0, 8, 0, 8)
    layout.setSpacing(12)
    return frame, layout


def _number_row(quantity: str, value: str, unit: str) -> QFrame:
    frame, layout = _spec_row()
    label = QLabel(quantity)
    label.setObjectName("RowLabel")
    number = QLabel(value)
    number.setObjectName("RowValue")
    number.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    unit_label = QLabel(unit)
    unit_label.setObjectName("RowUnit")
    unit_label.setFixedWidth(UNIT_COLUMN_WIDTH)
    layout.addWidget(label, stretch=1)
    layout.addWidget(number)
    layout.addWidget(unit_label)
    return frame


def _text_row(quantity: str, text: str) -> QFrame:
    frame, layout = _spec_row()
    label = QLabel(quantity)
    label.setObjectName("RowLabel")
    label.setFixedWidth(TEXT_LABEL_WIDTH)
    label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
    description = QLabel(text)
    description.setObjectName("RowText")
    description.setWordWrap(True)
    layout.addWidget(label, alignment=Qt.AlignmentFlag.AlignTop)
    layout.addWidget(description, stretch=1)
    return frame


class PlanformView(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.geometry_data: Geometry | None = None
        self.setMinimumHeight(300)

    def set_aircraft_geometry(self, geometry: Geometry) -> None:
        self.geometry_data = geometry
        self.update()

    def paintEvent(self, event) -> None:
        if self.geometry_data is None:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        span = self.geometry_data.wingspan
        chord = self.geometry_data.mean_chord
        length = FUSELAGE_LENGTH_PER_SPAN * span

        available_width = self.width() - 2.0 * DRAWING_MARGIN - CHORD_DIMENSION_BAND
        available_height = self.height() - 2.0 * DRAWING_MARGIN - DIMENSION_BAND
        scale = min(available_width / span, available_height / length)
        centre_x = (self.width() - CHORD_DIMENSION_BAND) / 2.0
        nose_y = DRAWING_MARGIN + (available_height - length * scale) / 2.0

        def point(across: float, along: float) -> QPointF:
            return QPointF(centre_x + across * scale, nose_y + along * scale)

        def rectangle(half_width: float, start: float, depth: float) -> QRectF:
            return QRectF(point(-half_width, start), point(half_width, start + depth))

        centreline = QPen(QColor(theme.MUTED_TEXT), 1.0, Qt.PenStyle.DashDotLine)
        painter.setPen(centreline)
        painter.drawLine(point(0.0, -0.06 * length), point(0.0, 1.06 * length))

        outline = QPen(QColor(theme.PRIMARY_TEXT), 1.3)
        outline.setJoinStyle(Qt.PenJoinStyle.MiterJoin)
        painter.setPen(outline)
        painter.setBrush(QColor(theme.BACKGROUND))
        wing_start = WING_LEADING_EDGE_PER_LENGTH * length
        tail_chord = TAIL_CHORD_PER_CHORD * chord
        painter.drawRect(rectangle(span / 2.0, wing_start, chord))
        painter.drawRect(rectangle(TAIL_SPAN_PER_SPAN * span / 2.0, length - tail_chord, tail_chord))

        half_body = FUSELAGE_WIDTH_PER_SPAN * span / 2.0
        fuselage = QPolygonF(
            [
                point(0.0, 0.0),
                point(half_body, 0.10 * length),
                point(half_body, 0.62 * length),
                point(0.28 * half_body, length),
                point(-0.28 * half_body, length),
                point(-half_body, 0.62 * length),
                point(-half_body, 0.10 * length),
            ]
        )
        path = QPainterPath()
        path.addPolygon(fuselage)
        path.closeSubpath()
        painter.drawPath(path)

        font = QFont(theme.MONO_FAMILY)
        font.setPointSizeF(8.0)
        painter.setFont(font)
        dimension = QPen(QColor(theme.SECONDARY_TEXT), 1.0)
        span_y = nose_y + length * scale + 26.0
        self._horizontal_dimension(painter, dimension, point(-span / 2.0, 0).x(), point(span / 2.0, 0).x(), span_y, f"b = {span:g} m")
        chord_x = point(span / 2.0, 0).x() + 34.0
        self._vertical_dimension(
            painter, dimension, chord_x, point(0, wing_start).y(), point(0, wing_start + chord).y(), f"c = {chord:g} m"
        )
        painter.end()

    def _horizontal_dimension(
        self, painter: QPainter, pen: QPen, left: float, right: float, y: float, text: str
    ) -> None:
        painter.setPen(pen)
        painter.drawLine(QPointF(left, y), QPointF(right, y))
        for x in (left, right):
            painter.drawLine(QPointF(x, y - 6), QPointF(x, y + 6))
        painter.drawText(
            QRectF(left, y + 8, right - left, 18), Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop, text
        )

    def _vertical_dimension(
        self, painter: QPainter, pen: QPen, x: float, top: float, bottom: float, text: str
    ) -> None:
        painter.setPen(pen)
        painter.drawLine(QPointF(x, top), QPointF(x, bottom))
        for y in (top, bottom):
            painter.drawLine(QPointF(x - 6, y), QPointF(x + 6, y))
        painter.drawText(
            QRectF(x + 10, top, CHORD_DIMENSION_BAND, bottom - top),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            text,
        )
