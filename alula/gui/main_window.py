from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from alula import __version__
from alula.aircraft import Aircraft, load_aircraft
from alula.gui.aircraft_page import AircraftPage
from alula.gui.assets import APP_ICON_PNG, LOGO_MARK_PNG
from alula.gui.theme import STYLE_SHEET

APPLICATION_NAME = "Alula"
TAGLINE = "6-DOF flight dynamics"
DEFAULT_AIRCRAFT_FILE = Path(__file__).resolve().parents[2] / "aircraft" / "light_aircraft.toml"
AIRCRAFT_FILE_FILTER = "Aircraft files (*.toml)"
SIDEBAR_WIDTH = 220
LOGO_MARK_SIZE = 26
ABOUT_TEXT = (
    f"<h3>{APPLICATION_NAME} {__version__}</h3>"
    "<p>A six-degree-of-freedom flight dynamics simulator, built from first principles.</p>"
    "<p>The <i>alula</i> is a small group of feathers on the leading edge of a bird's wing "
    "that delays the stall at low speed.</p>"
)


class MainWindow(QMainWindow):
    aircraft_loaded = Signal(object)

    def __init__(
        self, aircraft_file: Path | None = DEFAULT_AIRCRAFT_FILE, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.aircraft: Aircraft | None = None
        self.aircraft_file: Path | None = None
        self.navigation_buttons: dict[str, QPushButton] = {}

        self.setWindowTitle(APPLICATION_NAME)
        self.setWindowIcon(QIcon(str(APP_ICON_PNG)))
        self.setStyleSheet(STYLE_SHEET)
        self.resize(1280, 820)

        self.pages = QStackedWidget()
        self.navigation = QButtonGroup(self)
        self.navigation.setExclusive(True)
        self.sidebar = self._create_sidebar()

        self.aircraft_page = AircraftPage()
        self.aircraft_page.open_requested.connect(self.open_aircraft_dialog)
        self.add_page("Aircraft", self.aircraft_page)

        central = QWidget()
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.sidebar)
        layout.addWidget(self.pages, stretch=1)
        self.setCentralWidget(central)

        self._create_menus()
        self.statusBar().showMessage("Open an aircraft file to begin")
        self.show_page("Aircraft")

        if aircraft_file is not None and aircraft_file.exists():
            self.load_aircraft(aircraft_file)

    def add_page(self, name: str, page: QWidget) -> None:
        index = self.pages.addWidget(page)
        button = QPushButton(f"{index + 1:02d}    {name}")
        button.setObjectName("NavButton")
        button.setCheckable(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(lambda: self.pages.setCurrentIndex(index))
        self.navigation.addButton(button)
        self.navigation_layout.addWidget(button)
        self.navigation_buttons[name] = button

    def show_page(self, name: str) -> None:
        self.navigation_buttons[name].click()

    def current_page_name(self) -> str:
        return next(name for name, button in self.navigation_buttons.items() if button.isChecked())

    def load_aircraft(self, path: Path) -> bool:
        try:
            aircraft = load_aircraft(path)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot open aircraft", f"{path.name}\n\n{error}")
            return False

        self.aircraft = aircraft
        self.aircraft_file = path
        self.aircraft_page.show_aircraft(aircraft, path)
        self.setWindowTitle(f"{APPLICATION_NAME} - {aircraft.name}")
        self.statusBar().showMessage(f"Loaded {path}")
        self.aircraft_loaded.emit(aircraft)
        return True

    def open_aircraft_dialog(self) -> None:
        start_folder = self.aircraft_file.parent if self.aircraft_file else Path.cwd()
        path, _ = QFileDialog.getOpenFileName(
            self, "Open aircraft", str(start_folder), AIRCRAFT_FILE_FILTER
        )
        if path:
            self.load_aircraft(Path(path))

    def show_about(self) -> None:
        QMessageBox.about(self, f"About {APPLICATION_NAME}", ABOUT_TEXT)

    def _create_sidebar(self) -> QFrame:
        sidebar = QFrame()
        sidebar.setObjectName("Sidebar")
        sidebar.setFixedWidth(SIDEBAR_WIDTH)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 36, 14, 20)
        layout.setSpacing(2)
        self.logo = QLabel()
        self.logo.setObjectName("LogoMark")
        self.logo.setPixmap(
            QPixmap(str(LOGO_MARK_PNG)).scaled(
                LOGO_MARK_SIZE,
                LOGO_MARK_SIZE,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        name = QLabel(APPLICATION_NAME)
        name.setObjectName("AppName")
        brand = QHBoxLayout()
        brand.setContentsMargins(14, 0, 0, 0)
        brand.setSpacing(12)
        brand.addWidget(self.logo)
        brand.addWidget(name)
        brand.addStretch(1)
        tagline = QLabel(TAGLINE)
        tagline.setObjectName("AppTagline")
        layout.addLayout(brand)
        layout.addWidget(tagline)
        self.navigation_layout = QVBoxLayout()
        self.navigation_layout.setSpacing(4)
        layout.addLayout(self.navigation_layout)
        layout.addStretch(1)
        version = QLabel(f"v{__version__}")
        version.setObjectName("Version")
        layout.addWidget(version)
        return sidebar

    def _create_menus(self) -> None:
        self.open_action = QAction("&Open aircraft...", self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.open_action.triggered.connect(self.open_aircraft_dialog)

        self.quit_action = QAction("&Quit", self)
        self.quit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.quit_action.triggered.connect(self.close)

        self.about_action = QAction(f"&About {APPLICATION_NAME}", self)
        self.about_action.triggered.connect(self.show_about)

        file_menu = self.menuBar().addMenu("&File")
        file_menu.addAction(self.open_action)
        file_menu.addSeparator()
        file_menu.addAction(self.quit_action)

        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction(self.about_action)
