from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QTabWidget, QWidget

from alula import __version__
from alula.aircraft import Aircraft, load_aircraft
from alula.gui.aircraft_tab import AircraftTab

APPLICATION_NAME = "Alula"
DEFAULT_AIRCRAFT_FILE = Path(__file__).resolve().parents[2] / "aircraft" / "light_aircraft.toml"
AIRCRAFT_FILE_FILTER = "Aircraft files (*.toml)"
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

        self.setWindowTitle(APPLICATION_NAME)
        self.resize(1200, 800)

        self.aircraft_tab = AircraftTab()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.aircraft_tab, "Aircraft")
        self.setCentralWidget(self.tabs)

        self._create_menus()
        self.statusBar().showMessage("Open an aircraft file to begin")

        if aircraft_file is not None and aircraft_file.exists():
            self.load_aircraft(aircraft_file)

    def load_aircraft(self, path: Path) -> bool:
        try:
            aircraft = load_aircraft(path)
        except (OSError, ValueError) as error:
            QMessageBox.critical(self, "Cannot open aircraft", f"{path.name}\n\n{error}")
            return False

        self.aircraft = aircraft
        self.aircraft_file = path
        self.aircraft_tab.show_aircraft(aircraft, path)
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
