import sys

from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication

from alula.gui.assets import APP_ICON_ICO, APP_ICON_PNG
from alula.gui.main_window import APPLICATION_NAME, MainWindow

WINDOWS_APP_ID = "alula.flight-simulator"


def main() -> int:
    if sys.platform == "win32":
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(WINDOWS_APP_ID)

    application = QApplication.instance() or QApplication(sys.argv)
    application.setApplicationName(APPLICATION_NAME)
    application.setStyle("Fusion")
    icon_file = APP_ICON_ICO if sys.platform == "win32" else APP_ICON_PNG
    application.setWindowIcon(QIcon(str(icon_file)))
    window = MainWindow()
    window.show()
    return application.exec()
