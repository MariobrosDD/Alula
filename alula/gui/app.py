import sys

from PySide6.QtWidgets import QApplication

from alula.gui.main_window import APPLICATION_NAME, MainWindow


def main() -> int:
    application = QApplication.instance() or QApplication(sys.argv)
    application.setApplicationName(APPLICATION_NAME)
    window = MainWindow()
    window.show()
    return application.exec()
