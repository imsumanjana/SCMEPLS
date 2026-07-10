from __future__ import annotations

import os
import sys
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from .config import APP_NAME
from .gui.main_window import MainWindow


def main() -> int:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    return app.exec()
