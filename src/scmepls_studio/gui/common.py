from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure

from ..config import DEFAULT_EXPORT_DPI
from ..io.export import export_figure


class FigureCanvasPanel(QWidget):
    figure_changed = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.figure = Figure(figsize=(7, 5), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)

    def set_figure(self, figure: Figure) -> None:
        old_canvas = self.canvas
        old_toolbar = self.toolbar
        self.layout().removeWidget(old_toolbar)
        self.layout().removeWidget(old_canvas)
        old_toolbar.deleteLater()
        old_canvas.deleteLater()
        self.figure = figure
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.layout().insertWidget(0, self.toolbar)
        self.layout().addWidget(self.canvas, 1)
        self.canvas.draw_idle()
        self.figure_changed.emit()


class ScrollableControls(QScrollArea):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget()
        self.content_layout = QVBoxLayout(content)
        self.content_layout.setContentsMargins(8, 8, 8, 8)
        self.content_layout.setSpacing(8)
        self.content_layout.addStretch(1)
        self.setWidget(content)

    def add_widget(self, widget: QWidget) -> None:
        self.content_layout.insertWidget(self.content_layout.count() - 1, widget)


def section_label(text: str) -> QLabel:
    label = QLabel(text)
    font = QFont()
    font.setBold(True)
    font.setPointSize(11)
    label.setFont(font)
    label.setWordWrap(True)
    return label


def form_row(label_text: str, widget: QWidget) -> QWidget:
    frame = QWidget()
    layout = QHBoxLayout(frame)
    layout.setContentsMargins(0, 0, 0, 0)
    label = QLabel(label_text)
    label.setWordWrap(True)
    label.setMinimumWidth(145)
    layout.addWidget(label, 1)
    layout.addWidget(widget, 1)
    return frame


def make_double(value: float, minimum: float = -1e9, maximum: float = 1e9, decimals: int = 4, step: float = 0.1) -> QDoubleSpinBox:
    box = QDoubleSpinBox()
    box.setRange(minimum, maximum)
    box.setDecimals(decimals)
    box.setSingleStep(step)
    box.setValue(value)
    box.setKeyboardTracking(False)
    box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    return box


def make_int(value: int, minimum: int = 0, maximum: int = 100000, step: int = 1) -> QSpinBox:
    box = QSpinBox()
    box.setRange(minimum, maximum)
    box.setSingleStep(step)
    box.setValue(value)
    box.setKeyboardTracking(False)
    return box


def provenance_combo(default: str = "Illustrative") -> QComboBox:
    combo = QComboBox()
    combo.addItems(["Illustrative", "Literature-derived", "Simulation", "Experimental", "AI-calibrated"])
    combo.setCurrentText(default)
    return combo


class ExportBar(QFrame):
    export_requested = pyqtSignal(int, str)

    def __init__(self, default_name: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.default_name = default_name
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(QLabel("Export DPI:"))
        self.dpi = make_int(DEFAULT_EXPORT_DPI, 600, 2400, 100)
        self.dpi.setMaximumWidth(110)
        layout.addWidget(self.dpi)
        self.export_button = QPushButton("Save Article PNG")
        layout.addWidget(self.export_button)
        layout.addStretch(1)
        self.export_button.clicked.connect(self._emit_export)

    def _emit_export(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save article figure", self.default_name, "PNG image (*.png)")
        if path:
            self.export_requested.emit(self.dpi.value(), path)


def save_panel_figure(parent: QWidget, panel: FigureCanvasPanel, dpi: int, path: str, metadata: dict) -> tuple[Path, Path]:
    return export_figure(panel.figure, path, dpi=dpi, metadata=metadata)
