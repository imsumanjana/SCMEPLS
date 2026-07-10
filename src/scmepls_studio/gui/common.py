from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QPainter, QPixmap
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMessageBox,
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
from ..plotting.style import PlotStyle, apply_plot_style, available_font_families


class PlotStyleDialog(QDialog):
    """Toolbar dialog for publication typography settings."""

    def __init__(self, style: PlotStyle, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Plot typography")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)
        note = QLabel(
            "Apply font, size, and bold settings to the current Matplotlib figure. "
            "The same styling is used when saving high-resolution PNG files."
        )
        note.setWordWrap(True)
        layout.addWidget(note)

        form = QFormLayout()
        self.font_family = QComboBox()
        self.font_family.addItems(available_font_families())
        idx = self.font_family.findText(style.font_family)
        self.font_family.setCurrentIndex(max(0, idx))
        form.addRow("Font family", self.font_family)

        self.axis_label_size = make_int(style.axis_label_size, 6, 48, 1)
        self.tick_label_size = make_int(style.tick_label_size, 6, 48, 1)
        self.legend_size = make_int(style.legend_size, 6, 48, 1)
        self.annotation_size = make_int(style.annotation_size, 6, 60, 1)
        self.title_size = make_int(style.title_size, 6, 60, 1)
        form.addRow("Axis label size", self.axis_label_size)
        form.addRow("Tick label size", self.tick_label_size)
        form.addRow("Legend size", self.legend_size)
        form.addRow("Annotation size", self.annotation_size)
        form.addRow("Title size", self.title_size)

        self.bold_axis_labels = QCheckBox("Bold axis labels")
        self.bold_tick_labels = QCheckBox("Bold tick labels")
        self.bold_legends = QCheckBox("Bold legend text")
        self.bold_annotations = QCheckBox("Bold annotations / value labels")
        self.bold_titles = QCheckBox("Bold titles")
        self.apply_all = QCheckBox("Apply to all open figures")
        self.bold_axis_labels.setChecked(style.bold_axis_labels)
        self.bold_tick_labels.setChecked(style.bold_tick_labels)
        self.bold_legends.setChecked(style.bold_legends)
        self.bold_annotations.setChecked(style.bold_annotations)
        self.bold_titles.setChecked(style.bold_titles)
        form.addRow("", self.bold_axis_labels)
        form.addRow("", self.bold_tick_labels)
        form.addRow("", self.bold_legends)
        form.addRow("", self.bold_annotations)
        form.addRow("", self.bold_titles)
        form.addRow("", self.apply_all)
        layout.addLayout(form)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def style(self) -> PlotStyle:
        return PlotStyle(
            font_family=self.font_family.currentText(),
            axis_label_size=self.axis_label_size.value(),
            tick_label_size=self.tick_label_size.value(),
            legend_size=self.legend_size.value(),
            annotation_size=self.annotation_size.value(),
            title_size=self.title_size.value(),
            bold_axis_labels=self.bold_axis_labels.isChecked(),
            bold_tick_labels=self.bold_tick_labels.isChecked(),
            bold_legends=self.bold_legends.isChecked(),
            bold_annotations=self.bold_annotations.isChecked(),
            bold_titles=self.bold_titles.isChecked(),
        )


def _style_icon() -> QIcon:
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    font = QFont("Times New Roman", 14)
    font.setBold(True)
    painter.setFont(font)
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "Aa")
    painter.end()
    return QIcon(pixmap)


class PlotNavigationToolbar(NavigationToolbar2QT):
    """Matplotlib toolbar with a publication-typography action."""

    def __init__(self, panel: "FigureCanvasPanel") -> None:
        super().__init__(panel.canvas, panel)
        self.panel = panel
        self.style_action = self.addAction(_style_icon(), "Plot typography")
        self.style_action.setToolTip("Set font, tick labels, plot labels, legends, annotations, and bold options")
        self.style_action.triggered.connect(self.panel.open_style_dialog)

        # Place the style button next to Matplotlib's built-in plot-configuration tool
        # and before the save icon, so it feels like part of the native toolbar.
        save_action = next((a for a in self.actions() if "Save" in a.text()), None)
        if save_action is not None:
            self.removeAction(self.style_action)
            self.insertAction(save_action, self.style_action)


class FigureCanvasPanel(QWidget):
    figure_changed = pyqtSignal()
    all_panels: list["FigureCanvasPanel"] = []

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        FigureCanvasPanel.all_panels.append(self)
        self.plot_style = PlotStyle()
        self.figure = Figure(figsize=(7, 5), constrained_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = PlotNavigationToolbar(self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.apply_current_style()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self in FigureCanvasPanel.all_panels:
            FigureCanvasPanel.all_panels.remove(self)
        super().closeEvent(event)

    def set_figure(self, figure: Figure) -> None:
        old_canvas = self.canvas
        old_toolbar = self.toolbar
        self.layout().removeWidget(old_toolbar)
        self.layout().removeWidget(old_canvas)
        old_toolbar.deleteLater()
        old_canvas.deleteLater()
        self.figure = figure
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.toolbar = PlotNavigationToolbar(self)
        self.layout().insertWidget(0, self.toolbar)
        self.layout().addWidget(self.canvas, 1)
        self.apply_current_style(draw=False)
        self.canvas.draw_idle()
        self.figure_changed.emit()

    def apply_current_style(self, draw: bool = True) -> None:
        apply_plot_style(self.figure, self.plot_style)
        if draw:
            self.canvas.draw_idle()

    def open_style_dialog(self) -> None:
        dialog = PlotStyleDialog(self.plot_style, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        new_style = dialog.style()
        if dialog.apply_all.isChecked():
            for panel in list(FigureCanvasPanel.all_panels):
                panel.plot_style = new_style
                panel.apply_current_style()
        else:
            self.plot_style = new_style
            self.apply_current_style()


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
    panel.apply_current_style(draw=False)
    return export_figure(panel.figure, path, dpi=dpi, metadata=metadata)
