from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ...config import APP_NAME, APP_VERSION, DISCLAIMER


class OverviewTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        title = QLabel(f"<h1>{APP_NAME}</h1><h3>Version {APP_VERSION}</h3>")
        title.setWordWrap(True)
        layout.addWidget(title)
        text = QLabel(
            "<b>Purpose.</b> A single-window PyQt6 application for the hybrid maglev–mechanical "
            "rocket launchpad roll-out and locking project. It supports vibration assessment, closed-loop "
            "response analysis, energy accounting, multi-criteria comparison, reduced-order roll-out/load-transfer "
            "simulation, and optional data-driven calibration.<br><br>"
            "<b>Scientific workflow.</b> Use illustrative defaults only for workflow demonstration. Replace them "
            "with literature-derived, simulated, or experimental data before making quantitative claims. Every PNG "
            "export is accompanied by a JSON metadata file containing DPI and provenance.<br><br>"
            "<b>Primary article outputs.</b> Normalized RMS vibration comparison, closed-loop step response, "
            "specific energy/energy-performance comparison, radar chart with consistent benefit directions, and "
            "SC-MEPLS roll-out/load-transfer dashboards.<br><br>"
            f"<b>Limitation.</b> {DISCLAIMER}"
        )
        text.setWordWrap(True)
        text.setTextInteractionFlags(text.textInteractionFlags())
        layout.addWidget(text)
        layout.addStretch(1)
