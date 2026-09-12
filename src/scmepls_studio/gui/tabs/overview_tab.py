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
            "simulation, external GLB/STL 3D visualization, and optional data-driven calibration.<br><br>"
            "<b>Scientific workflow.</b> Use illustrative defaults only for workflow demonstration. Replace them "
            "with literature-derived, simulated, or experimental data before making quantitative claims. The "
            "SC-MEPLS time history now exposes true/measured/estimated gaps and M1–M8 current, force and pressure "
            "signals for independent MATLAB validation. Imported 3D geometry remains visualization-only.<br><br>"
            "<b>3D workflow.</b> Prefer named-node GLB assemblies and an optional <code>.manifest.json</code> sidecar "
            "for component roles and M1–M8 bindings. STL remains supported with explicit unit selection. Geometry "
            "does not alter mass, inertia, actuator coordinates, or controller equations.<br><br>"
            "<b>Reproducibility.</b> Publication PNG exports include metadata sidecars with software version and "
            "provenance. AI calibration exports include validation strategy, feature list, software versions, and a "
            "SHA-256 fingerprint of the calibration table used for fitting.<br><br>"
            f"<b>Limitation.</b> {DISCLAIMER}"
        )
        text.setWordWrap(True)
        text.setOpenExternalLinks(False)
        layout.addWidget(text)
        layout.addStretch(1)
