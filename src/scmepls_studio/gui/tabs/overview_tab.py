from __future__ import annotations

from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ...config import APP_NAME, APP_VERSION, DISCLAIMER


class OverviewTab(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        title = QLabel(f"<h1>{APP_NAME}</h1><h3>Version {APP_VERSION}</h3>")
        title.setWordWrap(True); layout.addWidget(title)
        text = QLabel(
            "<b>Primary workflow.</b> SC-MEPLS Simulation → imported-CAD 3D Digital Twin → Structural FEA. "
            "Supporting vibration, control, energy, MCDA and AI tools are grouped separately.<br><br>"
            "<b>Dynamics.</b> The eight-module model uses one right-handed X-roll-out/Y-lateral/Z-vertical convention, "
            "parameterized M1–M8 coordinates, true/sensor/estimated gaps, rigid-plane sensor reconstruction, actuator dynamics, "
            "faults/disturbances and interlocked EM→pneumatic/mechanical load transfer. Unsafe or detected-interlock states do not force lock engagement.<br><br>"
            "<b>3D workflow.</b> GLB/STL surface geometry can be used for visualization without changing physics. Named-node manifests "
            "bind components and may define physical lock/actuator strokes. After structural validation, the viewer uses validated CG and M1–M8 locations.<br><br>"
            "<b>Structural workflow.</b> A mandatory <code>.fea.json</code> manifest defines the closed structural body, material, mesh, "
            "M1–M8, propulsion, wind, lock and optional payload support patches. The software integrates mass/CG/inertia, reruns geometry-coupled dynamics, "
            "distributes loads over boundary patches, audits force/moment balance, solves linear-elastic tetrahedral FEA and can run coarse/base/fine convergence.<br><br>"
            "<b>Scope.</b> The built-in FEA is homogeneous isotropic, small-strain and quasi-static. Joint/contact, heterogeneous multi-material, buckling, fatigue, plasticity, flexible-body dynamics and certification require higher-fidelity independent validation.<br><br>"
            "<b>Reproducibility.</b> Project files preserve parameter and geometry references. Figures/results retain software/provenance metadata. AI models record cross-validation metrics, possible leakage warnings and dataset fingerprints and can be checked on an untouched external CSV.<br><br>"
            f"<b>Limitation.</b> {DISCLAIMER}"
        )
        text.setWordWrap(True); text.setOpenExternalLinks(False); layout.addWidget(text); layout.addStretch(1)
