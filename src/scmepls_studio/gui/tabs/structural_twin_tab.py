from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...fea import run_structural_validation
from ...models.rollout_sim import RolloutParameters
from ..common import ScrollableControls, form_row, make_double, section_label


class _StructuralWorker(QObject):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self,
        geometry_path: str,
        manifest_path: str,
        base_parameters: RolloutParameters,
        source_unit: str,
        axis_mode: str,
        rerun_dynamics: bool,
        run_convergence: bool,
    ) -> None:
        super().__init__()
        self.geometry_path = geometry_path
        self.manifest_path = manifest_path
        self.base_parameters = base_parameters
        self.source_unit = source_unit
        self.axis_mode = axis_mode
        self.rerun_dynamics = rerun_dynamics
        self.run_convergence = run_convergence

    @pyqtSlot()
    def run(self) -> None:
        try:
            result = run_structural_validation(
                self.geometry_path,
                self.manifest_path,
                base_rollout_parameters=self.base_parameters,
                source_unit=self.source_unit,
                axis_mode=self.axis_mode,
                rerun_geometry_coupled_dynamics=self.rerun_dynamics,
                run_mesh_convergence=self.run_convergence,
            )
            self.completed.emit(result)
        except Exception as exc:
            self.failed.emit(str(exc))


class StructuralTwinTab(QWidget):
    """GUI for the validated geometry → volume mesh → dynamics → FEA workflow."""

    validation_completed = pyqtSignal(object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.base_parameters = RolloutParameters()
        self.geometry_path: Path | None = None
        self.manifest_path: Path | None = None
        self.report = None
        self.mesh = None
        self.history = None
        self.frame_results: dict[str, object] = {}
        self._thread: QThread | None = None
        self._worker: _StructuralWorker | None = None

        self.controls = ScrollableControls()
        self.controls.add_widget(section_label("Structural digital twin"))
        note = QLabel(
            "Validated path: imported closed GLB/STL body → boundary-conforming tetrahedral mesh → integrated mass/CG/inertia → "
            "geometry-coupled SC-MEPLS dynamics → distributed load patches → linear-elastic node FEA. "
            "The .fea.json manifest is mandatory for quantitative structural analysis."
        )
        note.setWordWrap(True)
        self.controls.add_widget(note)

        self.geometry_btn = QPushButton("Select Structural GLB / STL…")
        self.geometry_label = QLabel("Geometry: not selected"); self.geometry_label.setWordWrap(True)
        self.manifest_btn = QPushButton("Select Structural Manifest (.fea.json)…")
        self.manifest_label = QLabel("Manifest: not selected"); self.manifest_label.setWordWrap(True)
        self.controls.add_widget(self.geometry_btn); self.controls.add_widget(self.geometry_label)
        self.controls.add_widget(self.manifest_btn); self.controls.add_widget(self.manifest_label)

        self.unit_combo = QComboBox(); self.unit_combo.addItems(["m", "mm", "cm"])
        self.axis_combo = QComboBox()
        self.axis_combo.addItem("Auto: GLB Y-up → Z-up; STL as stored", "auto")
        self.axis_combo.addItem("As stored", "as_stored")
        self.axis_combo.addItem("Force GLB Y-up → SC-MEPLS Z-up", "gltf_y_up_to_z_up")
        self.controls.add_widget(form_row("Source geometry unit", self.unit_combo))
        self.controls.add_widget(form_row("Axis conversion", self.axis_combo))

        self.rerun_check = QCheckBox("Use geometry-derived mass/inertia and M1–M8 coordinates in dynamics")
        self.rerun_check.setChecked(True)
        self.convergence_check = QCheckBox("Run coarse / base / fine mesh-convergence study")
        self.convergence_check.setChecked(True)
        self.controls.add_widget(self.rerun_check); self.controls.add_widget(self.convergence_check)

        self.run_btn = QPushButton("Run Structural Validation")
        self.run_btn.setEnabled(False)
        self.controls.add_widget(self.run_btn)
        self.status_label = QLabel("Structural validation: waiting for geometry and manifest")
        self.status_label.setWordWrap(True); self.controls.add_widget(self.status_label)

        self.controls.add_widget(section_label("FEA result view"))
        self.checkpoint_combo = QComboBox(); self.checkpoint_combo.setEnabled(False)
        self.result_combo = QComboBox()
        self.result_combo.addItem("Von Mises stress", "von_mises_pa")
        self.result_combo.addItem("Displacement magnitude", "displacement_magnitude_m")
        self.result_combo.addItem("Safety factor", "safety_factor")
        self.deformation_scale = make_double(1.0, 0.0, 10000.0, 2, 0.5)
        self.controls.add_widget(form_row("Checkpoint", self.checkpoint_combo))
        self.controls.add_widget(form_row("Displayed field", self.result_combo))
        self.controls.add_widget(form_row("Deformation scale", self.deformation_scale))

        self.controls.add_widget(section_label("Structural summary"))
        self.summary_table = QTableWidget(0, 2); self.summary_table.setHorizontalHeaderLabels(["Property", "Value"]); self.summary_table.setMinimumHeight(300)
        self.controls.add_widget(self.summary_table)
        self.controls.add_widget(section_label("Critical checkpoints"))
        self.checkpoint_table = QTableWidget(0, 8)
        self.checkpoint_table.setHorizontalHeaderLabels([
            "Label", "t (s)", "Max disp. (m)", "Max VM (Pa)", "Min SF", "Pre-FEA |F| (N)", "Pre-FEA |M| (N·m)", "Solve residual (N)"
        ])
        self.checkpoint_table.setMinimumHeight(260); self.controls.add_widget(self.checkpoint_table)
        self.controls.add_widget(section_label("Mesh convergence"))
        self.convergence_table = QTableWidget(0, 6)
        self.convergence_table.setHorizontalHeaderLabels(["Element size (m)", "Nodes", "Elements", "Max disp. (m)", "Max VM (Pa)", "Min SF"])
        self.convergence_table.setMinimumHeight(180); self.controls.add_widget(self.convergence_table)
        self.warning_label = QLabel(""); self.warning_label.setWordWrap(True); self.controls.add_widget(self.warning_label)

        export_row = QWidget(); export_layout = QHBoxLayout(export_row); export_layout.setContentsMargins(0, 0, 0, 0)
        self.export_report_btn = QPushButton("Export Validation JSON…")
        self.export_vtu_btn = QPushButton("Export Current FEA VTU…")
        self.export_report_btn.setEnabled(False); self.export_vtu_btn.setEnabled(False)
        export_layout.addWidget(self.export_report_btn); export_layout.addWidget(self.export_vtu_btn)
        self.controls.add_widget(export_row)

        right = QWidget(); right_layout = QVBoxLayout(right); right_layout.setContentsMargins(0,0,0,0)
        right_layout.addWidget(QLabel("Structural volume mesh / deformed FEA field"))
        self.plotter = QtInteractor(right); self.plotter.set_background("white"); self.plotter.add_axes()
        right_layout.addWidget(self.plotter.interactor, 1)
        splitter = QSplitter(Qt.Orientation.Horizontal); splitter.addWidget(self.controls); splitter.addWidget(right); splitter.setSizes([520, 980])
        layout = QHBoxLayout(self); layout.setContentsMargins(6,6,6,6); layout.addWidget(splitter)

        self.geometry_btn.clicked.connect(self.select_geometry)
        self.manifest_btn.clicked.connect(self.select_manifest)
        self.run_btn.clicked.connect(self.run_validation)
        self.checkpoint_combo.currentTextChanged.connect(self._render_checkpoint)
        self.result_combo.currentIndexChanged.connect(self._render_checkpoint)
        self.deformation_scale.valueChanged.connect(self._render_checkpoint)
        self.export_report_btn.clicked.connect(self.export_report)
        self.export_vtu_btn.clicked.connect(self.export_vtu)

    def set_base_parameters(self, parameters: RolloutParameters) -> None:
        self.base_parameters = parameters

    def _refresh_run_enabled(self) -> None:
        self.run_btn.setEnabled(self.geometry_path is not None and self.manifest_path is not None and self._thread is None)

    def select_geometry(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select structural geometry", "", "3D geometry (*.glb *.stl)")
        if not path: return
        self.geometry_path = Path(path)
        self.geometry_label.setText(f"Geometry: {self.geometry_path}")
        candidate = self.geometry_path.with_suffix(".fea.json")
        if candidate.exists():
            self.manifest_path = candidate
            self.manifest_label.setText(f"Manifest: {candidate}")
        self._refresh_run_enabled()

    def select_manifest(self) -> None:
        start = str(self.geometry_path.with_suffix(".fea.json")) if self.geometry_path is not None else ""
        path, _ = QFileDialog.getOpenFileName(self, "Select structural manifest", start, "Structural manifest (*.fea.json *.json)")
        if not path: return
        self.manifest_path = Path(path)
        self.manifest_label.setText(f"Manifest: {self.manifest_path}")
        self._refresh_run_enabled()

    def run_validation(self) -> None:
        if self.geometry_path is None or self.manifest_path is None or self._thread is not None: return
        self.run_btn.setEnabled(False)
        self.status_label.setText("Structural validation running: meshing, coupled dynamics, load mapping and FEA…")
        thread = QThread(self)
        worker = _StructuralWorker(
            str(self.geometry_path),
            str(self.manifest_path),
            self.base_parameters,
            self.unit_combo.currentText(),
            str(self.axis_combo.currentData()),
            self.rerun_check.isChecked(),
            self.convergence_check.isChecked(),
        )
        worker.moveToThread(thread); thread.started.connect(worker.run)
        worker.completed.connect(self._validation_ready); worker.failed.connect(self._validation_failed)
        worker.completed.connect(thread.quit); worker.failed.connect(thread.quit)
        thread.finished.connect(self._thread_finished)
        self._thread = thread; self._worker = worker; thread.start()

    @pyqtSlot(object)
    def _validation_ready(self, payload: object) -> None:
        try:
            report, mesh, history, frame_results = payload
            self.report = report; self.mesh = mesh; self.history = history; self.frame_results = dict(frame_results)
            self.status_label.setText(
                f"Structural validation complete — {mesh.node_count:,} nodes, {mesh.element_count:,} tetrahedra, {len(frame_results)} critical frames"
            )
            self._fill_tables()
            self.checkpoint_combo.blockSignals(True); self.checkpoint_combo.clear(); self.checkpoint_combo.addItems(list(self.frame_results)); self.checkpoint_combo.blockSignals(False)
            self.checkpoint_combo.setEnabled(bool(self.frame_results)); self.export_report_btn.setEnabled(True); self.export_vtu_btn.setEnabled(bool(self.frame_results))
            if self.frame_results:
                self.checkpoint_combo.setCurrentIndex(0); self._render_checkpoint()
            self.validation_completed.emit(history, report)
        except Exception as exc:
            self._validation_failed(str(exc))

    @pyqtSlot(str)
    def _validation_failed(self, message: str) -> None:
        self.status_label.setText(f"Structural validation FAILED: {message}")
        QMessageBox.critical(self, "Structural validation failed", message)

    @pyqtSlot()
    def _thread_finished(self) -> None:
        if self._worker is not None: self._worker.deleteLater()
        if self._thread is not None: self._thread.deleteLater()
        self._worker = None; self._thread = None; self._refresh_run_enabled()

    def _fill_tables(self) -> None:
        if self.report is None: return
        r = self.report
        summary = [
            ("Mass scope", r.mass_scope), ("Structural component", r.structural_component or "single body"),
            ("Nodes", f"{r.node_count:,}"), ("Elements", f"{r.element_count:,}"), ("Volume", f"{r.volume_m3:.6g} m³"),
            ("Integrated mass", f"{r.mass_kg:.6g} kg"), ("Centroid", ", ".join(f"{v:.6g}" for v in r.centroid_m) + " m"),
            ("Dimensions", " × ".join(f"{v:.6g}" for v in r.dimensions_m) + " m"),
            ("Mapping max snap", f"{r.mapping_max_snap_distance_m:.6g} m"),
            ("Min mean-ratio quality", f"{r.mesh_quality.minimum_mean_ratio:.5g}"),
            ("Poor elements", str(r.mesh_quality.poor_element_count)),
            ("Non-positive Jacobians", str(r.mesh_quality.nonpositive_jacobian_count)),
            ("Convergence Δ displacement", "not run" if r.convergence_displacement_change_percent is None else f"{r.convergence_displacement_change_percent:.4g}%"),
            ("Convergence Δ stress", "not run" if r.convergence_stress_change_percent is None else f"{r.convergence_stress_change_percent:.4g}%"),
        ]
        self.summary_table.setRowCount(len(summary))
        for row, (key, value) in enumerate(summary):
            self.summary_table.setItem(row,0,QTableWidgetItem(str(key))); self.summary_table.setItem(row,1,QTableWidgetItem(str(value)))
        self.summary_table.resizeColumnsToContents()

        self.checkpoint_table.setRowCount(len(r.checkpoints))
        for row, cp in enumerate(r.checkpoints):
            values = [cp.label, cp.time_s, cp.max_displacement_m, cp.max_von_mises_pa, cp.minimum_safety_factor, cp.pre_fea_force_residual_norm_n, cp.pre_fea_moment_residual_norm_nm, cp.equilibrium_residual_norm_n]
            for col, value in enumerate(values):
                text = str(value) if col == 0 else f"{float(value):.6g}"
                self.checkpoint_table.setItem(row,col,QTableWidgetItem(text))
        self.checkpoint_table.resizeColumnsToContents()

        self.convergence_table.setRowCount(len(r.convergence))
        for row, point in enumerate(r.convergence):
            values = [point.element_size_m, point.node_count, point.element_count, point.max_displacement_m, point.max_von_mises_pa, point.minimum_safety_factor]
            for col, value in enumerate(values): self.convergence_table.setItem(row,col,QTableWidgetItem(f"{value:.6g}" if isinstance(value,float) else str(value)))
        self.convergence_table.resizeColumnsToContents()
        self.warning_label.setText("Warnings:\n" + ("\n".join(f"• {w}" for w in r.warnings) if r.warnings else "None"))

    def _render_checkpoint(self) -> None:
        if self.mesh is None or not self.frame_results: return
        label = self.checkpoint_combo.currentText()
        if not label or label not in self.frame_results: return
        result = self.frame_results[label]
        scale = self.deformation_scale.value()
        grid = result.to_pyvista(self.mesh, deformation_scale=scale)
        field = str(self.result_combo.currentData())
        if field == "safety_factor":
            vm = np.asarray(grid.point_data["von_mises_pa"], dtype=float)
            yield_strength = float(result.fea.safety_factor_min * result.fea.max_von_mises_pa) if result.fea.max_von_mises_pa > 0 else 1.0
            grid.point_data["safety_factor"] = np.divide(yield_strength, vm, out=np.full_like(vm, np.inf), where=vm > 0)
            scalars = "safety_factor"
        else:
            scalars = field
        self.plotter.clear(); self.plotter.set_background("white"); self.plotter.add_axes()
        self.plotter.add_mesh(grid, scalars=scalars, show_edges=True, scalar_bar_args={"title": scalars.replace("_", " ")})
        self.plotter.reset_camera(); self.plotter.view_isometric(); self.plotter.render()

    def export_report(self) -> None:
        if self.report is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Export structural validation report", "scmepls_structural_validation.json", "JSON files (*.json)")
        if not path: return
        output = Path(path); output = output if output.suffix.lower() == ".json" else output.with_suffix(".json")
        output.write_text(json.dumps(self.report.to_dict(), indent=2, default=str), encoding="utf-8")
        QMessageBox.information(self, "Structural report exported", str(output))

    def export_vtu(self) -> None:
        if self.mesh is None or not self.frame_results: return
        label = self.checkpoint_combo.currentText()
        if not label: return
        path, _ = QFileDialog.getSaveFileName(self, "Export current FEA field", f"scmepls_fea_{label}.vtu", "VTU files (*.vtu)")
        if not path: return
        output = Path(path); output = output if output.suffix.lower() == ".vtu" else output.with_suffix(".vtu")
        result = self.frame_results[label]
        result.to_pyvista(self.mesh, deformation_scale=self.deformation_scale.value()).save(str(output))
        QMessageBox.information(self, "FEA VTU exported", str(output))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try: self.plotter.close()
        except Exception: pass
        super().closeEvent(event)
