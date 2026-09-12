from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QStatusBar, QTabWidget, QToolBar, QVBoxLayout, QWidget

from ..config import APP_NAME, APP_VERSION, DISCLAIMER, OUTPUT_DIR
from ..models.rollout_sim import RolloutParameters
from .tabs.calibration_tab import CalibrationTab
from .tabs.control_tab import ControlResponseTab
from .tabs.digital_twin_tab import DigitalTwinTab
from .tabs.energy_tab import EnergyTab
from .tabs.overview_tab import OverviewTab
from .tabs.radar_tab import RadarTab
from .tabs.rollout_tab import RolloutTab
from .tabs.structural_twin_tab import StructuralTwinTab
from .tabs.vibration_tab import VibrationTab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__(); self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}"); self.resize(1500,950); self.setMinimumSize(1100,720)
        self.tabs = QTabWidget(); self.tabs.setDocumentMode(True)
        self.overview_tab = OverviewTab()
        self.rollout_tab = RolloutTab()
        self.digital_twin_tab = DigitalTwinTab()
        self.structural_tab = StructuralTwinTab()
        self.vibration_tab = VibrationTab(); self.control_tab = ControlResponseTab(); self.energy_tab = EnergyTab(); self.radar_tab = RadarTab(); self.calibration_tab = CalibrationTab()

        supporting = QWidget(); supporting_layout = QVBoxLayout(supporting); supporting_layout.setContentsMargins(0,0,0,0)
        supporting_tabs = QTabWidget(); supporting_tabs.setDocumentMode(True)
        for widget,label in ((self.vibration_tab,"Vibration"),(self.control_tab,"Control Response"),(self.energy_tab,"Energy"),(self.radar_tab,"Radar / MCDA"),(self.calibration_tab,"AI Calibration")):
            supporting_tabs.addTab(widget,label)
        supporting_layout.addWidget(supporting_tabs)
        self.supporting_tabs = supporting_tabs

        for widget,label in (
            (self.overview_tab,"Overview"),
            (self.rollout_tab,"SC-MEPLS Simulation"),
            (self.digital_twin_tab,"3D Digital Twin"),
            (self.structural_tab,"Structural FEA"),
            (supporting,"Supporting Analyses"),
        ):
            self.tabs.addTab(widget,label)

        self.rollout_tab.simulation_updated.connect(self._simulation_updated)
        self.structural_tab.validation_completed.connect(self._structural_validation_completed)
        if not self.rollout_tab.df.empty:
            self._simulation_updated(self.rollout_tab.df,self.rollout_tab.metrics)
        self.setCentralWidget(self.tabs); self._build_menu(); status = QStatusBar(); status.showMessage(DISCLAIMER); self.setStatusBar(status)

    def _simulation_updated(self, history: object, metrics: object) -> None:
        self.digital_twin_tab.bind_simulation(history, metrics)
        try:
            active = self.rollout_tab.geometry_parameters if self.rollout_tab.source_combo.currentData() == "geometry" and self.rollout_tab.geometry_parameters is not None else self.rollout_tab.parameters()
            self.structural_tab.set_base_parameters(active)
        except Exception:
            pass

    def _structural_validation_completed(self, history: object, parameters: object, report: object) -> None:
        if isinstance(parameters, RolloutParameters):
            self.rollout_tab.bind_geometry_coupled_result(history, parameters, report)
        self.digital_twin_tab.bind_simulation(history, {"source":"structural_validation","structural_report": getattr(report,"to_dict",lambda: {})()})
        self.statusBar().showMessage("Structural validation completed; validated geometry-coupled dynamics are available in SC-MEPLS Simulation.", 10000)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File")
        open_project = QAction("Open Project…", self); open_project.triggered.connect(self.open_project); file_menu.addAction(open_project)
        save_project = QAction("Save Project…", self); save_project.triggered.connect(self.save_project); file_menu.addAction(save_project)
        file_menu.addSeparator()
        batch_action = QAction("Export All Current Article Figures…",self); batch_action.triggered.connect(self.export_all); file_menu.addAction(batch_action); file_menu.addSeparator()
        exit_action = QAction("Exit",self); exit_action.triggered.connect(self.close); file_menu.addAction(exit_action)
        help_menu = self.menuBar().addMenu("Help"); about_action = QAction("About",self); about_action.triggered.connect(self.about); help_menu.addAction(about_action)
        toolbar = QToolBar("Main"); toolbar.setMovable(False); toolbar.addAction(open_project); toolbar.addAction(save_project); toolbar.addSeparator(); toolbar.addAction(batch_action); self.addToolBar(Qt.ToolBarArea.TopToolBarArea,toolbar)

    @staticmethod
    def _camera_json(plotter) -> object:
        try:
            camera = plotter.camera_position
            return [[float(v) for v in point] for point in camera]
        except Exception:
            return None

    def _project_payload(self) -> dict:
        active_parameters = self.rollout_tab.geometry_parameters if self.rollout_tab.source_combo.currentData() == "geometry" and self.rollout_tab.geometry_parameters is not None else self.rollout_tab.parameters()
        structural_report = self.structural_tab.report.to_dict() if self.structural_tab.report is not None else None
        geometry_asset = self.digital_twin_tab.asset
        geometry_manifest = self.digital_twin_tab.manifest
        return {
            "schema_version": 1,
            "software_version": APP_VERSION,
            "rollout_parameters": asdict(active_parameters),
            "requested_dynamics_source": str(self.rollout_tab.source_combo.currentData()),
            "structural": {
                "geometry_path": None if self.structural_tab.geometry_path is None else str(self.structural_tab.geometry_path),
                "manifest_path": None if self.structural_tab.manifest_path is None else str(self.structural_tab.manifest_path),
                "source_unit": self.structural_tab.unit_combo.currentText(),
                "axis_mode": str(self.structural_tab.axis_combo.currentData()),
                "rerun_geometry_coupled_dynamics": self.structural_tab.rerun_check.isChecked(),
                "run_mesh_convergence": self.structural_tab.convergence_check.isChecked(),
                "latest_validation_report": structural_report,
                "camera": self._camera_json(self.structural_tab.plotter),
            },
            "digital_twin": {
                "geometry_path": None if geometry_asset is None else str(geometry_asset.source_path),
                "manifest_path": None if geometry_manifest is None else str(geometry_manifest.path),
                "selected_component": self.digital_twin_tab.component_combo.currentText() or None,
                "result_metric": str(self.digital_twin_tab.metric_combo.currentData()),
                "source_unit": self.digital_twin_tab.unit_combo.currentText(),
                "axis_mode": str(self.digital_twin_tab.axis_combo.currentData()),
                "camera": self._camera_json(self.digital_twin_tab.plotter),
            },
        }

    def save_project(self) -> None:
        path,_ = QFileDialog.getSaveFileName(self,"Save SC-MEPLS project","scmepls_project.json","SC-MEPLS project (*.json)")
        if not path: return
        output = Path(path); output = output if output.suffix.lower() == ".json" else output.with_suffix(".json")
        try:
            output.write_text(json.dumps(self._project_payload(),indent=2,default=str),encoding="utf-8")
            QMessageBox.information(self,"Project saved",str(output))
        except Exception as exc: QMessageBox.critical(self,"Project save error",str(exc))

    def _apply_rollout_parameters(self, params: RolloutParameters) -> None:
        tab = self.rollout_tab; tab._advanced_values = params
        tab.dt.setValue(params.dt_s); tab.end_time.setValue(params.end_time_s); tab.track_length.setValue(params.track_length_m)
        tab.platform_mass.setValue(params.platform_mass_kg); tab.payload_mass.setValue(params.payload_mass_kg)
        tab.initial_gap.setValue(params.initial_gap_m); tab.target_gap.setValue(params.target_gap_m); tab.cgx.setValue(params.cg_shift_x_m); tab.cgy.setValue(params.cg_shift_y_m)
        tab.wind_start.setValue(params.wind_start_s); tab.wind_end.setValue(params.wind_end_s); tab.coil_fault_time.setValue(params.coil_fault_time_s); tab.coil_fault_index.setValue(params.coil_fault_index)
        tab.sensor_fault_start.setValue(params.sensor_fault_start_s); tab.sensor_fault_end.setValue(params.sensor_fault_end_s); tab.sensor_fault_index.setValue(params.sensor_fault_index)
        tab.leak_time.setValue(params.leak_time_s); tab.leak_index.setValue(params.leak_index)

    def open_project(self) -> None:
        path,_ = QFileDialog.getOpenFileName(self,"Open SC-MEPLS project","","SC-MEPLS project (*.json)")
        if not path: return
        try:
            raw = json.loads(Path(path).read_text(encoding="utf-8"))
            if int(raw.get("schema_version",0)) != 1: raise ValueError("Unsupported project schema version.")
            params = RolloutParameters(**raw.get("rollout_parameters",{})); self._apply_rollout_parameters(params)
            structural = raw.get("structural",{}) if isinstance(raw.get("structural",{}),dict) else {}
            geometry = structural.get("geometry_path"); manifest = structural.get("manifest_path")
            self.structural_tab.geometry_path = Path(geometry) if geometry else None; self.structural_tab.manifest_path = Path(manifest) if manifest else None
            self.structural_tab.geometry_label.setText("Geometry: not selected" if not geometry else f"Geometry: {geometry}")
            self.structural_tab.manifest_label.setText("Manifest: not selected" if not manifest else f"Manifest: {manifest}")
            if structural.get("source_unit") in {"m","mm","cm"}: self.structural_tab.unit_combo.setCurrentText(structural["source_unit"])
            axis_index = self.structural_tab.axis_combo.findData(structural.get("axis_mode","auto")); self.structural_tab.axis_combo.setCurrentIndex(max(0,axis_index))
            self.structural_tab.rerun_check.setChecked(bool(structural.get("rerun_geometry_coupled_dynamics",True))); self.structural_tab.convergence_check.setChecked(bool(structural.get("run_mesh_convergence",True))); self.structural_tab._refresh_run_enabled()
            self.rollout_tab.source_combo.setCurrentIndex(0); self.rollout_tab.run_analysis()
            QMessageBox.information(self,"Project opened","Parameters and structural file references were restored. Re-run Structural Validation to recreate volume meshes/FEA fields and revalidate external files.")
        except Exception as exc: QMessageBox.critical(self,"Project open error",str(exc))

    def export_all(self) -> None:
        folder = QFileDialog.getExistingDirectory(self,"Select output folder",str(OUTPUT_DIR))
        if not folder: return
        out = Path(folder); out.mkdir(parents=True,exist_ok=True)
        try:
            from ..io.export import export_figure
            exports = [
                (self.vibration_tab.plot_panel.figure,out/"Fig_vibration_normalized_RMS.png",{"analysis":"Vibration","provenance":self.vibration_tab.provenance.currentText(),"metrics":None if self.vibration_tab.metrics is None else self.vibration_tab.metrics.to_dict(orient="records")}),
                (self.control_tab.plot_panel.figure,out/"Fig_closed_loop_step_response.png",{"analysis":"Control response","provenance":self.control_tab.provenance.currentText(),"models":{"Maglev System":self.control_tab.maglev.parameters().__dict__,"Conventional System":self.control_tab.conventional.parameters().__dict__},"metrics":self.control_tab.metrics_df.to_dict(orient="records")}),
                (self.energy_tab.plot_panel.figure,out/"Fig_energy_performance.png",{"analysis":"Energy","provenance":self.energy_tab.provenance.currentText(),"inputs":{"Crawler":self.energy_tab.crawler.values().__dict__,"Rail":self.energy_tab.rail.values().__dict__,"Maglev":self.energy_tab.maglev.values().__dict__},"metrics":self.energy_tab.comparison.to_dict(orient="records")}),
                (self.radar_tab.plot_panel.figure,out/"Fig_multicriteria_radar.png",{"analysis":"Radar/MCDA","provenance":self.radar_tab.provenance.currentText(),"weights":{c.name:c.weight for c in self.radar_tab.criteria},"scores":self.radar_tab.scores.to_dict(orient="records"),"ranking":self.radar_tab.ranked.to_dict(orient="records")}),
                (self.rollout_tab.plot_panel.figure,out/"Fig_SCMEPLS_rollout_dashboard.png",{"analysis":"SC-MEPLS simulation","provenance":self.rollout_tab.provenance.currentText(),"parameters":self.rollout_tab.metrics.get("parameters",{}),"metrics":{k:v for k,v in self.rollout_tab.metrics.items() if k != "parameters"}}),
            ]
            if self.digital_twin_tab.timeline is not None: exports.append((self.digital_twin_tab.history_plot.figure,out/"Fig_3D_Digital_Twin_Linked_History.png",self.digital_twin_tab.plot_metadata()))
            for figure,path,metadata in exports: export_figure(figure,path,600,metadata)
            if self.vibration_tab.metrics is not None: self.vibration_tab.metrics.to_csv(out/"vibration_metrics.csv",index=False)
            if not self.control_tab.metrics_df.empty: self.control_tab.metrics_df.to_csv(out/"control_response_metrics.csv",index=False)
            if not self.energy_tab.comparison.empty: self.energy_tab.comparison.to_csv(out/"energy_metrics.csv",index=False)
            if not self.radar_tab.ranked.empty: self.radar_tab.ranked.to_csv(out/"radar_weighted_ranking.csv",index=False)
            if not self.rollout_tab.df.empty: self.rollout_tab.df.to_csv(out/"scmepls_time_history.csv",index=False)
            if self.structural_tab.report is not None: (out/"structural_validation.json").write_text(json.dumps(self.structural_tab.report.to_dict(),indent=2,default=str),encoding="utf-8")
            QMessageBox.information(self,"Batch export complete",f"All current figures and result tables were saved to:\n{out}")
        except Exception as exc: QMessageBox.critical(self,"Batch export error",str(exc))

    def about(self) -> None:
        QMessageBox.information(self,"About SC-MEPLS Analysis Studio",f"{APP_NAME} v{APP_VERSION}\n\nSingle-window PyQt6 software for hybrid maglev–mechanical rocket launchpad transport, imported-geometry digital-twin and structural FEA analysis.\n\n{DISCLAIMER}")
