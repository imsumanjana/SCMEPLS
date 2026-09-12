from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QStatusBar, QTabWidget, QToolBar

from ..config import APP_NAME, APP_VERSION, DISCLAIMER, OUTPUT_DIR
from .tabs.calibration_tab import CalibrationTab
from .tabs.control_tab import ControlResponseTab
from .tabs.digital_twin_tab import DigitalTwinTab
from .tabs.energy_tab import EnergyTab
from .tabs.overview_tab import OverviewTab
from .tabs.radar_tab import RadarTab
from .tabs.rollout_tab import RolloutTab
from .tabs.vibration_tab import VibrationTab


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__(); self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}"); self.resize(1500,950); self.setMinimumSize(1100,720)
        self.tabs = QTabWidget(); self.tabs.setDocumentMode(True)
        self.overview_tab = OverviewTab(); self.vibration_tab = VibrationTab(); self.control_tab = ControlResponseTab(); self.energy_tab = EnergyTab(); self.radar_tab = RadarTab()
        self.rollout_tab = RolloutTab(); self.digital_twin_tab = DigitalTwinTab(); self.calibration_tab = CalibrationTab()
        for widget,label in ((self.overview_tab,"Overview"),(self.vibration_tab,"Vibration"),(self.control_tab,"Control Response"),(self.energy_tab,"Energy"),(self.radar_tab,"Radar / MCDA"),(self.rollout_tab,"SC-MEPLS Simulation"),(self.digital_twin_tab,"3D Digital Twin"),(self.calibration_tab,"AI Calibration")):
            self.tabs.addTab(widget,label)
        self.rollout_tab.simulation_updated.connect(self.digital_twin_tab.bind_simulation)
        if not self.rollout_tab.df.empty: self.digital_twin_tab.bind_simulation(self.rollout_tab.df,self.rollout_tab.metrics)
        self.setCentralWidget(self.tabs); self._build_menu(); status = QStatusBar(); status.showMessage(DISCLAIMER); self.setStatusBar(status)

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("File"); batch_action = QAction("Export All Current Article Figures…",self); batch_action.triggered.connect(self.export_all); file_menu.addAction(batch_action); file_menu.addSeparator()
        exit_action = QAction("Exit",self); exit_action.triggered.connect(self.close); file_menu.addAction(exit_action)
        help_menu = self.menuBar().addMenu("Help"); about_action = QAction("About",self); about_action.triggered.connect(self.about); help_menu.addAction(about_action)
        toolbar = QToolBar("Main"); toolbar.setMovable(False); toolbar.addAction(batch_action); self.addToolBar(Qt.ToolBarArea.TopToolBarArea,toolbar)

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
                (self.rollout_tab.plot_panel.figure,out/"Fig_SCMEPLS_rollout_dashboard.png",{"analysis":"SC-MEPLS simulation","provenance":self.rollout_tab.provenance.currentText(),"parameters":self.rollout_tab.parameters().__dict__,"metrics":{k:v for k,v in self.rollout_tab.metrics.items() if k != "parameters"}}),
            ]
            if self.digital_twin_tab.timeline is not None:
                exports.append((self.digital_twin_tab.history_plot.figure,out/"Fig_3D_Digital_Twin_Linked_History.png",self.digital_twin_tab.plot_metadata()))
            for figure,path,metadata in exports: export_figure(figure,path,600,metadata)
            if self.vibration_tab.metrics is not None: self.vibration_tab.metrics.to_csv(out/"vibration_metrics.csv",index=False)
            if not self.control_tab.metrics_df.empty: self.control_tab.metrics_df.to_csv(out/"control_response_metrics.csv",index=False)
            if not self.energy_tab.comparison.empty: self.energy_tab.comparison.to_csv(out/"energy_metrics.csv",index=False)
            if not self.radar_tab.ranked.empty: self.radar_tab.ranked.to_csv(out/"radar_weighted_ranking.csv",index=False)
            if not self.rollout_tab.df.empty: self.rollout_tab.df.to_csv(out/"scmepls_time_history.csv",index=False)
            QMessageBox.information(self,"Batch export complete",f"All current figures and result tables were saved to:\n{out}")
        except Exception as exc: QMessageBox.critical(self,"Batch export error",str(exc))

    def about(self) -> None:
        QMessageBox.information(self,"About SC-MEPLS Analysis Studio",f"{APP_NAME} v{APP_VERSION}\n\nSingle-window PyQt6 software for hybrid maglev–mechanical rocket launchpad transport analysis.\n\n{DISCLAIMER}")
