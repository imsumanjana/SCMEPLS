from __future__ import annotations

import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPushButton, QSplitter,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from ...models.rollout_sim import (
    MAX_GAP_M, MAX_TIME_STEP_S, MIN_COMPLETE_RUN_TIME_S, MIN_GAP_M,
    RolloutParameters, simulate_rollout,
)
from ...plotting.figures import rollout_dashboard_figure
from ..common import ExportBar, FigureCanvasPanel, ScrollableControls, form_row, make_double, make_int, provenance_combo, save_panel_figure, section_label


class RolloutTab(QWidget):
    simulation_updated = pyqtSignal(object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controls = ScrollableControls(); self.plot_panel = FigureCanvasPanel(); self.provenance = provenance_combo("Simulation")
        self.controls.add_widget(section_label("Reduced-order SC-MEPLS roll-out and load-transfer simulation"))
        note = QLabel("The model uses eight electromagnetic lift modules, force-balanced electromagnetic-to-pneumatic/mechanical handover, actuator time constants, center-of-gravity shift, wind disturbance, coil degradation, sensor bias, and pneumatic leakage. It is a reduced-order engineering model for feasibility assessment; safety-critical use requires validated parameters and independent review.")
        note.setWordWrap(True); self.controls.add_widget(note); self.controls.add_widget(form_row("Provenance", self.provenance))
        self.dt = make_double(0.01,0.001,MAX_TIME_STEP_S,4,0.001); self.end_time = make_double(80.0,MIN_COMPLETE_RUN_TIME_S,500.0,2,5.0)
        self.track_length = make_double(1.50,0.1,100.0,3,0.1); self.platform_mass = make_double(25.0,0.1,1e6,3,1.0); self.payload_mass = make_double(50.0,0.0,1e6,3,1.0)
        self.initial_gap = make_double(0.002,MIN_GAP_M,MAX_GAP_M,5,0.001); self.target_gap = make_double(0.010,MIN_GAP_M,MAX_GAP_M,5,0.001)
        self.cgx = make_double(0.035,-1.0,1.0,4,0.005); self.cgy = make_double(-0.025,-1.0,1.0,4,0.005)
        self.wind_start = make_double(20.0,0.0,500.0,2,1.0); self.wind_end = make_double(50.0,0.0,500.0,2,1.0)
        self.coil_fault_time = make_double(28.0,0.0,500.0,2,1.0); self.coil_fault_index = make_int(3,0,8)
        self.sensor_fault_start = make_double(36.0,0.0,500.0,2,1.0); self.sensor_fault_end = make_double(45.0,0.0,500.0,2,1.0); self.sensor_fault_index = make_int(5,0,8)
        self.leak_time = make_double(56.0,0.0,500.0,2,1.0); self.leak_index = make_int(6,0,8)
        for label,widget in [
            ("Time step (s)",self.dt),("End time (s)",self.end_time),("Track length (m)",self.track_length),("Platform mass (kg)",self.platform_mass),("Payload mass (kg)",self.payload_mass),
            ("Initial gap (m)",self.initial_gap),("Target gap (m)",self.target_gap),("CG shift x (m)",self.cgx),("CG shift y (m)",self.cgy),("Wind start (s)",self.wind_start),("Wind end (s)",self.wind_end),
            ("Coil fault time (s)",self.coil_fault_time),("Coil fault module (0=off)",self.coil_fault_index),("Sensor fault start (s)",self.sensor_fault_start),("Sensor fault end (s)",self.sensor_fault_end),
            ("Sensor fault module (0=off)",self.sensor_fault_index),("Pneumatic leak time (s)",self.leak_time),("Leak module (0=off)",self.leak_index),
        ]: self.controls.add_widget(form_row(label,widget))
        self.run_btn = QPushButton("Run SC-MEPLS Simulation"); self.controls.add_widget(self.run_btn); self.save_history_btn = QPushButton("Save Time-History CSV"); self.controls.add_widget(self.save_history_btn)
        self.metrics_table = QTableWidget(0,2); self.metrics_table.setHorizontalHeaderLabels(["Metric","Value"]); self.metrics_table.setMinimumHeight(260); self.controls.add_widget(section_label("Validation metrics")); self.controls.add_widget(self.metrics_table)
        self.export_bar = ExportBar("scmepls_rollout_dashboard_600dpi.png"); self.controls.add_widget(self.export_bar)
        splitter = QSplitter(Qt.Orientation.Horizontal); splitter.addWidget(self.controls)
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0); rl.addWidget(QLabel("Roll-out, levitation, load-transfer, and fault dashboard")); rl.addWidget(self.plot_panel,1); splitter.addWidget(right); splitter.setSizes([450,980])
        layout = QHBoxLayout(self); layout.setContentsMargins(6,6,6,6); layout.addWidget(splitter)
        self.run_btn.clicked.connect(self.run_analysis); self.save_history_btn.clicked.connect(self.save_history); self.export_bar.export_requested.connect(self.export_plot)
        self.df = pd.DataFrame(); self.metrics: dict = {}; self.run_analysis()

    def parameters(self) -> RolloutParameters:
        return RolloutParameters(dt_s=self.dt.value(),end_time_s=self.end_time.value(),track_length_m=self.track_length.value(),platform_mass_kg=self.platform_mass.value(),payload_mass_kg=self.payload_mass.value(),initial_gap_m=self.initial_gap.value(),target_gap_m=self.target_gap.value(),cg_shift_x_m=self.cgx.value(),cg_shift_y_m=self.cgy.value(),wind_start_s=self.wind_start.value(),wind_end_s=self.wind_end.value(),coil_fault_time_s=self.coil_fault_time.value(),coil_fault_index=self.coil_fault_index.value(),sensor_fault_start_s=self.sensor_fault_start.value(),sensor_fault_end_s=self.sensor_fault_end.value(),sensor_fault_index=self.sensor_fault_index.value(),leak_time_s=self.leak_time.value(),leak_index=self.leak_index.value())

    def run_analysis(self) -> None:
        try:
            self.df,self.metrics = simulate_rollout(self.parameters()); self.plot_panel.set_figure(rollout_dashboard_figure(self.df,float(self.metrics["design_weight_n"])))
            display = [(k,v) for k,v in self.metrics.items() if k != "parameters"]; self.metrics_table.setRowCount(len(display))
            for r,(key,value) in enumerate(display): self.metrics_table.setItem(r,0,QTableWidgetItem(key.replace("_"," ").title())); self.metrics_table.setItem(r,1,QTableWidgetItem(f"{value:.6g}" if isinstance(value,(int,float)) else str(value)))
            self.metrics_table.resizeColumnsToContents(); self.simulation_updated.emit(self.df,self.metrics)
        except Exception as exc: QMessageBox.critical(self,"SC-MEPLS simulation error",str(exc))

    def save_history(self) -> None:
        if self.df.empty: return
        path,_ = QFileDialog.getSaveFileName(self,"Save simulation time history","scmepls_time_history.csv","CSV files (*.csv)")
        if path: self.df.to_csv(path,index=False)

    def export_plot(self,dpi:int,path:str) -> None:
        try:
            out,meta = save_panel_figure(self,self.plot_panel,dpi,path,{"analysis":"SC-MEPLS reduced-order rollout and force-balanced load transfer","provenance":self.provenance.currentText(),"parameters":self.parameters().__dict__,"metrics":{k:v for k,v in self.metrics.items() if k != "parameters"},"warning":"Reduced-order simulation; replace with identified and validated parameters before quantitative design claims."})
            QMessageBox.information(self,"Export complete",f"Saved:\n{out}\n{meta}")
        except Exception as exc: QMessageBox.critical(self,"Export error",str(exc))
