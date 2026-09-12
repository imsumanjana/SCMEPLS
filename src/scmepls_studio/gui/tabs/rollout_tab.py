from __future__ import annotations

from dataclasses import fields

import pandas as pd
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...models.rollout_sim import (
    MAX_GAP_M,
    MAX_TIME_STEP_S,
    MIN_COMPLETE_RUN_TIME_S,
    MIN_GAP_M,
    RolloutParameters,
    simulate_rollout,
)
from ...plotting.figures import rollout_dashboard_figure
from ..common import ExportBar, FigureCanvasPanel, ScrollableControls, form_row, make_double, make_int, provenance_combo, save_panel_figure, section_label


_BASIC_FIELDS = {
    "dt_s", "end_time_s", "track_length_m", "platform_mass_kg", "payload_mass_kg",
    "initial_gap_m", "target_gap_m", "cg_shift_x_m", "cg_shift_y_m", "wind_start_s", "wind_end_s",
    "coil_fault_time_s", "coil_fault_index", "sensor_fault_start_s", "sensor_fault_end_s", "sensor_fault_index",
    "leak_time_s", "leak_index",
}


class _AdvancedParameterDialog(QDialog):
    """Edit every non-basic RolloutParameters field without crowding the main tab."""

    def __init__(self, values: RolloutParameters, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced SC-MEPLS Physical Parameters")
        self.resize(680, 760)
        layout = QVBoxLayout(self)
        note = QLabel(
            "These values define actuator laws, controller gains, damping, mechanical lock stiffness, interlock tolerances, "
            "phase timing, fault magnitudes, body inertia and M1–M8 coordinates. All values are exported with the simulation."
        )
        note.setWordWrap(True); layout.addWidget(note)
        scroll = QScrollArea(); scroll.setWidgetResizable(True)
        content = QWidget(); form = QFormLayout(content)
        self.editors: dict[str, QLineEdit] = {}
        for field in fields(RolloutParameters):
            if field.name in _BASIC_FIELDS:
                continue
            value = getattr(values, field.name)
            edit = QLineEdit()
            if isinstance(value, tuple):
                edit.setText(", ".join(f"{float(v):.12g}" for v in value))
            else:
                edit.setText(str(value))
            edit.setToolTip(field.name)
            self.editors[field.name] = edit
            form.addRow(field.name.replace("_", " "), edit)
        scroll.setWidget(content); layout.addWidget(scroll, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Reset)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject); buttons.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(self._reset)
        layout.addWidget(buttons)

    def _reset(self) -> None:
        defaults = RolloutParameters()
        for name, edit in self.editors.items():
            value = getattr(defaults, name)
            edit.setText(", ".join(str(v) for v in value) if isinstance(value, tuple) else str(value))

    def parsed(self, base: RolloutParameters) -> dict[str, object]:
        result: dict[str, object] = {}
        for field in fields(RolloutParameters):
            name = field.name
            if name not in self.editors:
                continue
            current = getattr(base, name)
            text = self.editors[name].text().strip()
            try:
                if isinstance(current, tuple):
                    parsed = tuple(float(part.strip()) for part in text.split(",") if part.strip())
                    result[name] = parsed
                elif isinstance(current, int) and not isinstance(current, bool):
                    result[name] = int(text)
                else:
                    result[name] = float(text)
            except ValueError as exc:
                raise ValueError(f"Invalid advanced parameter '{name}': {text}") from exc
        return result


class RolloutTab(QWidget):
    simulation_updated = pyqtSignal(object, object)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controls = ScrollableControls(); self.plot_panel = FigureCanvasPanel(); self.provenance = provenance_combo("Simulation")
        self.controls.add_widget(section_label("SC-MEPLS reduced-order dynamics"))
        note = QLabel(
            "Eight-module rigid-body engineering model with right-handed X roll-out / Y lateral / Z vertical convention, "
            "fault-tolerant gap estimation, interlocked EM→pneumatic/mechanical load transfer, actuator dynamics, CG shift and disturbances. "
            "Use the Structural FEA tab for validated geometry-derived mass/inertia and M1–M8 coordinates."
        )
        note.setWordWrap(True); self.controls.add_widget(note); self.controls.add_widget(form_row("Provenance", self.provenance))

        self.source_combo = QComboBox()
        self.source_combo.addItem("Manual / configured reduced-order parameters", "manual")
        self.source_combo.addItem("Validated geometry-coupled history", "geometry")
        self.controls.add_widget(form_row("Dynamics source", self.source_combo))
        self.geometry_status = QLabel("Geometry-coupled history: not available")
        self.geometry_status.setWordWrap(True); self.controls.add_widget(self.geometry_status)

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
            ("Initial / seated gap (m)",self.initial_gap),("Levitation target gap (m)",self.target_gap),("CG shift x (m)",self.cgx),("CG shift y (m)",self.cgy),("Wind start (s)",self.wind_start),("Wind end (s)",self.wind_end),
            ("Coil fault time (s)",self.coil_fault_time),("Coil fault module (0=off)",self.coil_fault_index),("Sensor fault start (s)",self.sensor_fault_start),("Sensor fault end (s)",self.sensor_fault_end),
            ("Sensor fault module (0=off)",self.sensor_fault_index),("Pneumatic leak time (s)",self.leak_time),("Leak module (0=off)",self.leak_index),
        ]: self.controls.add_widget(form_row(label,widget))

        self._advanced_values = RolloutParameters()
        self.advanced_btn = QPushButton("Advanced Physical Parameters…")
        self.controls.add_widget(self.advanced_btn)
        run_row = QWidget(); run_layout = QHBoxLayout(run_row); run_layout.setContentsMargins(0,0,0,0)
        self.run_btn = QPushButton("Run SC-MEPLS Simulation"); self.save_history_btn = QPushButton("Save Time-History CSV")
        run_layout.addWidget(self.run_btn); run_layout.addWidget(self.save_history_btn); self.controls.add_widget(run_row)
        self.metrics_table = QTableWidget(0,2); self.metrics_table.setHorizontalHeaderLabels(["Metric","Value"]); self.metrics_table.setMinimumHeight(280); self.controls.add_widget(section_label("Validation / state metrics")); self.controls.add_widget(self.metrics_table)
        self.export_bar = ExportBar("scmepls_rollout_dashboard_600dpi.png"); self.controls.add_widget(self.export_bar)
        splitter = QSplitter(Qt.Orientation.Horizontal); splitter.addWidget(self.controls)
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0); rl.addWidget(QLabel("Roll-out, levitation, interlock, load-transfer and fault dashboard")); rl.addWidget(self.plot_panel,1); splitter.addWidget(right); splitter.setSizes([470,980])
        layout = QHBoxLayout(self); layout.setContentsMargins(6,6,6,6); layout.addWidget(splitter)
        self.run_btn.clicked.connect(self.run_analysis); self.save_history_btn.clicked.connect(self.save_history); self.export_bar.export_requested.connect(self.export_plot)
        self.advanced_btn.clicked.connect(self.edit_advanced_parameters); self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.df = pd.DataFrame(); self.metrics: dict = {}; self.geometry_df = pd.DataFrame(); self.geometry_parameters: RolloutParameters | None = None; self.geometry_report = None
        self.run_analysis()

    def _basic_kwargs(self) -> dict[str, object]:
        return dict(dt_s=self.dt.value(),end_time_s=self.end_time.value(),track_length_m=self.track_length.value(),platform_mass_kg=self.platform_mass.value(),payload_mass_kg=self.payload_mass.value(),initial_gap_m=self.initial_gap.value(),target_gap_m=self.target_gap.value(),cg_shift_x_m=self.cgx.value(),cg_shift_y_m=self.cgy.value(),wind_start_s=self.wind_start.value(),wind_end_s=self.wind_end.value(),coil_fault_time_s=self.coil_fault_time.value(),coil_fault_index=self.coil_fault_index.value(),sensor_fault_start_s=self.sensor_fault_start.value(),sensor_fault_end_s=self.sensor_fault_end.value(),sensor_fault_index=self.sensor_fault_index.value(),leak_time_s=self.leak_time.value(),leak_index=self.leak_index.value())

    def parameters(self) -> RolloutParameters:
        current = {field.name: getattr(self._advanced_values, field.name) for field in fields(RolloutParameters)}
        current.update(self._basic_kwargs())
        return RolloutParameters(**current)

    def edit_advanced_parameters(self) -> None:
        base = self.parameters()
        dialog = _AdvancedParameterDialog(base, self)
        if dialog.exec() != QDialog.DialogCode.Accepted: return
        try:
            values = {field.name: getattr(base, field.name) for field in fields(RolloutParameters)}
            values.update(dialog.parsed(base))
            candidate = RolloutParameters(**values)
            simulate_rollout(candidate)  # validate the complete parameter set before accepting it
            self._advanced_values = candidate
            QMessageBox.information(self, "Advanced parameters", "Advanced physical parameters are valid and have been applied.")
        except Exception as exc:
            QMessageBox.critical(self, "Invalid advanced parameters", str(exc))

    def bind_geometry_coupled_result(self, history: object, parameters: object, report: object = None) -> None:
        if not isinstance(history, pd.DataFrame) or not isinstance(parameters, RolloutParameters):
            return
        self.geometry_df = history.copy(); self.geometry_parameters = parameters; self.geometry_report = report
        self.geometry_status.setText(
            f"Geometry-coupled history available: {len(history):,} samples; integrated/validated structural parameters ready."
        )

    def _source_changed(self) -> None:
        if self.source_combo.currentData() == "geometry" and (self.geometry_parameters is None or self.geometry_df.empty):
            QMessageBox.information(self, "Geometry-coupled dynamics", "Run Structural Validation first. The manual model remains active.")
            self.source_combo.setCurrentIndex(0)
            return
        self.run_analysis()

    def _geometry_metrics(self, df: pd.DataFrame, p: RolloutParameters) -> dict:
        transfer = (df["time_s"] > p.prelock_end_s) & (df["time_s"] < p.hardlock_start_s)
        hard = df["time_s"] > min(p.hardlock_start_s + 2.0, p.end_time_s - p.dt_s)
        return {
            "design_weight_n": (p.platform_mass_kg + p.payload_mass_kg) * 9.81,
            "final_position_error_mm": abs(p.track_length_m - float(df.iloc[-1]["x_m"])) * 1000.0,
            "final_lateral_error_mm": abs(float(df.iloc[-1]["y_m"])) * 1000.0,
            "final_yaw_error_deg": abs(float(df.iloc[-1]["yaw_rad"])) * 180.0 / 3.141592653589793,
            "final_mean_gap_mm": float(df.iloc[-1]["mean_gap_m"]) * 1000.0,
            "maximum_roll_deg": float(df["roll_rad"].abs().max()) * 180.0 / 3.141592653589793,
            "maximum_pitch_deg": float(df["pitch_rad"].abs().max()) * 180.0 / 3.141592653589793,
            "support_force_cv_during_transfer_percent": 100.0 * float(df.loc[transfer,"total_support_force_n"].std()) / max(float(df.loc[transfer,"total_support_force_n"].mean()),1e-9),
            "mean_em_force_after_hard_lock_n": float(df.loc[hard,"em_force_n"].mean()),
            "mean_pneumatic_force_after_hard_lock_n": float(df.loc[hard,"pneumatic_force_n"].mean()),
            "maximum_fault_severity": float(df["fault_severity"].max()),
            "any_unsafe_flag": int(df["unsafe_flag"].max()),
            "any_detected_interlock": int(df.get("detected_interlock",pd.Series([0])).max()),
            "hard_lock_confirmed": int(df.get("hard_lock_confirmed",pd.Series([0])).iloc[-1]),
            "dynamics_source": "validated_geometry",
            "parameters": {field.name:getattr(p,field.name) for field in fields(RolloutParameters)},
        }

    def run_analysis(self) -> None:
        try:
            if self.source_combo.currentData() == "geometry" and self.geometry_parameters is not None and not self.geometry_df.empty:
                self.df = self.geometry_df.copy(); active = self.geometry_parameters; self.metrics = self._geometry_metrics(self.df, active)
            else:
                active = self.parameters(); self.df,self.metrics = simulate_rollout(active); self.metrics["dynamics_source"] = "manual_configured"
            self.plot_panel.set_figure(rollout_dashboard_figure(self.df,float(self.metrics["design_weight_n"])))
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
            params = self.geometry_parameters if self.source_combo.currentData() == "geometry" and self.geometry_parameters is not None else self.parameters()
            out,meta = save_panel_figure(self,self.plot_panel,dpi,path,{"analysis":"SC-MEPLS reduced-order rollout and interlocked force-balanced load transfer","provenance":self.provenance.currentText(),"dynamics_source":self.source_combo.currentData(),"parameters":{field.name:getattr(params,field.name) for field in fields(RolloutParameters)},"metrics":{k:v for k,v in self.metrics.items() if k != "parameters"},"warning":"Reduced-order dynamics; use validated geometry/material/load mapping and independent validation before quantitative design claims."})
            QMessageBox.information(self,"Export complete",f"Saved:\n{out}\n{meta}")
        except Exception as exc: QMessageBox.critical(self,"Export error",str(exc))
