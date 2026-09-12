from __future__ import annotations

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox,QHBoxLayout,QLabel,QMessageBox,QPushButton,QSplitter,QTableWidget,QTableWidgetItem,QVBoxLayout,QWidget

from ...models.control_response import ResponseParameters,calculate_response_metrics,simulate_step
from ...plotting.figures import step_response_figure
from ..common import ExportBar,FigureCanvasPanel,ScrollableControls,form_row,make_double,provenance_combo,save_panel_figure,section_label


class _SystemControls(QWidget):
    def __init__(self,title:str,default_type:str,defaults:dict[str,float],parent:QWidget|None=None)->None:
        super().__init__(parent);layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0);layout.addWidget(section_label(title))
        self.model_type=QComboBox();self.model_type.addItem("First-order equivalent","first_order");self.model_type.addItem("Second-order equivalent","second_order");self.model_type.addItem("Mass–damper plant with filtered PID","mass_damper_pid");idx=self.model_type.findData(default_type);self.model_type.setCurrentIndex(max(0,idx));layout.addWidget(form_row("Model",self.model_type))
        self.gain=make_double(defaults.get("gain",1.0),0.001,10.0,4,0.05);self.tau=make_double(defaults.get("tau",1.0),0.001,100.0,4,0.05);self.wn=make_double(defaults.get("wn",2.0),0.001,100.0,4,0.1);self.zeta=make_double(defaults.get("zeta",0.9),0.01,5.0,4,0.05)
        self.mass=make_double(defaults.get("mass",75.0),0.01,1e7,3,1.0);self.damping=make_double(defaults.get("damping",150.0),0.0,1e8,3,10.0);self.actuator_gain=make_double(defaults.get("actuator_gain",1.0),0.001,1e6,4,0.1);self.kp=make_double(defaults.get("kp",90.0),0.0,1e7,3,5.0);self.ki=make_double(defaults.get("ki",15.0),0.0,1e7,3,1.0);self.kd=make_double(defaults.get("kd",60.0),0.0,1e7,3,5.0);self.derivative_filter=make_double(defaults.get("derivative_filter",100.0),0.01,1e6,3,5.0)
        definitions=[("gain","DC gain",self.gain),("tau","Time constant τ (s)",self.tau),("wn","Natural frequency ωn (rad/s)",self.wn),("zeta","Damping ratio ζ",self.zeta),("mass","Moving mass (kg)",self.mass),("damping","Viscous damping (N·s/m)",self.damping),("actuator_gain","Actuator gain",self.actuator_gain),("kp","Kp",self.kp),("ki","Ki",self.ki),("kd","Kd",self.kd),("derivative_filter","Derivative filter N (rad/s)",self.derivative_filter)]
        self.rows={}
        for key,label,widget in definitions:
            row=form_row(label,widget);layout.addWidget(row);self.rows[key]=row
        self.limit_note=QLabel("Linear comparison model only: derivative roll-off is included, but actuator saturation/rate limits and nonlinear switching are not. Use the SC-MEPLS dynamics for those design questions.");self.limit_note.setWordWrap(True);layout.addWidget(self.limit_note)
        self.model_type.currentIndexChanged.connect(self._update_visibility);self._update_visibility()

    def _update_visibility(self)->None:
        model=str(self.model_type.currentData());visible={"first_order":{"gain","tau"},"second_order":{"gain","wn","zeta"},"mass_damper_pid":{"mass","damping","actuator_gain","kp","ki","kd","derivative_filter"}}[model]
        for key,row in self.rows.items():row.setVisible(key in visible)
        self.limit_note.setVisible(model=="mass_damper_pid")

    def parameters(self)->ResponseParameters:
        return ResponseParameters(model_type=self.model_type.currentData(),gain=self.gain.value(),time_constant_s=self.tau.value(),natural_frequency_rad_s=self.wn.value(),damping_ratio=self.zeta.value(),mass_kg=self.mass.value(),damping_ns_m=self.damping.value(),actuator_gain=self.actuator_gain.value(),kp=self.kp.value(),ki=self.ki.value(),kd=self.kd.value(),derivative_filter_rad_s=self.derivative_filter.value())


class ControlResponseTab(QWidget):
    def __init__(self,parent:QWidget|None=None)->None:
        super().__init__(parent);self.controls=ScrollableControls();self.plot_panel=FigureCanvasPanel();self.provenance=provenance_combo("Simulation");self.duration=make_double(10.0,1.0,200.0,2,1.0);self.points=make_double(2000,100,200000,0,100)
        self.controls.add_widget(section_label("Analysis definition"));note=QLabel("Equivalent transfer-function comparison. Only parameters belonging to the selected model are shown. Filtered PID derivative is used; actuator saturation/rate limits require the nonlinear SC-MEPLS model.");note.setWordWrap(True);self.controls.add_widget(note);self.controls.add_widget(form_row("Provenance",self.provenance));self.controls.add_widget(form_row("Simulation duration (s)",self.duration));self.controls.add_widget(form_row("Time samples",self.points))
        self.maglev=_SystemControls("Maglev closed-loop model","first_order",{"tau":0.80});self.conventional=_SystemControls("Conventional transport model","first_order",{"tau":2.80});self.controls.add_widget(self.maglev);self.controls.add_widget(self.conventional)
        self.run_btn=QPushButton("Run Step-Response Analysis");self.controls.add_widget(self.run_btn);self.table=QTableWidget(0,8);self.table.setHorizontalHeaderLabels(["System","Rise time (s)","Settling time (s)","Overshoot (%)","Final value","Steady-state error","IAE","ITAE"]);self.table.setMinimumHeight(180);self.controls.add_widget(section_label("Response metrics"));self.controls.add_widget(self.table);self.export_bar=ExportBar("closed_loop_step_response_600dpi.png");self.controls.add_widget(self.export_bar)
        splitter=QSplitter(Qt.Orientation.Horizontal);splitter.addWidget(self.controls);right=QWidget();rl=QVBoxLayout(right);rl.setContentsMargins(0,0,0,0);rl.addWidget(QLabel("Closed-loop tracking comparison"));rl.addWidget(self.plot_panel,1);splitter.addWidget(right);splitter.setSizes([430,950]);layout=QHBoxLayout(self);layout.setContentsMargins(6,6,6,6);layout.addWidget(splitter)
        self.run_btn.clicked.connect(self.run_analysis);self.export_bar.export_requested.connect(self.export_plot);self.responses:dict[str,tuple]={};self.metrics_df=pd.DataFrame();self.run_analysis()

    def run_analysis(self)->None:
        try:
            systems={"Maglev System":self.maglev.parameters(),"Conventional System":self.conventional.parameters()};rows=[];self.responses={}
            for name,params in systems.items():
                t,y=simulate_step(params,self.duration.value(),int(self.points.value()));self.responses[name]=(t,y);m=calculate_response_metrics(t,y);rows.append({"system":name,**m.__dict__})
            self.metrics_df=pd.DataFrame(rows);self.plot_panel.set_figure(step_response_figure(self.responses));self._fill_table()
        except Exception as exc:QMessageBox.critical(self,"Control-response error",str(exc))

    def _fill_table(self)->None:
        self.table.setRowCount(len(self.metrics_df));keys=["system","rise_time_s","settling_time_s","overshoot_percent","steady_state_value","steady_state_error","iae","itae"]
        for r,row in self.metrics_df.iterrows():
            for c,key in enumerate(keys):
                value=row[key];text=str(value) if key=="system" else ("nan" if pd.isna(value) else f"{float(value):.5g}");self.table.setItem(r,c,QTableWidgetItem(text))
        self.table.resizeColumnsToContents()

    def export_plot(self,dpi:int,path:str)->None:
        try:
            out,meta=save_panel_figure(self,self.plot_panel,dpi,path,{"analysis":"Closed-loop step response","provenance":self.provenance.currentText(),"models":{"Maglev System":self.maglev.parameters().__dict__,"Conventional System":self.conventional.parameters().__dict__},"metrics":self.metrics_df.to_dict(orient="records"),"warning":"Equivalent/reduced-order linear models require identified plant validation. Filtered derivative is included; actuator saturation/rate limits are not represented here."});QMessageBox.information(self,"Export complete",f"Saved:\n{out}\n{meta}")
        except Exception as exc:QMessageBox.critical(self,"Export error",str(exc))
