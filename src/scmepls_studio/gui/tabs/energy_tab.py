from __future__ import annotations

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox,QHBoxLayout,QLabel,QMessageBox,QPushButton,QSplitter,QTableWidget,QTableWidgetItem,QVBoxLayout,QWidget

from ...models.energy import EnergyInput,compare_energy
from ...plotting.figures import energy_figure
from ..common import ExportBar,FigureCanvasPanel,ScrollableControls,form_row,make_double,provenance_combo,save_panel_figure,section_label


class _EnergySystem(QWidget):
    def __init__(self,title:str,defaults:dict[str,float],parent:QWidget|None=None)->None:
        super().__init__(parent);layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0);layout.addWidget(section_label(title));self.mass=make_double(defaults.get("mass",1000.0),0.001,1e7,3,10.0);self.distance=make_double(defaults.get("distance",50.0),0.001,1e6,3,1.0);self.traction=make_double(defaults.get("traction",5.0),0.0,1e9,5,0.1);self.levitation=make_double(defaults.get("levitation",0.0),0.0,1e9,5,0.1);self.aux=make_double(defaults.get("aux",1.0),0.0,1e9,5,0.1);self.recovered=make_double(defaults.get("recovered",0.0),0.0,1e9,5,0.1)
        for label,widget in [("Total moving mass (tonnes)",self.mass),("Travel distance (m)",self.distance),("Traction energy (kWh)",self.traction),("Levitation energy (kWh)",self.levitation),("Auxiliary energy (kWh)",self.aux),("Recovered energy (kWh)",self.recovered)]:layout.addWidget(form_row(label,widget))
    def values(self)->EnergyInput:return EnergyInput(mass_tonnes=self.mass.value(),distance_m=self.distance.value(),traction_kwh=self.traction.value(),levitation_kwh=self.levitation.value(),auxiliaries_kwh=self.aux.value(),recovered_kwh=self.recovered.value())


class EnergyTab(QWidget):
    def __init__(self,parent:QWidget|None=None)->None:
        super().__init__(parent);self.controls=ScrollableControls();self.plot_panel=FigureCanvasPanel();self.provenance=provenance_combo("Illustrative");self.plot_mode=QComboBox();self.plot_mode.addItem("Net specific energy consumption","specific");self.plot_mode.addItem("Relative efficiency (best SEC / system SEC)","performance")
        self.controls.add_widget(section_label("Scientific definition"));note=QLabel("Energy performance is computed from gross traction/levitation/auxiliary energy minus recovered energy. The publication metric should remain net specific energy consumption in kWh/(tonne·km). The optional relative index is a transparent best-SEC/system-SEC ratio, not an independent physical quantity.");note.setWordWrap(True);self.controls.add_widget(note);self.controls.add_widget(form_row("Provenance",self.provenance));self.controls.add_widget(form_row("Article plot",self.plot_mode))
        self.crawler=_EnergySystem("Crawler transporter",{"traction":8.5,"aux":1.5,"recovered":0.2});self.rail=_EnergySystem("Rail platform",{"traction":6.5,"aux":1.0,"recovered":0.5});self.maglev=_EnergySystem("Maglev platform",{"traction":3.8,"levitation":1.8,"aux":0.7,"recovered":0.8});self.controls.add_widget(self.crawler);self.controls.add_widget(self.rail);self.controls.add_widget(self.maglev);self.run_btn=QPushButton("Calculate Energy Performance");self.controls.add_widget(self.run_btn)
        self.table=QTableWidget(0,6);self.table.setHorizontalHeaderLabels(["System","Gross energy (kWh)","Net energy (kWh)","Specific energy (kWh/t·km)","Recovery fraction","Relative efficiency"]);self.table.setMinimumHeight(180);self.controls.add_widget(section_label("Computed energy metrics"));self.controls.add_widget(self.table);self.export_bar=ExportBar("energy_performance_600dpi.png");self.controls.add_widget(self.export_bar)
        splitter=QSplitter(Qt.Orientation.Horizontal);splitter.addWidget(self.controls);right=QWidget();rl=QVBoxLayout(right);rl.setContentsMargins(0,0,0,0);rl.addWidget(QLabel("Article-ready energy comparison"));rl.addWidget(self.plot_panel,1);splitter.addWidget(right);splitter.setSizes([430,950]);layout=QHBoxLayout(self);layout.setContentsMargins(6,6,6,6);layout.addWidget(splitter);self.run_btn.clicked.connect(self.run_analysis);self.plot_mode.currentIndexChanged.connect(self.run_analysis);self.export_bar.export_requested.connect(self.export_plot);self.comparison=pd.DataFrame();self.run_analysis()

    def run_analysis(self)->None:
        try:self.comparison=compare_energy({"Crawler Transporter":self.crawler.values(),"Rail Platform":self.rail.values(),"Maglev Platform":self.maglev.values()});self.plot_panel.set_figure(energy_figure(self.comparison,self.plot_mode.currentData()));self._fill_table()
        except Exception as exc:QMessageBox.critical(self,"Energy-analysis error",str(exc))
    def _fill_table(self)->None:
        self.table.setRowCount(len(self.comparison));keys=["system","gross_energy_kwh","net_energy_kwh","specific_energy_kwh_per_tonne_km","recovery_fraction","normalized_energy_performance"]
        for r,row in self.comparison.iterrows():
            for c,key in enumerate(keys):value=row[key];self.table.setItem(r,c,QTableWidgetItem(str(value) if key=="system" else f"{float(value):.6g}"))
        self.table.resizeColumnsToContents()
    def export_plot(self,dpi:int,path:str)->None:
        try:
            out,meta=save_panel_figure(self,self.plot_panel,dpi,path,{"analysis":"Transport energy performance","provenance":self.provenance.currentText(),"plot_mode":self.plot_mode.currentData(),"equations":["E_net = E_traction + E_levitation + E_aux - E_recovered","SEC = E_net/(mass_tonnes*distance_km)","relative_efficiency = best_SEC/system_SEC"],"inputs":{"Crawler":self.crawler.values().__dict__,"Rail":self.rail.values().__dict__,"Maglev":self.maglev.values().__dict__},"metrics":self.comparison.to_dict(orient="records"),"warning":"Use the physical SEC for scientific claims; the relative index is a comparison ratio only."});QMessageBox.information(self,"Export complete",f"Saved:\n{out}\n{meta}")
        except Exception as exc:QMessageBox.critical(self,"Export error",str(exc))
