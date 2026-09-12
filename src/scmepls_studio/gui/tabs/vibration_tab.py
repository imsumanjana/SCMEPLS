from __future__ import annotations

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import QFileDialog,QHBoxLayout,QInputDialog,QLabel,QMessageBox,QPushButton,QSplitter,QTableWidget,QTableWidgetItem,QVBoxLayout,QWidget,QComboBox
from PyQt6.QtCore import Qt

from ...models.vibration import generate_synthetic_timeseries, metrics_from_dataframe
from ...plotting.figures import vibration_figure, vibration_timeseries_figure
from ..common import ExportBar,FigureCanvasPanel,ScrollableControls,form_row,make_double,make_int,provenance_combo,save_panel_figure,section_label


class VibrationTab(QWidget):
    def __init__(self,parent: QWidget|None=None)->None:
        super().__init__(parent)
        self.dataframe:pd.DataFrame|None=None; self.metrics:pd.DataFrame|None=None
        self.controls=ScrollableControls(); self.plot_panel=FigureCanvasPanel(); self.series_panel=FigureCanvasPanel(); self.provenance=provenance_combo("Illustrative")
        self.controls.add_widget(section_label("Data provenance and preprocessing")); self.controls.add_widget(form_row("Provenance",self.provenance))
        note=QLabel("Select preprocessing explicitly. Raw acceleration includes DC/gravity projection; remove-mean/detrend/high-pass options are applied consistently before RMS, peak, crest factor and Welch PSD.");note.setWordWrap(True);self.controls.add_widget(note)
        self.preprocessing=QComboBox(); self.preprocessing.addItem("Raw / no correction","raw"); self.preprocessing.addItem("Remove mean (recommended for vibration)","remove_mean"); self.preprocessing.addItem("Linear detrend","detrend"); self.preprocessing.addItem("Zero-phase high-pass","highpass"); self.preprocessing.setCurrentIndex(1)
        self.highpass_cutoff=make_double(0.5,0.001,1000.0,3,0.1)
        self.controls.add_widget(form_row("Preprocessing",self.preprocessing)); self.controls.add_widget(form_row("High-pass cutoff (Hz)",self.highpass_cutoff))
        self.controls.add_widget(section_label("Illustrative time-series generator"))
        self.duration=make_double(20.0,1.0,600.0,2,1.0); self.sample_rate=make_double(200.0,10.0,5000.0,1,10.0); self.crawler_rms=make_double(0.90,0.0001,100.0,4,0.05); self.rail_rms=make_double(0.60,0.0001,100.0,4,0.05); self.maglev_rms=make_double(0.10,0.0001,100.0,4,0.02); self.seed=make_int(42,0,999999)
        for name,widget in [("Duration (s)",self.duration),("Sample rate (Hz)",self.sample_rate),("Crawler target RMS (m/s²)",self.crawler_rms),("Rail target RMS (m/s²)",self.rail_rms),("Maglev target RMS (m/s²)",self.maglev_rms),("Random seed",self.seed)]: self.controls.add_widget(form_row(name,widget))
        btns=QWidget();bl=QVBoxLayout(btns);bl.setContentsMargins(0,0,0,0);self.generate_btn=QPushButton("Generate Illustrative Data");self.import_btn=QPushButton("Import Experimental/Simulation CSV");self.export_csv_btn=QPushButton("Save Metrics CSV");bl.addWidget(self.generate_btn);bl.addWidget(self.import_btn);bl.addWidget(self.export_csv_btn);self.controls.add_widget(btns)
        self.table=QTableWidget(0,7);self.table.setHorizontalHeaderLabels(["System","RMS","Normalized RMS","Peak","Peak-to-peak","Crest factor","Dominant frequency (Hz)"]);self.table.horizontalHeader().setStretchLastSection(True);self.table.setMinimumHeight(190);self.controls.add_widget(section_label("Computed metrics"));self.controls.add_widget(self.table)
        self.export_bar=ExportBar("vibration_comparison_600dpi.png");self.controls.add_widget(self.export_bar)
        right=QWidget();rl=QVBoxLayout(right);rl.setContentsMargins(0,0,0,0);rl.addWidget(QLabel("Article comparison plot"));rl.addWidget(self.plot_panel,1);rl.addWidget(QLabel("Acceleration time histories"));rl.addWidget(self.series_panel,1)
        splitter=QSplitter(Qt.Orientation.Horizontal);splitter.addWidget(self.controls);splitter.addWidget(right);splitter.setSizes([420,980]);layout=QHBoxLayout(self);layout.setContentsMargins(6,6,6,6);layout.addWidget(splitter)
        self.generate_btn.clicked.connect(self.generate);self.import_btn.clicked.connect(self.import_csv);self.export_csv_btn.clicked.connect(self.save_metrics);self.export_bar.export_requested.connect(self.export_plot);self.preprocessing.currentIndexChanged.connect(lambda _:self._reprocess());self.highpass_cutoff.valueChanged.connect(lambda _:self._reprocess())
        self.generate()

    def _reprocess(self)->None:
        if self.dataframe is None:return
        try:self._process(imported=self.provenance.currentText()!="Illustrative")
        except Exception as exc:QMessageBox.warning(self,"Vibration preprocessing",str(exc))

    def generate(self)->None:
        try:
            self.dataframe=generate_synthetic_timeseries(self.duration.value(),self.sample_rate.value(),{"Crawler Transporter":self.crawler_rms.value(),"Rail Platform":self.rail_rms.value(),"Maglev Platform":self.maglev_rms.value()},seed=self.seed.value());self.provenance.setCurrentText("Illustrative");self._process(imported=False)
        except Exception as exc:QMessageBox.critical(self,"Vibration analysis error",str(exc))

    def import_csv(self)->None:
        path,_=QFileDialog.getOpenFileName(self,"Import vibration time series","","CSV files (*.csv)")
        if not path:return
        provenance,accepted=QInputDialog.getItem(self,"Imported data provenance","Select the provenance of this CSV:",["Experimental","Simulation","Literature-derived"],0,False)
        if not accepted:return
        try:self.dataframe=pd.read_csv(path);self.provenance.setCurrentText(str(provenance));self._process(imported=True)
        except Exception as exc:QMessageBox.critical(self,"CSV import error",str(exc))

    def _process(self,imported:bool)->None:
        assert self.dataframe is not None
        self.metrics=metrics_from_dataframe(self.dataframe,str(self.preprocessing.currentData()),highpass_cutoff_hz=self.highpass_cutoff.value());self.metrics["uncertainty"]=np.nan if imported else 0.0;self._fill_table();self.plot_panel.set_figure(vibration_figure(self.metrics));self.series_panel.set_figure(vibration_timeseries_figure(self.dataframe))

    def _fill_table(self)->None:
        assert self.metrics is not None;self.table.setRowCount(len(self.metrics));columns=["system","rms","normalized_rms","peak","peak_to_peak","crest_factor","dominant_frequency_hz"]
        for r,row in self.metrics.iterrows():
            for c,key in enumerate(columns):
                value=row[key];self.table.setItem(r,c,QTableWidgetItem(str(value) if key=="system" else f"{float(value):.5g}"))
        self.table.resizeColumnsToContents()

    def save_metrics(self)->None:
        if self.metrics is None:return
        path,_=QFileDialog.getSaveFileName(self,"Save vibration metrics","vibration_metrics.csv","CSV files (*.csv)")
        if path:self.metrics.to_csv(path,index=False)

    def export_plot(self,dpi:int,path:str)->None:
        try:
            out,meta=save_panel_figure(self,self.plot_panel,dpi,path,{"analysis":"Normalized RMS vibration comparison","provenance":self.provenance.currentText(),"preprocessing":str(self.preprocessing.currentData()),"highpass_cutoff_hz":self.highpass_cutoff.value() if self.preprocessing.currentData()=="highpass" else None,"equation":"a_RMS = sqrt(mean(a_processed(t)^2)); normalized by maximum/reference RMS","uncertainty_status":"Not supplied for imported data" if self.metrics is not None and self.metrics["uncertainty"].isna().all() else "Configured in metrics table","warning":"Illustrative values are not direct experimental measurements." if self.provenance.currentText()=="Illustrative" else ""});QMessageBox.information(self,"Export complete",f"Saved:\n{out}\n{meta}")
        except Exception as exc:QMessageBox.critical(self,"Export error",str(exc))
