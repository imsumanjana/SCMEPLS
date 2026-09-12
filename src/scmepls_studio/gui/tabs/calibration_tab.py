from __future__ import annotations

from pathlib import Path

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox,QFileDialog,QHBoxLayout,QLabel,QLineEdit,QMessageBox,QPushButton,QSplitter,QTableWidget,QTableWidgetItem,QVBoxLayout,QWidget

from ...models.calibration import evaluate_external_validation,save_calibration,train_regressor
from ...plotting.figures import feature_importance_figure
from ..common import FigureCanvasPanel,ScrollableControls,form_row,section_label


class CalibrationTab(QWidget):
    def __init__(self,parent:QWidget|None=None)->None:
        super().__init__(parent);self.controls=ScrollableControls();self.plot_panel=FigureCanvasPanel();self.df:pd.DataFrame|None=None;self.result=None;self.external_result=None;self.source_path:Path|None=None
        self.controls.add_widget(section_label("Optional AI/data-driven calibration"));note=QLabel("AI calibration does not validate physics. Choose features explicitly, use grouped/chronological validation where appropriate, and preserve a genuinely untouched external dataset for final claims. Near-perfect feature/target correlations are flagged for possible target leakage.");note.setWordWrap(True);self.controls.add_widget(note)
        self.import_btn=QPushButton("Import Calibration CSV");self.controls.add_widget(self.import_btn);self.target_combo=QComboBox();self.controls.add_widget(form_row("Prediction target",self.target_combo));self.features_edit=QLineEdit();self.features_edit.setPlaceholderText("comma-separated numeric feature columns");self.controls.add_widget(form_row("Input features",self.features_edit));self.algorithm=QComboBox();self.algorithm.addItems(["Random Forest","Gradient Boosting"]);self.controls.add_widget(form_row("Algorithm",self.algorithm));self.validation_mode=QComboBox();self.validation_mode.addItems(["KFold","Chronological"]);self.controls.add_widget(form_row("Validation split",self.validation_mode));self.group_combo=QComboBox();self.group_combo.addItem("None");self.controls.add_widget(form_row("Group/run column",self.group_combo))
        self.train_btn=QPushButton("Train and Cross-Validate");self.external_btn=QPushButton("Evaluate Untouched External CSV…");self.external_btn.setEnabled(False);self.save_btn=QPushButton("Save Calibrated Model (.joblib)");self.controls.add_widget(self.train_btn);self.controls.add_widget(self.external_btn);self.controls.add_widget(self.save_btn)
        self.metrics_table=QTableWidget(0,2);self.metrics_table.setHorizontalHeaderLabels(["Metric","Value"]);self.metrics_table.setMinimumHeight(280);self.controls.add_widget(section_label("Validation metrics"));self.controls.add_widget(self.metrics_table);self.preview=QTableWidget(0,0);self.preview.setMinimumHeight(220);self.controls.add_widget(section_label("Imported data preview"));self.controls.add_widget(self.preview)
        splitter=QSplitter(Qt.Orientation.Horizontal);splitter.addWidget(self.controls);right=QWidget();rl=QVBoxLayout(right);rl.setContentsMargins(0,0,0,0);rl.addWidget(QLabel("Feature importance from the fitted calibration model"));rl.addWidget(self.plot_panel,1);splitter.addWidget(right);splitter.setSizes([540,900]);layout=QHBoxLayout(self);layout.setContentsMargins(6,6,6,6);layout.addWidget(splitter)
        self.import_btn.clicked.connect(self.import_csv);self.train_btn.clicked.connect(self.train);self.external_btn.clicked.connect(self.evaluate_external);self.save_btn.clicked.connect(self.save_model);self.target_combo.currentTextChanged.connect(self._refresh_feature_defaults);self.group_combo.currentTextChanged.connect(self._refresh_feature_defaults)

    def import_csv(self)->None:
        path,_=QFileDialog.getOpenFileName(self,"Import calibration dataset","","CSV files (*.csv)")
        if not path:return
        try:
            self.source_path=Path(path);self.df=pd.read_csv(path);numeric=self.df.select_dtypes(include="number").columns.tolist();self.target_combo.blockSignals(True);self.target_combo.clear();self.target_combo.addItems(numeric);self.target_combo.blockSignals(False);self.group_combo.blockSignals(True);self.group_combo.clear();self.group_combo.addItem("None");self.group_combo.addItems([str(c) for c in self.df.columns]);self.group_combo.blockSignals(False);self._refresh_feature_defaults();self._fill_preview();self.result=None;self.external_result=None;self.external_btn.setEnabled(False)
        except Exception as exc:QMessageBox.critical(self,"Calibration import error",str(exc))

    def _refresh_feature_defaults(self)->None:
        if self.df is None:self.features_edit.clear();return
        target=self.target_combo.currentText();group=self.group_combo.currentText();numeric=self.df.select_dtypes(include="number").columns.tolist();features=[c for c in numeric if c!=target and (group=="None" or c!=group)];self.features_edit.setText(", ".join(features))

    def _selected_features(self)->list[str]:return [name.strip() for name in self.features_edit.text().split(",") if name.strip()]

    def _fill_preview(self)->None:
        assert self.df is not None;view=self.df.head(20);self.preview.setRowCount(len(view));self.preview.setColumnCount(len(view.columns));self.preview.setHorizontalHeaderLabels([str(c) for c in view.columns])
        for r,(_,row) in enumerate(view.iterrows()):
            for c,value in enumerate(row):self.preview.setItem(r,c,QTableWidgetItem(str(value)))
        self.preview.resizeColumnsToContents()

    def _show_metrics(self)->None:
        if self.result is None:return
        metrics=[("Algorithm",self.result.algorithm),("Target",self.result.target),("Features",", ".join(self.result.feature_names)),("Validation mode",self.result.validation_mode),("Group column",self.result.group_column or "None"),("Training samples",self.result.sample_count),("Validation predictions",self.result.validation_sample_count),("Cross-validated R²",self.result.r2),("Cross-validated MAE",self.result.mae),("Cross-validated RMSE",self.result.rmse),("Leakage warnings","; ".join(self.result.leakage_warnings) if self.result.leakage_warnings else "None"),("Training data SHA-256",self.result.data_fingerprint_sha256)]
        if self.external_result is not None:metrics.extend([("External samples",self.external_result.sample_count),("External R²",self.external_result.r2),("External MAE",self.external_result.mae),("External RMSE",self.external_result.rmse),("External residual 5–95%",f"{self.external_result.residual_p05:.6g} to {self.external_result.residual_p95:.6g}"),("External data SHA-256",self.external_result.dataset_fingerprint_sha256)])
        self.metrics_table.setRowCount(len(metrics))
        for r,(key,value) in enumerate(metrics):self.metrics_table.setItem(r,0,QTableWidgetItem(str(key)));self.metrics_table.setItem(r,1,QTableWidgetItem(f"{value:.6g}" if isinstance(value,float) else str(value)))
        self.metrics_table.resizeColumnsToContents()

    def train(self)->None:
        if self.df is None or not self.target_combo.currentText():QMessageBox.warning(self,"No data","Import a numeric calibration dataset first.");return
        try:
            group=self.group_combo.currentText();group_column=None if group=="None" else group;self.result=train_regressor(self.df,self.target_combo.currentText(),self.algorithm.currentText(),feature_names=self._selected_features(),validation_mode=self.validation_mode.currentText(),group_column=group_column);self.external_result=None;self._show_metrics();self.plot_panel.set_figure(feature_importance_figure(self.result.feature_importance));self.external_btn.setEnabled(True)
            if self.result.leakage_warnings:QMessageBox.warning(self,"Possible feature leakage","\n".join(self.result.leakage_warnings))
            if self.result.r2<0.5:QMessageBox.warning(self,"Weak model","Cross-validated R² is below 0.5. Do not use this model for quantitative claims.")
        except Exception as exc:QMessageBox.critical(self,"Calibration error",str(exc))

    def evaluate_external(self)->None:
        if self.result is None:return
        path,_=QFileDialog.getOpenFileName(self,"Select untouched external validation dataset","","CSV files (*.csv)")
        if not path:return
        try:self.external_result=evaluate_external_validation(self.result,pd.read_csv(path));self._show_metrics();QMessageBox.information(self,"External validation complete",f"External R² = {self.external_result.r2:.4g}\nExternal RMSE = {self.external_result.rmse:.4g}")
        except Exception as exc:QMessageBox.critical(self,"External validation error",str(exc))

    def save_model(self)->None:
        if self.result is None:QMessageBox.warning(self,"No model","Train a model first.");return
        path,_=QFileDialog.getSaveFileName(self,"Save calibrated model","scmepls_calibration.joblib","Joblib files (*.joblib)")
        if path:
            try:save_calibration(self.result,path);QMessageBox.information(self,"Model saved",path)
            except Exception as exc:QMessageBox.critical(self,"Save error",str(exc))
