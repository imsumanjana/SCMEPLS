from __future__ import annotations

import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
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

from ...models.mcda import Criterion, weighted_scores
from ...plotting.figures import radar_figure
from ..common import ExportBar, FigureCanvasPanel, ScrollableControls, provenance_combo, save_panel_figure, section_label


class RadarTab(QWidget):
    CRITERIA = [
        "Vibration reduction",
        "Positioning accuracy",
        "Maintainability",
        "Energy performance",
        "Cost advantage",
        "Low complexity",
        "Load capability",
        "Fail-safe robustness",
        "Environmental robustness",
    ]
    SYSTEMS = ["Crawler Transporter", "Rail Platform", "Maglev Platform"]

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.controls = ScrollableControls(); self.plot_panel = FigureCanvasPanel()
        self.provenance = provenance_combo("Illustrative")
        self.controls.add_widget(section_label("Multi-criteria comparison"))
        note = QLabel(
            "All criteria are written so that a higher score is better. Scores use a 1–5 scale and must be supported by "
            "traceable measurements, simulation outputs, or cited literature before being interpreted quantitatively."
        ); note.setWordWrap(True); self.controls.add_widget(note)
        self.controls.add_widget(self.provenance)
        self.table = QTableWidget(len(self.CRITERIA), 5)
        self.table.setHorizontalHeaderLabels(["Criterion", "Weight", *self.SYSTEMS])
        defaults = {
            "Vibration reduction": [2.0, 3.0, 5.0],
            "Positioning accuracy": [2.0, 3.0, 5.0],
            "Maintainability": [2.0, 3.0, 4.0],
            "Energy performance": [2.5, 3.5, 4.2],
            "Cost advantage": [4.0, 3.5, 2.0],
            "Low complexity": [4.0, 3.5, 2.0],
            "Load capability": [5.0, 4.5, 3.5],
            "Fail-safe robustness": [5.0, 4.5, 3.0],
            "Environmental robustness": [4.5, 4.0, 2.8],
        }
        for r, criterion in enumerate(self.CRITERIA):
            name_item = QTableWidgetItem(criterion); name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(r, 0, name_item); self.table.setItem(r, 1, QTableWidgetItem("1.0"))
            for c, value in enumerate(defaults[criterion], start=2):
                self.table.setItem(r, c, QTableWidgetItem(f"{value:.2f}"))
        self.table.setMinimumHeight(360); self.table.resizeColumnsToContents(); self.controls.add_widget(self.table)
        self.run_btn = QPushButton("Update Radar and Weighted Ranking"); self.controls.add_widget(self.run_btn)
        self.ranking = QTableWidget(0, 2); self.ranking.setHorizontalHeaderLabels(["System", "Weighted score"]); self.ranking.setMinimumHeight(120)
        self.controls.add_widget(section_label("Weighted ranking")); self.controls.add_widget(self.ranking)
        self.export_bar = ExportBar("multicriteria_radar_600dpi.png"); self.controls.add_widget(self.export_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal); splitter.addWidget(self.controls)
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(0,0,0,0); rl.addWidget(QLabel("Article-ready radar chart")); rl.addWidget(self.plot_panel, 1)
        splitter.addWidget(right); splitter.setSizes([500, 900])
        layout = QHBoxLayout(self); layout.setContentsMargins(6,6,6,6); layout.addWidget(splitter)
        self.run_btn.clicked.connect(self.run_analysis); self.export_bar.export_requested.connect(self.export_plot)
        self.scores = pd.DataFrame(); self.ranked = pd.DataFrame(); self.run_analysis()

    def _read_table(self) -> tuple[pd.DataFrame, list[Criterion]]:
        criteria: list[Criterion] = []
        rows = {system: {"system": system} for system in self.SYSTEMS}
        for r, criterion_name in enumerate(self.CRITERIA):
            try:
                weight = float(self.table.item(r, 1).text())
                criteria.append(Criterion(criterion_name, weight, True))
                for c, system in enumerate(self.SYSTEMS, start=2):
                    value = float(self.table.item(r, c).text())
                    if not 1.0 <= value <= 5.0:
                        raise ValueError(f"{criterion_name}: scores must be between 1 and 5.")
                    rows[system][criterion_name] = value
            except (AttributeError, ValueError) as exc:
                raise ValueError(f"Invalid value in row '{criterion_name}': {exc}") from exc
        return pd.DataFrame(rows.values()), criteria

    def run_analysis(self) -> None:
        try:
            self.scores, criteria = self._read_table()
            self.ranked = weighted_scores(self.scores, criteria)
            self.plot_panel.set_figure(radar_figure(self.scores, self.CRITERIA))
            self.ranking.setRowCount(len(self.ranked))
            for r, row in self.ranked.iterrows():
                self.ranking.setItem(r, 0, QTableWidgetItem(str(row["system"])))
                self.ranking.setItem(r, 1, QTableWidgetItem(f"{float(row['weighted_score']):.4f}"))
            self.ranking.resizeColumnsToContents()
        except Exception as exc:
            QMessageBox.critical(self, "Radar-analysis error", str(exc))

    def export_plot(self, dpi: int, path: str) -> None:
        try:
            out, meta = save_panel_figure(self, self.plot_panel, dpi, path, {
                "analysis": "Multi-criteria radar comparison",
                "provenance": self.provenance.currentText(),
                "scale": "1 to 5; higher is better for every criterion",
                "scores": self.scores.to_dict(orient="records"),
                "ranking": self.ranked[["system", "weighted_score"]].to_dict(orient="records"),
                "warning": "Expert scores must be supported by traceable evidence before quantitative publication claims.",
            })
            QMessageBox.information(self, "Export complete", f"Saved:\n{out}\n{meta}")
        except Exception as exc:
            QMessageBox.critical(self, "Export error", str(exc))
