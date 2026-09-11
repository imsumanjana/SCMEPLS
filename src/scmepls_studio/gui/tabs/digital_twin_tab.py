from __future__ import annotations

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
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

from ...geometry import GeometryAsset, GeometryImportError, GeometryPart, load_geometry
from ..common import ScrollableControls, form_row, section_label


class DigitalTwinTab(QWidget):
    """External GLB/STL geometry viewer, deliberately decoupled from model physics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.asset: GeometryAsset | None = None
        self.actors: dict[str, object] = {}
        self.datasets: dict[int, str] = {}

        self.controls = ScrollableControls()
        self.controls.add_widget(section_label("External 3D geometry"))
        note = QLabel(
            "Import a .glb or .stl model for digital-twin visualization. Imported geometry is display-only: "
            "it does not change mass, inertia, actuator positions, controller gains, or SC-MEPLS simulation results."
        )
        note.setWordWrap(True)
        self.controls.add_widget(note)

        self.unit_combo = QComboBox()
        self.unit_combo.addItems(["m", "mm", "cm"])
        self.controls.add_widget(form_row("Source geometry unit", self.unit_combo))
        self.axis_combo = QComboBox()
        self.axis_combo.addItem("Auto: GLB Y-up → Z-up; STL as stored", "auto")
        self.axis_combo.addItem("As stored", "as_stored")
        self.axis_combo.addItem("Force GLB Y-up → SC-MEPLS Z-up", "gltf_y_up_to_z_up")
        self.controls.add_widget(form_row("Axis conversion", self.axis_combo))

        self.import_btn = QPushButton("Import GLB / STL…")
        self.clear_btn = QPushButton("Clear Geometry")
        self.controls.add_widget(self.import_btn)
        self.controls.add_widget(self.clear_btn)

        self.file_label = QLabel("No geometry loaded")
        self.file_label.setWordWrap(True)
        self.controls.add_widget(self.file_label)

        self.controls.add_widget(section_label("Component view"))
        self.component_combo = QComboBox()
        self.component_combo.setEnabled(False)
        self.controls.add_widget(form_row("Selected component", self.component_combo))
        self.isolate_btn = QPushButton("Isolate Selected")
        self.show_all_btn = QPushButton("Full Assembly")
        self.isolate_btn.setEnabled(False)
        self.show_all_btn.setEnabled(False)
        self.controls.add_widget(self.isolate_btn)
        self.controls.add_widget(self.show_all_btn)

        self.controls.add_widget(section_label("Camera"))
        self.iso_btn = QPushButton("Isometric")
        self.front_btn = QPushButton("Front")
        self.side_btn = QPushButton("Side")
        self.top_btn = QPushButton("Top")
        self.fit_btn = QPushButton("Fit View")
        for button in (self.iso_btn, self.front_btn, self.side_btn, self.top_btn, self.fit_btn):
            self.controls.add_widget(button)

        self.controls.add_widget(section_label("Geometry validation"))
        self.info_table = QTableWidget(0, 2)
        self.info_table.setHorizontalHeaderLabels(["Property", "Value"])
        self.info_table.setMinimumHeight(260)
        self.controls.add_widget(self.info_table)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("3D Digital Twin — external geometry viewer"))
        self.plotter = QtInteractor(right)
        right_layout.addWidget(self.plotter.interactor, 1)
        self.plotter.set_background("white")
        self.plotter.add_axes()
        self.plotter.enable_mesh_picking(
            callback=self._mesh_picked,
            show=False,
            show_message=False,
            left_clicking=True,
        )

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.controls)
        splitter.addWidget(right)
        splitter.setSizes([410, 1050])
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(splitter)

        self.import_btn.clicked.connect(self.import_geometry)
        self.clear_btn.clicked.connect(self.clear_geometry)
        self.component_combo.currentTextChanged.connect(self._selection_changed)
        self.isolate_btn.clicked.connect(self.isolate_selected)
        self.show_all_btn.clicked.connect(self.show_full_assembly)
        self.iso_btn.clicked.connect(self.plotter.view_isometric)
        self.front_btn.clicked.connect(self.plotter.view_xz)
        self.side_btn.clicked.connect(self.plotter.view_yz)
        self.top_btn.clicked.connect(self.plotter.view_xy)
        self.fit_btn.clicked.connect(self._fit_view)

    @staticmethod
    def _polydata(part: GeometryPart) -> pv.PolyData:
        vtk_faces = np.column_stack((np.full(part.n_faces, 3, dtype=np.int64), part.faces)).ravel()
        poly = pv.PolyData(part.vertices_m, vtk_faces)
        if part.cell_rgb is not None and len(part.cell_rgb) == part.n_faces:
            poly.cell_data["face_rgb"] = part.cell_rgb
        return poly

    def import_geometry(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import external 3D geometry",
            "",
            "3D geometry (*.glb *.stl);;GLB files (*.glb);;STL files (*.stl)",
        )
        if not path:
            return
        try:
            asset = load_geometry(
                path,
                source_unit=self.unit_combo.currentText(),
                axis_mode=str(self.axis_combo.currentData()),
            )
            self._display_asset(asset)
        except GeometryImportError as exc:
            QMessageBox.critical(self, "Geometry import failed", str(exc))
        except Exception as exc:
            QMessageBox.critical(self, "3D viewer error", str(exc))

    def _display_asset(self, asset: GeometryAsset) -> None:
        self.asset = asset
        self.plotter.clear()
        self.plotter.set_background("white")
        self.plotter.add_axes()
        self.actors.clear()
        self.datasets.clear()

        for part in asset.parts:
            poly = self._polydata(part)
            kwargs: dict = {
                "name": part.component_id,
                "smooth_shading": True,
                "show_edges": False,
                "pickable": True,
            }
            if "face_rgb" in poly.cell_data:
                kwargs.update({"scalars": "face_rgb", "rgb": True, "show_scalar_bar": False})
            else:
                kwargs["color"] = "lightgray"
            actor = self.plotter.add_mesh(poly, **kwargs)
            self.actors[part.component_id] = actor
            self.datasets[id(poly)] = part.component_id

        self.component_combo.blockSignals(True)
        self.component_combo.clear()
        self.component_combo.addItems([part.component_id for part in asset.parts])
        self.component_combo.blockSignals(False)
        self.component_combo.setEnabled(True)
        self.isolate_btn.setEnabled(True)
        self.show_all_btn.setEnabled(True)
        self.file_label.setText(str(asset.source_path))
        self._update_info()
        self.show_full_assembly()
        self._fit_view()
        if self.component_combo.count():
            self._highlight(self.component_combo.currentText())

        if asset.warnings:
            QMessageBox.warning(self, "Geometry imported with warnings", "\n".join(asset.warnings))

    def clear_geometry(self) -> None:
        self.asset = None
        self.actors.clear()
        self.datasets.clear()
        self.component_combo.clear()
        self.component_combo.setEnabled(False)
        self.isolate_btn.setEnabled(False)
        self.show_all_btn.setEnabled(False)
        self.file_label.setText("No geometry loaded")
        self.info_table.setRowCount(0)
        self.plotter.clear()
        self.plotter.set_background("white")
        self.plotter.add_axes()
        self.plotter.render()

    def _update_info(self) -> None:
        if self.asset is None:
            self.info_table.setRowCount(0)
            return
        dims = self.asset.dimensions_m
        rows = [
            ("Format", self.asset.source_path.suffix.lower()),
            ("Source unit", self.asset.source_unit),
            ("Axis mode", self.asset.axis_mode),
            ("Scale to metre", f"{self.asset.scale_to_m:g}"),
            ("Components", str(len(self.asset.parts))),
            ("Triangles", f"{self.asset.triangle_count:,}"),
            ("Size X (m)", f"{dims[0]:.6g}"),
            ("Size Y (m)", f"{dims[1]:.6g}"),
            ("Size Z (m)", f"{dims[2]:.6g}"),
            ("Warnings", str(len(self.asset.warnings))),
        ]
        self.info_table.setRowCount(len(rows))
        for row, (key, value) in enumerate(rows):
            self.info_table.setItem(row, 0, QTableWidgetItem(key))
            self.info_table.setItem(row, 1, QTableWidgetItem(value))
        self.info_table.resizeColumnsToContents()

    def _mesh_picked(self, mesh: pv.DataSet) -> None:
        component_id = self.datasets.get(id(mesh))
        if component_id is None:
            return
        index = self.component_combo.findText(component_id)
        if index >= 0:
            self.component_combo.setCurrentIndex(index)

    def _selection_changed(self, component_id: str) -> None:
        if component_id:
            self._highlight(component_id)

    def _highlight(self, component_id: str) -> None:
        for name, actor in self.actors.items():
            try:
                actor.prop.show_edges = name == component_id
                actor.prop.line_width = 2.5 if name == component_id else 1.0
                if name == component_id:
                    actor.prop.edge_color = "black"
            except Exception:
                pass
        self.plotter.render()

    def isolate_selected(self) -> None:
        selected = self.component_combo.currentText()
        if not selected:
            return
        for name, actor in self.actors.items():
            actor.SetVisibility(name == selected)
        self._highlight(selected)
        self.plotter.reset_camera()
        self.plotter.render()

    def show_full_assembly(self) -> None:
        for actor in self.actors.values():
            actor.SetVisibility(True)
        selected = self.component_combo.currentText()
        if selected:
            self._highlight(selected)
        self.plotter.render()

    def _fit_view(self) -> None:
        if self.actors:
            self.plotter.reset_camera()
        self.plotter.render()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            self.plotter.close()
        finally:
            super().closeEvent(event)
