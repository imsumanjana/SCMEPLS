from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
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

from ...geometry import (
    GeometryAsset,
    GeometryImportError,
    GeometryManifest,
    GeometryManifestError,
    GeometryPart,
    load_geometry,
    load_geometry_manifest,
    manifest_path_for_geometry,
)
from ..common import ScrollableControls, form_row, section_label


class _GeometryLoadWorker(QObject):
    loaded = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, path: str, source_unit: str, axis_mode: str) -> None:
        super().__init__()
        self.path = path
        self.source_unit = source_unit
        self.axis_mode = axis_mode

    @pyqtSlot()
    def run(self) -> None:
        try:
            self.loaded.emit(load_geometry(self.path, self.source_unit, self.axis_mode))
        except Exception as exc:
            self.failed.emit(str(exc))


class DigitalTwinTab(QWidget):
    """External GLB/STL geometry viewer, deliberately decoupled from model physics."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.asset: GeometryAsset | None = None
        self.manifest: GeometryManifest | None = None
        self.manifest_error: str | None = None
        self.actors: dict[str, object] = {}
        self.component_by_index: dict[int, str] = {}
        self._load_thread: QThread | None = None
        self._load_worker: _GeometryLoadWorker | None = None

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
        self.manifest_btn = QPushButton("Load / Validate Manifest…")
        self.screenshot_btn = QPushButton("Export 3D Screenshot…")
        self.screenshot_btn.setEnabled(False)
        self.manifest_btn.setEnabled(False)
        self.controls.add_widget(self.import_btn)
        self.controls.add_widget(self.clear_btn)
        self.controls.add_widget(self.manifest_btn)
        self.controls.add_widget(self.screenshot_btn)

        self.file_label = QLabel("No geometry loaded")
        self.file_label.setWordWrap(True)
        self.controls.add_widget(self.file_label)
        self.manifest_label = QLabel("Manifest: not loaded")
        self.manifest_label.setWordWrap(True)
        self.controls.add_widget(self.manifest_label)

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
        self.info_table.setMinimumHeight(300)
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
        splitter.setSizes([430, 1050])
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.addWidget(splitter)

        self.import_btn.clicked.connect(self.import_geometry)
        self.clear_btn.clicked.connect(self.clear_geometry)
        self.manifest_btn.clicked.connect(self.load_manifest_dialog)
        self.screenshot_btn.clicked.connect(self.export_screenshot)
        self.component_combo.currentTextChanged.connect(self._selection_changed)
        self.isolate_btn.clicked.connect(self.isolate_selected)
        self.show_all_btn.clicked.connect(self.show_full_assembly)
        self.iso_btn.clicked.connect(self.plotter.view_isometric)
        self.front_btn.clicked.connect(self.plotter.view_xz)
        self.side_btn.clicked.connect(self.plotter.view_yz)
        self.top_btn.clicked.connect(self.plotter.view_xy)
        self.fit_btn.clicked.connect(self._fit_view)

    @staticmethod
    def _polydata(part: GeometryPart, component_index: int) -> pv.PolyData:
        vtk_faces = np.column_stack((np.full(part.n_faces, 3, dtype=np.int64), part.faces)).ravel()
        poly = pv.PolyData(part.vertices_m, vtk_faces)
        poly.field_data["scmepls_component_index"] = np.array([component_index], dtype=np.int32)
        if part.cell_rgb is not None and len(part.cell_rgb) == part.n_faces:
            poly.cell_data["face_rgb"] = part.cell_rgb
        return poly

    def import_geometry(self) -> None:
        if self._load_thread is not None:
            QMessageBox.information(self, "Geometry import", "A geometry file is already being loaded.")
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import external 3D geometry",
            "",
            "3D geometry (*.glb *.stl);;GLB files (*.glb);;STL files (*.stl)",
        )
        if not path:
            return
        if Path(path).suffix.lower() == ".glb" and self.unit_combo.currentText() != "m":
            answer = QMessageBox.question(
                self,
                "Nonstandard GLB units",
                "glTF/GLB convention is metres. Continue with the selected non-metre override?",
            )
            if answer != QMessageBox.StandardButton.Yes:
                return

        self.import_btn.setEnabled(False)
        self.file_label.setText(f"Loading: {path}")
        thread = QThread(self)
        worker = _GeometryLoadWorker(path, self.unit_combo.currentText(), str(self.axis_combo.currentData()))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.loaded.connect(self._geometry_loaded)
        worker.failed.connect(self._geometry_failed)
        worker.loaded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(self._geometry_thread_finished)
        self._load_thread = thread
        self._load_worker = worker
        thread.start()

    @pyqtSlot(object)
    def _geometry_loaded(self, asset: object) -> None:
        if not isinstance(asset, GeometryAsset):
            self._geometry_failed("Geometry loader returned an invalid asset object.")
            return
        self._display_asset(asset)
        self._auto_load_manifest()

    @pyqtSlot(str)
    def _geometry_failed(self, message: str) -> None:
        self.file_label.setText("No geometry loaded")
        QMessageBox.critical(self, "Geometry import failed", message)

    @pyqtSlot()
    def _geometry_thread_finished(self) -> None:
        self.import_btn.setEnabled(True)
        if self._load_worker is not None:
            self._load_worker.deleteLater()
        if self._load_thread is not None:
            self._load_thread.deleteLater()
        self._load_worker = None
        self._load_thread = None

    def _display_asset(self, asset: GeometryAsset) -> None:
        self.asset = asset
        self.manifest = None
        self.manifest_error = None
        self.plotter.clear()
        self.plotter.set_background("white")
        self.plotter.add_axes()
        self.actors.clear()
        self.component_by_index.clear()

        for component_index, part in enumerate(asset.parts, start=1):
            poly = self._polydata(part, component_index)
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
            self.component_by_index[component_index] = part.component_id

        self.component_combo.blockSignals(True)
        self.component_combo.clear()
        self.component_combo.addItems([part.component_id for part in asset.parts])
        self.component_combo.blockSignals(False)
        self.component_combo.setEnabled(True)
        self.isolate_btn.setEnabled(True)
        self.show_all_btn.setEnabled(True)
        self.manifest_btn.setEnabled(True)
        self.screenshot_btn.setEnabled(True)
        self.file_label.setText(str(asset.source_path))
        self.manifest_label.setText("Manifest: not loaded")
        self._update_info()
        self.show_full_assembly()
        self._fit_view()
        if self.component_combo.count():
            self._highlight(self.component_combo.currentText())

        if asset.warnings:
            QMessageBox.warning(self, "Geometry imported with warnings", "\n".join(asset.warnings))

    def _auto_load_manifest(self) -> None:
        if self.asset is None:
            return
        candidate = manifest_path_for_geometry(self.asset.source_path)
        if not candidate.exists():
            self.manifest_label.setText(f"Manifest: optional sidecar not found ({candidate.name})")
            self._update_info()
            return
        self._load_manifest(candidate, show_success=False)

    def load_manifest_dialog(self) -> None:
        if self.asset is None:
            return
        start = str(manifest_path_for_geometry(self.asset.source_path))
        path, _ = QFileDialog.getOpenFileName(self, "Load geometry manifest", start, "JSON files (*.json)")
        if path:
            self._load_manifest(Path(path), show_success=True)

    def _load_manifest(self, path: Path, show_success: bool) -> None:
        assert self.asset is not None
        try:
            available = {part.component_id for part in self.asset.parts}
            self.manifest = load_geometry_manifest(path, available)
            self.manifest_error = None
            self.manifest_label.setText(f"Manifest: VALID — {path.name}")
            if show_success:
                QMessageBox.information(self, "Geometry manifest", "Manifest is valid for the currently loaded geometry.")
        except GeometryManifestError as exc:
            self.manifest = None
            self.manifest_error = str(exc)
            self.manifest_label.setText(f"Manifest: INVALID — {exc}")
            QMessageBox.warning(self, "Geometry manifest invalid", str(exc))
        self._update_info()

    def clear_geometry(self) -> None:
        if self._load_thread is not None:
            QMessageBox.warning(self, "Geometry import", "Wait for the current import to finish before clearing the viewer.")
            return
        self.asset = None
        self.manifest = None
        self.manifest_error = None
        self.actors.clear()
        self.component_by_index.clear()
        self.component_combo.clear()
        self.component_combo.setEnabled(False)
        self.isolate_btn.setEnabled(False)
        self.show_all_btn.setEnabled(False)
        self.manifest_btn.setEnabled(False)
        self.screenshot_btn.setEnabled(False)
        self.file_label.setText("No geometry loaded")
        self.manifest_label.setText("Manifest: not loaded")
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
        if self.manifest is not None:
            manifest_status = f"VALID ({len(self.manifest.components)} bindings)"
        elif self.manifest_error:
            manifest_status = "INVALID"
        else:
            manifest_status = "Not loaded"
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
            ("Manifest", manifest_status),
        ]
        self.info_table.setRowCount(len(rows))
        for row, (key, value) in enumerate(rows):
            self.info_table.setItem(row, 0, QTableWidgetItem(key))
            self.info_table.setItem(row, 1, QTableWidgetItem(value))
        self.info_table.resizeColumnsToContents()

    def _mesh_picked(self, mesh: pv.DataSet) -> None:
        try:
            values = np.asarray(mesh.field_data["scmepls_component_index"]).ravel()
            component_index = int(values[0])
        except Exception:
            return
        component_id = self.component_by_index.get(component_index)
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

    def export_screenshot(self) -> None:
        if self.asset is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Export 3D screenshot",
            "scmepls_3d_digital_twin.png",
            "PNG image (*.png)",
        )
        if not path:
            return
        output = Path(path)
        if output.suffix.lower() != ".png":
            output = output.with_suffix(".png")
        try:
            self.plotter.screenshot(str(output))
            metadata_path = output.with_suffix(".metadata.json")
            metadata = {
                "image_file": output.name,
                "exported_utc": datetime.now(timezone.utc).isoformat(),
                "analysis": "SC-MEPLS external-geometry 3D digital twin",
                "scientific_status": "Visualization only; geometry does not modify model physics.",
                "geometry_source": str(self.asset.source_path),
                "source_unit": self.asset.source_unit,
                "axis_mode": self.asset.axis_mode,
                "components": [part.component_id for part in self.asset.parts],
                "triangle_count": self.asset.triangle_count,
                "dimensions_m": self.asset.dimensions_m.tolist(),
                "geometry_warnings": list(self.asset.warnings),
                "manifest": str(self.manifest.path) if self.manifest else None,
                "manifest_error": self.manifest_error,
                "selected_component": self.component_combo.currentText(),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            QMessageBox.information(self, "3D export complete", f"Saved:\n{output}\n{metadata_path}")
        except Exception as exc:
            QMessageBox.critical(self, "3D export error", str(exc))

    def _fit_view(self) -> None:
        if self.actors:
            self.plotter.reset_camera()
        self.plotter.render()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        try:
            if self._load_thread is not None and self._load_thread.isRunning():
                self._load_thread.quit()
                self._load_thread.wait(1500)
            self.plotter.close()
        finally:
            super().closeEvent(event)
