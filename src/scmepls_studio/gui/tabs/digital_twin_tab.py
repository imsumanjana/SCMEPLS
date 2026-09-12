from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import pyvista as pv
from pyvistaqt import QtInteractor
from vtkmodules.vtkCommonMath import vtkMatrix4x4
from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ...digital_twin import (
    MeshRegistry,
    PlaybackController,
    ResultField,
    SimulationTimeline,
    actor_style,
    build_scene_bindings,
    component_result,
    component_transforms,
    digital_twin_history_figure,
    force_vector_n,
)
from ...geometry import (
    GeometryAsset,
    GeometryManifest,
    GeometryManifestError,
    load_geometry,
    load_geometry_manifest,
    manifest_path_for_geometry,
)
from ...version import __version__
from ..common import ExportBar, FigureCanvasPanel, ScrollableControls, form_row, save_panel_figure, section_label


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


def _vtk_matrix(values: np.ndarray) -> vtkMatrix4x4:
    matrix = vtkMatrix4x4()
    for row in range(4):
        for column in range(4):
            matrix.SetElement(row, column, float(values[row, column]))
    return matrix


def _result_rgb(metric: str, normalized: float) -> tuple[float, float, float]:
    value = float(np.clip(normalized, 0.0, 1.0))
    if metric == "health":
        severity = 1.0 - value
        return (0.15 + 0.75 * severity, 0.75 - 0.55 * severity, 0.20 - 0.10 * severity)
    return (0.15 + 0.75 * value, 0.35 - 0.18 * value, 0.85 - 0.72 * value)


class DigitalTwinTab(QWidget):
    """Imported geometry + mesh + SC-MEPLS physics + animation + linked results."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.asset: GeometryAsset | None = None
        self.manifest: GeometryManifest | None = None
        self.manifest_error: str | None = None
        self.bindings = None
        self.meshes: MeshRegistry | None = None
        self.actors: dict[str, object] = {}
        self.timeline: SimulationTimeline | None = None
        self.playback: PlaybackController | None = None
        self.result_field: ResultField | None = None
        self.simulation_metrics: dict = {}
        self._load_thread: QThread | None = None
        self._load_worker: _GeometryLoadWorker | None = None
        self._last_tick = time.monotonic()
        self._plot_time_lines: list[object] = []
        self._current_frame = None

        self.controls = ScrollableControls()
        self.controls.add_widget(section_label("External 3D geometry"))
        note = QLabel(
            "Import GLB/STL geometry and bind it to the independent SC-MEPLS simulation. Geometry remains visualization-only: "
            "simulation mass, inertia, actuator coordinates, controllers, and equations are never derived from the mesh."
        )
        note.setWordWrap(True)
        self.controls.add_widget(note)

        self.unit_combo = QComboBox(); self.unit_combo.addItems(["m", "mm", "cm"])
        self.axis_combo = QComboBox()
        self.axis_combo.addItem("Auto: GLB Y-up → Z-up; STL as stored", "auto")
        self.axis_combo.addItem("As stored", "as_stored")
        self.axis_combo.addItem("Force GLB Y-up → SC-MEPLS Z-up", "gltf_y_up_to_z_up")
        self.controls.add_widget(form_row("Source geometry unit", self.unit_combo))
        self.controls.add_widget(form_row("Axis conversion", self.axis_combo))
        self.import_btn = QPushButton("Import GLB / STL…")
        self.clear_btn = QPushButton("Clear Geometry")
        self.manifest_btn = QPushButton("Load / Validate Manifest…")
        self.screenshot_btn = QPushButton("Export 3D Screenshot…")
        for button in (self.manifest_btn, self.screenshot_btn): button.setEnabled(False)
        for button in (self.import_btn, self.clear_btn, self.manifest_btn, self.screenshot_btn): self.controls.add_widget(button)
        self.file_label = QLabel("No geometry loaded"); self.file_label.setWordWrap(True); self.controls.add_widget(self.file_label)
        self.manifest_label = QLabel("Manifest: not loaded"); self.manifest_label.setWordWrap(True); self.controls.add_widget(self.manifest_label)

        self.controls.add_widget(section_label("Mesh / component view"))
        self.mesh_mode_combo = QComboBox()
        self.mesh_mode_combo.addItem("Surface", "surface")
        self.mesh_mode_combo.addItem("Surface + edges", "surface_edges")
        self.mesh_mode_combo.addItem("Wireframe", "wireframe")
        self.controls.add_widget(form_row("Mesh display", self.mesh_mode_combo))
        self.component_combo = QComboBox(); self.component_combo.setEnabled(False)
        self.controls.add_widget(form_row("Selected component", self.component_combo))
        self.isolate_btn = QPushButton("Isolate Selected"); self.isolate_btn.setEnabled(False)
        self.show_all_btn = QPushButton("Full Assembly"); self.show_all_btn.setEnabled(False)
        self.controls.add_widget(self.isolate_btn); self.controls.add_widget(self.show_all_btn)

        self.controls.add_widget(section_label("Simulation / animation"))
        self.simulation_label = QLabel("Simulation: waiting for SC-MEPLS history")
        self.simulation_label.setWordWrap(True); self.controls.add_widget(self.simulation_label)
        self.load_history_btn = QPushButton("Import Time-History CSV…"); self.controls.add_widget(self.load_history_btn)
        playback_row = QWidget(); playback_layout = QHBoxLayout(playback_row); playback_layout.setContentsMargins(0,0,0,0)
        self.play_btn = QPushButton("Play"); self.pause_btn = QPushButton("Pause"); self.reset_btn = QPushButton("Reset")
        playback_layout.addWidget(self.play_btn); playback_layout.addWidget(self.pause_btn); playback_layout.addWidget(self.reset_btn)
        self.controls.add_widget(playback_row)
        self.speed_combo = QComboBox()
        for text, value in (("0.25×",0.25),("0.5×",0.5),("1×",1.0),("2×",2.0),("4×",4.0)):
            self.speed_combo.addItem(text, value)
        self.speed_combo.setCurrentText("1×")
        self.loop_check = QCheckBox("Loop playback")
        self.controls.add_widget(form_row("Playback speed", self.speed_combo)); self.controls.add_widget(self.loop_check)
        self.time_slider = QSlider(Qt.Orientation.Horizontal); self.time_slider.setRange(0, 10000); self.time_slider.setEnabled(False)
        self.time_label = QLabel("t = —")
        self.controls.add_widget(self.time_slider); self.controls.add_widget(self.time_label)

        self.controls.add_widget(section_label("Result visualization"))
        self.metric_combo = QComboBox()
        for label, key in (("Air gap","gap"),("Coil current","current"),("EM force","em_force"),("Pressure","pressure"),("Pneumatic force","pneumatic_force"),("Health","health")):
            self.metric_combo.addItem(label, key)
        self.metric_combo.setCurrentIndex(2)
        self.controls.add_widget(form_row("Module result", self.metric_combo))
        self.color_results_check = QCheckBox("Color bound module geometry by result"); self.color_results_check.setChecked(True)
        self.force_check = QCheckBox("Show module force vectors"); self.force_check.setChecked(True)
        self.force_kind_combo = QComboBox()
        self.force_kind_combo.addItem("Total support", "total"); self.force_kind_combo.addItem("Electromagnetic", "em"); self.force_kind_combo.addItem("Pneumatic", "pneumatic")
        self.cg_check = QCheckBox("Show CG marker"); self.cg_check.setChecked(True)
        self.controls.add_widget(self.color_results_check); self.controls.add_widget(self.force_check)
        self.controls.add_widget(form_row("Force vectors", self.force_kind_combo)); self.controls.add_widget(self.cg_check)

        self.controls.add_widget(section_label("Current frame"))
        self.frame_table = QTableWidget(0,2); self.frame_table.setHorizontalHeaderLabels(["Signal","Value"]); self.frame_table.setMinimumHeight(300)
        self.controls.add_widget(self.frame_table)
        self.plot_export_bar = ExportBar("digital_twin_linked_history_600dpi.png"); self.controls.add_widget(self.plot_export_bar)

        self.controls.add_widget(section_label("Camera"))
        self.iso_btn = QPushButton("Isometric"); self.front_btn = QPushButton("Front"); self.side_btn = QPushButton("Side"); self.top_btn = QPushButton("Top"); self.fit_btn = QPushButton("Fit View")
        for button in (self.iso_btn,self.front_btn,self.side_btn,self.top_btn,self.fit_btn): self.controls.add_widget(button)

        self.controls.add_widget(section_label("Geometry / binding validation"))
        self.info_table = QTableWidget(0,2); self.info_table.setHorizontalHeaderLabels(["Property","Value"]); self.info_table.setMinimumHeight(300)
        self.controls.add_widget(self.info_table)

        right = QWidget(); right_layout = QVBoxLayout(right); right_layout.setContentsMargins(0,0,0,0)
        self.viewer_title = QLabel("3D Digital Twin — geometry, mesh, physics, animation and results")
        right_layout.addWidget(self.viewer_title)
        right_splitter = QSplitter(Qt.Orientation.Vertical)
        viewer = QWidget(); viewer_layout = QVBoxLayout(viewer); viewer_layout.setContentsMargins(0,0,0,0)
        self.plotter = QtInteractor(viewer); viewer_layout.addWidget(self.plotter.interactor, 1)
        self.plotter.set_background("white"); self.plotter.add_axes()
        self.plotter.enable_mesh_picking(callback=self._mesh_picked, show=False, show_message=False, left_clicking=True)
        right_splitter.addWidget(viewer)
        plot_widget = QWidget(); plot_layout = QVBoxLayout(plot_widget); plot_layout.setContentsMargins(0,0,0,0)
        plot_layout.addWidget(QLabel("Linked simulation history")); self.history_plot = FigureCanvasPanel(); plot_layout.addWidget(self.history_plot,1)
        right_splitter.addWidget(plot_widget); right_splitter.setSizes([620,300]); right_layout.addWidget(right_splitter,1)

        splitter = QSplitter(Qt.Orientation.Horizontal); splitter.addWidget(self.controls); splitter.addWidget(right); splitter.setSizes([450,1050])
        layout = QHBoxLayout(self); layout.setContentsMargins(6,6,6,6); layout.addWidget(splitter)

        self.import_btn.clicked.connect(self.import_geometry); self.clear_btn.clicked.connect(self.clear_geometry)
        self.manifest_btn.clicked.connect(self.load_manifest_dialog); self.screenshot_btn.clicked.connect(self.export_screenshot)
        self.mesh_mode_combo.currentIndexChanged.connect(self._mesh_mode_changed)
        self.component_combo.currentTextChanged.connect(self._selection_changed)
        self.isolate_btn.clicked.connect(self.isolate_selected); self.show_all_btn.clicked.connect(self.show_full_assembly)
        self.load_history_btn.clicked.connect(self.import_history_csv)
        self.play_btn.clicked.connect(self.play); self.pause_btn.clicked.connect(self.pause); self.reset_btn.clicked.connect(self.reset_playback)
        self.speed_combo.currentIndexChanged.connect(self._speed_changed); self.loop_check.toggled.connect(self._loop_changed)
        self.time_slider.valueChanged.connect(self._slider_changed)
        self.metric_combo.currentIndexChanged.connect(self._metric_changed)
        self.color_results_check.toggled.connect(lambda _: self._apply_current_frame())
        self.force_check.toggled.connect(lambda _: self._apply_current_frame())
        self.force_kind_combo.currentIndexChanged.connect(lambda _: self._apply_current_frame())
        self.cg_check.toggled.connect(lambda _: self._apply_current_frame())
        self.iso_btn.clicked.connect(self.plotter.view_isometric); self.front_btn.clicked.connect(self.plotter.view_xz)
        self.side_btn.clicked.connect(self.plotter.view_yz); self.top_btn.clicked.connect(self.plotter.view_xy); self.fit_btn.clicked.connect(self._fit_view)
        self.plot_export_bar.export_requested.connect(self.export_linked_plot)

        self._timer = QTimer(self); self._timer.setInterval(33); self._timer.timeout.connect(self._animation_tick); self._timer.start()

    @pyqtSlot(object, object)
    def bind_simulation(self, history: object, metrics: object = None) -> None:
        try:
            if not isinstance(history, pd.DataFrame):
                raise ValueError("Digital twin requires a pandas simulation time-history table.")
            self.timeline = SimulationTimeline(history)
            self.playback = PlaybackController(self.timeline, speed=float(self.speed_combo.currentData()), loop=self.loop_check.isChecked())
            self.simulation_metrics = dict(metrics) if isinstance(metrics, dict) else {}
            self.result_field = ResultField(self.timeline, str(self.metric_combo.currentData()))
            self.time_slider.setEnabled(True)
            self.simulation_label.setText(f"Simulation: bound — {len(self.timeline):,} samples, {self.timeline.time_s[0]:.3f}–{self.timeline.time_s[-1]:.3f} s")
            self._last_tick = time.monotonic()
            self._apply_frame(self.timeline.frame_at_index(0), rebuild_plot=True)
        except Exception as exc:
            self.timeline = None; self.playback = None; self.result_field = None; self.time_slider.setEnabled(False)
            self.simulation_label.setText(f"Simulation binding failed: {exc}")
            QMessageBox.warning(self, "Digital-twin simulation binding", str(exc))

    def import_history_csv(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Import SC-MEPLS time history", "", "CSV files (*.csv)")
        if not path: return
        try:
            history = pd.read_csv(path)
            self.bind_simulation(history, {"source_csv": str(Path(path).resolve())})
        except Exception as exc:
            QMessageBox.critical(self, "Time-history import failed", str(exc))

    def import_geometry(self) -> None:
        if self._load_thread is not None:
            QMessageBox.information(self, "Geometry import", "A geometry file is already being loaded."); return
        path, _ = QFileDialog.getOpenFileName(self, "Import external 3D geometry", "", "3D geometry (*.glb *.stl);;GLB files (*.glb);;STL files (*.stl)")
        if not path: return
        if Path(path).suffix.lower() == ".glb" and self.unit_combo.currentText() != "m":
            answer = QMessageBox.question(self, "Nonstandard GLB units", "glTF/GLB convention is metres. Continue with the selected non-metre override?")
            if answer != QMessageBox.StandardButton.Yes: return
        self.import_btn.setEnabled(False); self.file_label.setText(f"Loading: {path}")
        thread = QThread(self); worker = _GeometryLoadWorker(path, self.unit_combo.currentText(), str(self.axis_combo.currentData()))
        worker.moveToThread(thread); thread.started.connect(worker.run); worker.loaded.connect(self._geometry_loaded); worker.failed.connect(self._geometry_failed)
        worker.loaded.connect(thread.quit); worker.failed.connect(thread.quit); thread.finished.connect(self._geometry_thread_finished)
        self._load_thread = thread; self._load_worker = worker; thread.start()

    @pyqtSlot(object)
    def _geometry_loaded(self, asset: object) -> None:
        if not isinstance(asset, GeometryAsset): self._geometry_failed("Geometry loader returned an invalid asset object."); return
        self.asset = asset; self.manifest = None; self.manifest_error = None
        candidate = manifest_path_for_geometry(asset.source_path)
        if candidate.exists():
            try:
                self.manifest = load_geometry_manifest(candidate, {part.component_id for part in asset.parts})
                self.manifest_label.setText(f"Manifest: VALID — {candidate.name}")
            except GeometryManifestError as exc:
                self.manifest_error = str(exc); self.manifest_label.setText(f"Manifest: INVALID — {exc}")
        else:
            self.manifest_label.setText(f"Manifest: optional sidecar not found ({candidate.name}); conservative name binding used")
        self._rebuild_scene(); self.file_label.setText(str(asset.source_path)); self.manifest_btn.setEnabled(True); self.screenshot_btn.setEnabled(True)
        if asset.warnings: QMessageBox.warning(self, "Geometry imported with warnings", "\n".join(asset.warnings))

    @pyqtSlot(str)
    def _geometry_failed(self, message: str) -> None:
        self.file_label.setText("No geometry loaded"); QMessageBox.critical(self, "Geometry import failed", message)

    @pyqtSlot()
    def _geometry_thread_finished(self) -> None:
        self.import_btn.setEnabled(True)
        if self._load_worker is not None: self._load_worker.deleteLater()
        if self._load_thread is not None: self._load_thread.deleteLater()
        self._load_worker = None; self._load_thread = None

    def load_manifest_dialog(self) -> None:
        if self.asset is None: return
        start = str(manifest_path_for_geometry(self.asset.source_path))
        path, _ = QFileDialog.getOpenFileName(self, "Load geometry manifest", start, "JSON files (*.json)")
        if not path: return
        try:
            self.manifest = load_geometry_manifest(path, {part.component_id for part in self.asset.parts})
            self.manifest_error = None; self.manifest_label.setText(f"Manifest: VALID — {Path(path).name}")
            self._rebuild_scene(); QMessageBox.information(self, "Geometry manifest", "Manifest is valid and has been applied to geometry bindings.")
        except GeometryManifestError as exc:
            self.manifest = None; self.manifest_error = str(exc); self.manifest_label.setText(f"Manifest: INVALID — {exc}")
            QMessageBox.warning(self, "Geometry manifest invalid", str(exc)); self._update_info()

    def _rebuild_scene(self) -> None:
        if self.asset is None: return
        selected = self.component_combo.currentText()
        camera = self.plotter.camera_position if self.actors else None
        self.bindings = build_scene_bindings(self.asset, self.manifest)
        self.meshes = MeshRegistry.from_asset(self.asset, self.bindings)
        self.plotter.clear(); self.plotter.set_background("white"); self.plotter.add_axes(); self.actors.clear()
        mode = str(self.mesh_mode_combo.currentData())
        for record in self.meshes.records.values():
            actor = self.plotter.add_mesh(record.polydata, **actor_style(record, mode))
            self.actors[record.component_id] = actor
        self.component_combo.blockSignals(True); self.component_combo.clear(); self.component_combo.addItems(list(self.meshes.records))
        if selected:
            index = self.component_combo.findText(selected)
            if index >= 0: self.component_combo.setCurrentIndex(index)
        self.component_combo.blockSignals(False); self.component_combo.setEnabled(bool(self.actors)); self.isolate_btn.setEnabled(bool(self.actors)); self.show_all_btn.setEnabled(bool(self.actors))
        if camera is not None:
            try: self.plotter.camera_position = camera
            except Exception: pass
        else: self._fit_view()
        self._update_info(); self._apply_current_frame(rebuild_plot=True)

    def clear_geometry(self) -> None:
        if self._load_thread is not None:
            QMessageBox.warning(self, "Geometry import", "Wait for the current import to finish before clearing the viewer."); return
        self.asset = None; self.manifest = None; self.manifest_error = None; self.bindings = None; self.meshes = None; self.actors.clear()
        self.component_combo.clear(); self.component_combo.setEnabled(False); self.isolate_btn.setEnabled(False); self.show_all_btn.setEnabled(False)
        self.manifest_btn.setEnabled(False); self.screenshot_btn.setEnabled(False); self.file_label.setText("No geometry loaded"); self.manifest_label.setText("Manifest: not loaded")
        self.info_table.setRowCount(0); self.plotter.clear(); self.plotter.set_background("white"); self.plotter.add_axes(); self.plotter.render()

    def _mesh_mode_changed(self) -> None:
        if self.asset is not None: self._rebuild_scene()

    def _mesh_picked(self, mesh: pv.DataSet) -> None:
        try: component_index = int(np.asarray(mesh.field_data["scmepls_component_index"]).ravel()[0])
        except Exception: return
        if self.meshes is None: return
        component_id = self.meshes.component_by_index.get(component_index)
        if component_id is None: return
        index = self.component_combo.findText(component_id)
        if index >= 0: self.component_combo.setCurrentIndex(index)

    def _selection_changed(self, component_id: str) -> None:
        if component_id: self._highlight(component_id)
        self._update_frame_table(self._current_frame); self._refresh_plot()

    def _highlight(self, component_id: str) -> None:
        for name, actor in self.actors.items():
            try:
                prop = actor.GetProperty(); prop.SetEdgeVisibility(name == component_id); prop.SetLineWidth(2.5 if name == component_id else 1.0)
                if name == component_id: prop.SetEdgeColor(0.05,0.05,0.05)
            except Exception: pass
        self.plotter.render()

    def isolate_selected(self) -> None:
        selected = self.component_combo.currentText()
        if not selected: return
        for name, actor in self.actors.items(): actor.SetVisibility(name == selected)
        self._highlight(selected); self.plotter.reset_camera(); self.plotter.render()

    def show_full_assembly(self) -> None:
        for actor in self.actors.values(): actor.SetVisibility(True)
        selected = self.component_combo.currentText()
        if selected: self._highlight(selected)
        self.plotter.render()

    def play(self) -> None:
        if self.playback is None: QMessageBox.information(self, "Animation", "Run or import an SC-MEPLS time history first."); return
        if self.playback.current_time_s >= self.playback.maximum_time_s: self.playback.reset()
        self.playback.play(); self._last_tick = time.monotonic()

    def pause(self) -> None:
        if self.playback is not None: self.playback.pause()

    def reset_playback(self) -> None:
        if self.playback is None: return
        self.playback.reset(); self._apply_frame(self.playback.timeline.frame_at_index(0), rebuild_plot=False)

    def _speed_changed(self) -> None:
        if self.playback is not None: self.playback.set_speed(float(self.speed_combo.currentData()))

    def _loop_changed(self, checked: bool) -> None:
        if self.playback is not None: self.playback.loop = bool(checked)

    def _slider_changed(self, value: int) -> None:
        if self.playback is None: return
        frame = self.playback.seek_fraction(value / 10000.0); self._apply_frame(frame, update_slider=False, rebuild_plot=False)

    def _metric_changed(self) -> None:
        if self.timeline is not None: self.result_field = ResultField(self.timeline, str(self.metric_combo.currentData()))
        self._apply_current_frame(rebuild_plot=True)

    def _animation_tick(self) -> None:
        if self.playback is None or not self.playback.playing: return
        now = time.monotonic(); elapsed = max(0.0, now - self._last_tick); self._last_tick = now
        frame = self.playback.advance(elapsed); self._apply_frame(frame, rebuild_plot=False)

    def _apply_current_frame(self, rebuild_plot: bool = False) -> None:
        if self.playback is None: return
        self._apply_frame(self.playback.timeline.frame_at_time(self.playback.current_time_s), rebuild_plot=rebuild_plot)

    def _apply_frame(self, frame, *, update_slider: bool = True, rebuild_plot: bool = False) -> None:
        self._current_frame = frame
        transforms = None
        if self.timeline is not None and self.bindings is not None:
            transforms = component_transforms(self.bindings, self.timeline, frame)
            for component_id, matrix in transforms.items():
                actor = self.actors.get(component_id)
                if actor is not None: actor.SetUserMatrix(_vtk_matrix(matrix))
            self._apply_result_styles(frame); self._update_force_vectors(frame, transforms); self._update_cg(frame)
            self.plotter.render()
        if self.playback is not None:
            if update_slider:
                self.time_slider.blockSignals(True); self.time_slider.setValue(int(round(self.playback.progress * 10000))); self.time_slider.blockSignals(False)
            self.time_label.setText(f"t = {frame.time_s:.3f} s | mode = {frame.mode} | lock = {frame.lock_fraction:.3f}")
        self._update_frame_table(frame)
        if rebuild_plot or not self._plot_time_lines: self._refresh_plot()
        else: self._update_plot_cursor(frame.time_s)

    def _apply_result_styles(self, frame) -> None:
        if self.meshes is None or self.bindings is None: return
        metric = str(self.metric_combo.currentData())
        color_results = self.color_results_check.isChecked() and self.result_field is not None
        for component_id, actor in self.actors.items():
            record = self.meshes.for_component(component_id); binding = self.bindings.for_component(component_id)
            try:
                mapper = actor.GetMapper(); prop = actor.GetProperty(); prop.SetOpacity(0.25 + 0.75 * frame.lock_fraction if binding.role.lower() == "lock" else 1.0)
                if color_results and binding.simulation_module is not None:
                    result = component_result(binding, frame, self.result_field); mapper.ScalarVisibilityOff(); prop.SetColor(*_result_rgb(metric, result.normalized))
                else:
                    if "face_rgb" in record.polydata.cell_data and str(self.mesh_mode_combo.currentData()) != "wireframe": mapper.ScalarVisibilityOn()
                    else: mapper.ScalarVisibilityOff(); prop.SetColor(0.82,0.82,0.82)
            except Exception: pass
        selected = self.component_combo.currentText()
        if selected: self._highlight(selected)

    def _remove_overlay(self, name: str) -> None:
        try: self.plotter.remove_actor(name, render=False)
        except Exception: pass

    def _update_force_vectors(self, frame, transforms: dict[str,np.ndarray]) -> None:
        self._remove_overlay("dt_force_vectors")
        if not self.force_check.isChecked() or self.asset is None or self.meshes is None or self.bindings is None: return
        origins: list[np.ndarray] = []; forces: list[float] = []
        kind = str(self.force_kind_combo.currentData())
        for module_state in frame.modules:
            component_ids = self.bindings.components_for_module(module_state.module)
            component_id = next((name for name in component_ids if name in self.meshes.records), None)
            if component_id is None: continue
            centroid = np.array([*self.meshes.for_component(component_id).base_centroid_m, 1.0])
            origin = transforms[component_id] @ centroid
            origins.append(origin[:3]); forces.append(float(force_vector_n(module_state, kind)[2]))
        if not origins: return
        maximum = max(max(forces), 1e-9); characteristic = max(float(np.max(self.asset.dimensions_m)), 0.1)
        vectors = np.array([[0.0,0.0,0.22 * characteristic * force / maximum] for force in forces], dtype=float)
        self.plotter.add_arrows(np.asarray(origins), vectors, mag=1.0, color="orange", name="dt_force_vectors")

    def _update_cg(self, frame) -> None:
        self._remove_overlay("dt_cg")
        if not self.cg_check.isChecked() or self.asset is None or self.timeline is None or self.bindings is None: return
        origin = np.asarray(self.bindings.platform_origin_m, dtype=float)
        local = np.array([origin[0] + frame.cg_shift_x_m, origin[1] + frame.cg_shift_y_m, origin[2], 1.0])
        matrix = self.timeline.relative_transform(frame, self.bindings.platform_origin_m)
        world = matrix @ local; radius = max(float(np.max(self.asset.dimensions_m)) * 0.012, 0.002)
        self.plotter.add_mesh(pv.Sphere(radius=radius, center=world[:3]), color="magenta", name="dt_cg", pickable=False)

    def _update_frame_table(self, frame) -> None:
        if frame is None: self.frame_table.setRowCount(0); return
        body = frame.rigid_body
        rows = [
            ("Time", f"{frame.time_s:.4f} s"),("Mode", frame.mode),("X / Y / Z", f"{body.x_m:.5f}, {body.y_m:.5f}, {body.z_m:.5f} m"),
            ("Roll / Pitch / Yaw", f"{np.degrees(body.roll_rad):.4f}, {np.degrees(body.pitch_rad):.4f}, {np.degrees(body.yaw_rad):.4f} deg"),
            ("Lock fraction", f"{frame.lock_fraction:.4f}"),("Unsafe", "YES" if frame.unsafe else "No"),("Fault severity", f"{frame.fault_severity:.4f}"),
            ("CG shift", f"{frame.cg_shift_x_m:.4f}, {frame.cg_shift_y_m:.4f} m"),("Wind force", f"{frame.wind_force_n:.3f} N"),
        ]
        selected = self.component_combo.currentText()
        if selected and self.bindings is not None:
            binding = self.bindings.bindings.get(selected)
            if binding is not None:
                rows.extend([("Component", selected),("Role / group", f"{binding.role} / {binding.dynamic_group}"),("Binding source", binding.source)])
                if binding.simulation_module is not None:
                    module = frame.modules[binding.simulation_module - 1]
                    rows.extend([
                        ("Module", f"M{binding.simulation_module}"),("True / sensor / estimated gap", f"{module.true_gap_m*1000:.3f} / {module.sensor_gap_m*1000:.3f} / {module.estimated_gap_m*1000:.3f} mm"),
                        ("Coil current", f"{module.coil_current_a:.3f} A"),("EM force", f"{module.em_force_n:.3f} N"),("Pneumatic pressure", f"{module.pressure_pa:.3f} Pa"),
                        ("Pneumatic force", f"{module.pneumatic_force_n:.3f} N"),("Health / efficiency", f"{module.health:.3f} / {module.coil_efficiency:.3f}"),
                    ])
                    if self.result_field is not None:
                        result = component_result(binding, frame, self.result_field); rows.append(("Displayed result", f"{result.value:.6g} {result.unit}"))
        self.frame_table.setRowCount(len(rows))
        for row,(key,value) in enumerate(rows): self.frame_table.setItem(row,0,QTableWidgetItem(str(key))); self.frame_table.setItem(row,1,QTableWidgetItem(str(value)))
        self.frame_table.resizeColumnsToContents()

    def _refresh_plot(self) -> None:
        if self.timeline is None: return
        binding = None; selected = self.component_combo.currentText()
        if selected and self.bindings is not None: binding = self.bindings.bindings.get(selected)
        current_time = self._current_frame.time_s if self._current_frame is not None else self.timeline.time_s[0]
        fig = digital_twin_history_figure(self.timeline, binding, str(self.metric_combo.currentData()), current_time)
        self.history_plot.set_figure(fig); self._plot_time_lines = [axis.lines[-1] for axis in fig.axes if axis.lines]

    def _update_plot_cursor(self, time_s: float) -> None:
        if not self._plot_time_lines: return
        for line in self._plot_time_lines:
            try: line.set_xdata([time_s,time_s])
            except Exception: pass
        self.history_plot.canvas.draw_idle()

    def _update_info(self) -> None:
        if self.asset is None: self.info_table.setRowCount(0); return
        dims = self.asset.dimensions_m; manifest_status = "VALID" if self.manifest is not None else ("INVALID" if self.manifest_error else "Not loaded")
        dynamic_count = len(self.bindings.platform_components) if self.bindings is not None else 0
        bound_modules = sorted({b.simulation_module for b in self.bindings.bindings.values() if b.simulation_module is not None}) if self.bindings is not None else []
        rows = [
            ("Format",self.asset.source_path.suffix.lower()),("Source unit",self.asset.source_unit),("Axis mode",self.asset.axis_mode),("Scale to metre",f"{self.asset.scale_to_m:g}"),
            ("Components",str(len(self.asset.parts))),("Mesh points",f"{self.meshes.point_count:,}" if self.meshes else "—"),("Triangles",f"{self.asset.triangle_count:,}"),
            ("Size X / Y / Z (m)",f"{dims[0]:.6g} / {dims[1]:.6g} / {dims[2]:.6g}"),("Dynamic components",str(dynamic_count)),
            ("Simulation modules bound",", ".join(f"M{x}" for x in bound_modules) if bound_modules else "None"),("Manifest",manifest_status),("Warnings",str(len(self.asset.warnings))),
        ]
        self.info_table.setRowCount(len(rows))
        for row,(key,value) in enumerate(rows): self.info_table.setItem(row,0,QTableWidgetItem(key)); self.info_table.setItem(row,1,QTableWidgetItem(value))
        self.info_table.resizeColumnsToContents()

    def _fit_view(self) -> None:
        try: self.plotter.reset_camera(); self.plotter.view_isometric(); self.plotter.render()
        except Exception: pass

    def plot_metadata(self) -> dict:
        return {
            "analysis":"3D Digital Twin linked history","software_version":__version__,"geometry":None if self.asset is None else str(self.asset.source_path),
            "manifest":None if self.manifest is None else str(self.manifest.path),"selected_component":self.component_combo.currentText() or None,
            "result_metric":str(self.metric_combo.currentData()),"current_time_s":None if self._current_frame is None else float(self._current_frame.time_s),
            "simulation_metrics":{k:v for k,v in self.simulation_metrics.items() if k != "parameters"},
        }

    def export_linked_plot(self, dpi: int, path: str) -> None:
        if self.timeline is None: QMessageBox.information(self,"Export","No simulation is bound to the digital twin."); return
        try:
            out, meta = save_panel_figure(self,self.history_plot,dpi,path,self.plot_metadata()); QMessageBox.information(self,"Export complete",f"Saved:\n{out}\n{meta}")
        except Exception as exc: QMessageBox.critical(self,"Export error",str(exc))

    def export_screenshot(self) -> None:
        if self.asset is None: return
        path, _ = QFileDialog.getSaveFileName(self,"Export 3D screenshot","scmepls_3d_digital_twin.png","PNG image (*.png)")
        if not path: return
        output = Path(path); output = output if output.suffix.lower() == ".png" else output.with_suffix(".png")
        try:
            self.plotter.screenshot(str(output)); metadata = self.plot_metadata(); metadata.update({"created_utc":datetime.now(timezone.utc).isoformat(),"geometry_unit":self.asset.source_unit,"axis_mode":self.asset.axis_mode,"triangle_count":self.asset.triangle_count})
            meta_path = output.with_suffix(output.suffix + ".json"); meta_path.write_text(json.dumps(metadata,indent=2,default=str),encoding="utf-8")
            QMessageBox.information(self,"3D export complete",f"Saved:\n{output}\n{meta_path}")
        except Exception as exc: QMessageBox.critical(self,"3D export error",str(exc))

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._timer.stop()
        try: self.plotter.close()
        except Exception: pass
        super().closeEvent(event)
