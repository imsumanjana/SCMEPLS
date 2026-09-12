from __future__ import annotations

from pathlib import Path

import numpy as np
import pyvista as pv
from PyQt6.QtWidgets import QLabel, QMessageBox

from ...digital_twin import component_result, force_vector_n
from ...geometry import GeometryAsset, GeometryManifestError, load_geometry, load_geometry_manifest, manifest_path_for_geometry
from .digital_twin_tab import DigitalTwinTab as _BaseDigitalTwinTab, _result_rgb


class DigitalTwinTab(_BaseDigitalTwinTab):
    """Scientific UI refinements for the imported-geometry digital twin.

    The base tab remains responsible for asynchronous loading, component selection,
    playback and linked plots. This refinement binds validated structural context to
    force-vector origins and the CG marker, removes the old lock-opacity surrogate,
    and provides deterministic project restore helpers.
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.structural_centroid_m: np.ndarray | None = None
        self.structural_module_points_m: np.ndarray | None = None
        self.structural_context_validated = False
        self.viewer_title.setText("3D Digital Twin — imported CAD, physical animation and linked results")
        # Clarify that this control changes only the render surface. The tetrahedral
        # FEA mesh has its own Structural FEA tab and should never be confused with it.
        for label in self.findChildren(QLabel):
            if label.text() == "Mesh / component view":
                label.setText("Render surface / component view")
            elif label.text() == "Mesh display":
                label.setText("Render surface")
        self.cg_check.setText("Show validated CG marker")
        self.force_check.setText("Show module force vectors at physical M1–M8 locations")

    def bind_structural_context(self, report: object | None) -> None:
        """Bind centroid and actuator locations from a completed structural validation."""
        self.structural_centroid_m = None
        self.structural_module_points_m = None
        self.structural_context_validated = False
        if report is None:
            self._apply_current_frame()
            return
        try:
            centroid = np.asarray(getattr(report, "centroid_m"), dtype=float)
            modules = np.asarray(getattr(report, "module_points_m"), dtype=float)
            if centroid.shape != (3,) or modules.shape != (8, 3):
                raise ValueError("Structural report centroid/module geometry has invalid shape.")
            if np.any(~np.isfinite(centroid)) or np.any(~np.isfinite(modules)):
                raise ValueError("Structural report centroid/module geometry contains non-finite values.")
            self.structural_centroid_m = centroid
            self.structural_module_points_m = modules
            self.structural_context_validated = True
            self._apply_current_frame()
            self._update_info()
        except Exception as exc:
            QMessageBox.warning(self, "Structural 3D context", f"Could not bind structural geometry context: {exc}")

    def load_geometry_path(
        self,
        path: str | Path,
        *,
        source_unit: str = "m",
        axis_mode: str = "auto",
        manifest_path: str | Path | None = None,
    ) -> None:
        """Synchronously restore a project geometry after validating the referenced file."""
        source = Path(path)
        asset = load_geometry(source, source_unit=source_unit, axis_mode=axis_mode)
        self.asset = asset
        self.manifest = None
        self.manifest_error = None
        candidate = Path(manifest_path) if manifest_path else manifest_path_for_geometry(source)
        if candidate.exists():
            try:
                self.manifest = load_geometry_manifest(candidate, {part.component_id for part in asset.parts})
                self.manifest_label.setText(f"Manifest: VALID — {candidate.name}")
            except GeometryManifestError as exc:
                self.manifest_error = str(exc)
                self.manifest_label.setText(f"Manifest: INVALID — {exc}")
        else:
            self.manifest_label.setText("Manifest: not found; conservative binding inference active")
        self.unit_combo.setCurrentText(source_unit)
        axis_index = self.axis_combo.findData(axis_mode)
        self.axis_combo.setCurrentIndex(max(0, axis_index))
        self._rebuild_scene()
        self.file_label.setText(str(source))
        self.manifest_btn.setEnabled(True)
        self.screenshot_btn.setEnabled(True)

    def restore_camera(self, camera: object) -> None:
        try:
            if isinstance(camera, list) and len(camera) == 3:
                self.plotter.camera_position = tuple(tuple(float(v) for v in point) for point in camera)
                self.plotter.render()
        except Exception:
            pass

    def _apply_result_styles(self, frame) -> None:
        """Result coloring without using opacity as a fake lock-engagement animation."""
        if self.meshes is None or self.bindings is None:
            return
        metric = str(self.metric_combo.currentData())
        color_results = self.color_results_check.isChecked() and self.result_field is not None
        for component_id, actor in self.actors.items():
            record = self.meshes.for_component(component_id)
            binding = self.bindings.for_component(component_id)
            try:
                mapper = actor.GetMapper()
                prop = actor.GetProperty()
                prop.SetOpacity(1.0)
                if color_results and binding.simulation_module is not None:
                    result = component_result(binding, frame, self.result_field)
                    mapper.ScalarVisibilityOff()
                    prop.SetColor(*_result_rgb(metric, result.normalized))
                else:
                    if "face_rgb" in record.polydata.cell_data and str(self.mesh_mode_combo.currentData()) != "wireframe":
                        mapper.ScalarVisibilityOn()
                    else:
                        mapper.ScalarVisibilityOff()
                        prop.SetColor(0.82, 0.82, 0.82)
            except Exception:
                pass
        selected = self.component_combo.currentText()
        if selected:
            self._highlight(selected)

    def _update_force_vectors(self, frame, transforms: dict[str, np.ndarray]) -> None:
        self._remove_overlay("dt_force_vectors")
        if not self.force_check.isChecked() or self.asset is None or self.bindings is None:
            return
        origins: list[np.ndarray] = []
        forces: list[float] = []
        kind = str(self.force_kind_combo.currentData())
        platform_transform = self.timeline.relative_transform(frame, self.bindings.platform_origin_m) if self.timeline is not None else np.eye(4)

        for module_state in frame.modules:
            if self.structural_module_points_m is not None:
                local = np.array([*self.structural_module_points_m[module_state.module - 1], 1.0], dtype=float)
                origin = platform_transform @ local
            else:
                # Fallback is visual-only and explicitly does not claim physical load
                # application accuracy until Structural Validation has supplied M1–M8.
                component_ids = self.bindings.components_for_module(module_state.module)
                component_id = next((name for name in component_ids if self.meshes is not None and name in self.meshes.records), None)
                if component_id is None or self.meshes is None:
                    continue
                centroid = np.array([*self.meshes.for_component(component_id).base_centroid_m, 1.0])
                origin = transforms[component_id] @ centroid
            origins.append(origin[:3])
            forces.append(float(force_vector_n(module_state, kind)[2]))
        if not origins:
            return
        maximum = max(max(np.abs(forces)), 1e-9)
        characteristic = max(float(np.max(self.asset.dimensions_m)), 0.1)
        vectors = np.array([[0.0, 0.0, 0.22 * characteristic * force / maximum] for force in forces], dtype=float)
        self.plotter.add_arrows(np.asarray(origins), vectors, mag=1.0, color="orange", name="dt_force_vectors")

    def _update_cg(self, frame) -> None:
        self._remove_overlay("dt_cg")
        if not self.cg_check.isChecked() or self.asset is None or self.timeline is None or self.bindings is None:
            return
        if self.structural_centroid_m is not None:
            base = self.structural_centroid_m
        else:
            base = np.asarray(self.bindings.platform_origin_m, dtype=float)
        local = np.array([base[0] + frame.cg_shift_x_m, base[1] + frame.cg_shift_y_m, base[2], 1.0])
        matrix = self.timeline.relative_transform(frame, self.bindings.platform_origin_m)
        world = matrix @ local
        radius = max(float(np.max(self.asset.dimensions_m)) * 0.012, 0.002)
        self.plotter.add_mesh(pv.Sphere(radius=radius, center=world[:3]), color="magenta", name="dt_cg", pickable=False)

    def _update_info(self) -> None:
        super()._update_info()
        if self.asset is None:
            return
        row = self.info_table.rowCount()
        self.info_table.insertRow(row)
        self.info_table.setItem(row, 0, __import__("PyQt6.QtWidgets", fromlist=["QTableWidgetItem"]).QTableWidgetItem("Structural context"))
        self.info_table.setItem(row, 1, __import__("PyQt6.QtWidgets", fromlist=["QTableWidgetItem"]).QTableWidgetItem("VALIDATED" if self.structural_context_validated else "Not bound — visual fallback only"))
        self.info_table.resizeColumnsToContents()
