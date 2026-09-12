from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pyvista as pv

from ..geometry import GeometryAsset, GeometryPart
from .binding import SceneBinding, SceneBindingRegistry

MeshDisplayMode = Literal["surface", "surface_edges", "wireframe"]


@dataclass
class MeshRecord:
    component_id: str
    component_index: int
    polydata: pv.PolyData
    binding: SceneBinding
    base_centroid_m: tuple[float, float, float]
    base_bounds_m: tuple[float, float, float, float, float, float]


@dataclass
class MeshRegistry:
    records: dict[str, MeshRecord]
    component_by_index: dict[int, str]

    @classmethod
    def from_asset(cls, asset: GeometryAsset, bindings: SceneBindingRegistry) -> "MeshRegistry":
        records: dict[str, MeshRecord] = {}
        component_by_index: dict[int, str] = {}
        for index, part in enumerate(asset.parts, start=1):
            if part.component_id in records:
                raise ValueError(f"Duplicate component ID in mesh registry: {part.component_id}")
            if part.component_id not in bindings.bindings:
                raise ValueError(f"Missing scene binding for geometry component: {part.component_id}")
            poly = polydata_from_part(part, index)
            centroid = tuple(float(v) for v in np.mean(part.vertices_m, axis=0))
            bounds = tuple(float(v) for v in poly.bounds)
            records[part.component_id] = MeshRecord(
                component_id=part.component_id,
                component_index=index,
                polydata=poly,
                binding=bindings.for_component(part.component_id),
                base_centroid_m=centroid,
                base_bounds_m=bounds,
            )
            component_by_index[index] = part.component_id
        return cls(records=records, component_by_index=component_by_index)

    def for_component(self, component_id: str) -> MeshRecord:
        return self.records[component_id]

    @property
    def triangle_count(self) -> int:
        return int(sum(record.polydata.n_cells for record in self.records.values()))

    @property
    def point_count(self) -> int:
        return int(sum(record.polydata.n_points for record in self.records.values()))


def polydata_from_part(part: GeometryPart, component_index: int) -> pv.PolyData:
    """Convert immutable imported triangles to an exact PyVista surface mesh."""
    vtk_faces = np.column_stack((np.full(part.n_faces, 3, dtype=np.int64), part.faces)).ravel()
    poly = pv.PolyData(np.asarray(part.vertices_m, dtype=float), vtk_faces)
    poly.field_data["scmepls_component_index"] = np.array([component_index], dtype=np.int32)
    poly.field_data["scmepls_component_id"] = np.array([part.component_id])
    if part.cell_rgb is not None and len(part.cell_rgb) == part.n_faces:
        poly.cell_data["face_rgb"] = np.asarray(part.cell_rgb, dtype=np.uint8)
    return poly


def actor_style(record: MeshRecord, mode: MeshDisplayMode = "surface") -> dict[str, object]:
    """Return renderer kwargs without modifying the engineering mesh."""
    if mode not in {"surface", "surface_edges", "wireframe"}:
        raise ValueError(f"Unsupported mesh display mode: {mode}")
    style: dict[str, object] = {
        "name": record.component_id,
        "pickable": True,
        "smooth_shading": mode != "wireframe",
    }
    if mode == "wireframe":
        style.update({"style": "wireframe", "color": "dimgray", "line_width": 1.0})
        return style
    style["show_edges"] = mode == "surface_edges"
    if "face_rgb" in record.polydata.cell_data:
        style.update({"scalars": "face_rgb", "rgb": True, "show_scalar_bar": False})
    else:
        style["color"] = "lightgray"
    return style
