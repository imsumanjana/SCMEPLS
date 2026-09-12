from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import trimesh

SUPPORTED_EXTENSIONS = {".glb", ".stl"}
_UNIT_TO_METRE = {"m": 1.0, "mm": 1e-3, "cm": 1e-2}
_AXIS_MODES = {"auto", "as_stored", "gltf_y_up_to_z_up"}


class GeometryImportError(ValueError):
    """Raised when an external geometry file cannot be used safely."""


@dataclass(frozen=True)
class GeometryPart:
    component_id: str
    vertices_m: np.ndarray
    faces: np.ndarray
    cell_rgb: np.ndarray | None

    @property
    def n_vertices(self) -> int:
        return int(self.vertices_m.shape[0])

    @property
    def n_faces(self) -> int:
        return int(self.faces.shape[0])

    @property
    def bounds_m(self) -> np.ndarray:
        return np.vstack((np.min(self.vertices_m, axis=0), np.max(self.vertices_m, axis=0)))


@dataclass(frozen=True)
class GeometryAsset:
    source_path: Path
    source_unit: str
    axis_mode: str
    scale_to_m: float
    parts: tuple[GeometryPart, ...]
    warnings: tuple[str, ...]

    @property
    def bounds_m(self) -> np.ndarray:
        mins = np.vstack([p.bounds_m[0] for p in self.parts])
        maxs = np.vstack([p.bounds_m[1] for p in self.parts])
        return np.vstack((np.min(mins, axis=0), np.max(maxs, axis=0)))

    @property
    def dimensions_m(self) -> np.ndarray:
        bounds = self.bounds_m
        return bounds[1] - bounds[0]

    @property
    def triangle_count(self) -> int:
        return int(sum(p.n_faces for p in self.parts))


def _safe_component_id(raw: str, index: int) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw.strip())
    return text.strip("_.-") or f"part_{index:03d}"


def _scene_from_path(path: Path) -> trimesh.Scene:
    try:
        if path.suffix.lower() == ".stl":
            # STL stores independent triangle vertices and has no topology graph.
            # Processing is required here to merge coincident vertices so a genuinely
            # closed exported solid is recognized as watertight for structural meshing.
            loaded = trimesh.load_mesh(path, process=True)
            return loaded if isinstance(loaded, trimesh.Scene) else trimesh.Scene(loaded)
        # Preserve GLB scene hierarchy/named nodes; structural validation is applied
        # later to the explicitly selected part.
        return trimesh.load_scene(path, process=False)
    except Exception as exc:  # pragma: no cover - backend-specific details
        raise GeometryImportError(f"Could not read geometry file: {exc}") from exc


def _face_rgb(mesh: trimesh.Trimesh) -> np.ndarray | None:
    try:
        visual = mesh.visual.to_color()
        colors = np.asarray(visual.face_colors)
        if colors.ndim == 2 and colors.shape[0] == len(mesh.faces) and colors.shape[1] >= 3:
            return np.ascontiguousarray(colors[:, :3], dtype=np.uint8)
    except Exception:
        pass
    return None


def _validate_topology(component_id: str, vertices: np.ndarray, faces: np.ndarray, mesh: trimesh.Trimesh) -> list[str]:
    warnings: list[str] = []
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise GeometryImportError(f"Component '{component_id}' contains out-of-range face indices.")

    repeated_vertex_faces = np.any(
        np.column_stack((faces[:, 0] == faces[:, 1], faces[:, 1] == faces[:, 2], faces[:, 0] == faces[:, 2])),
        axis=1,
    )
    repeated_count = int(np.count_nonzero(repeated_vertex_faces))
    if repeated_count:
        warnings.append(f"{component_id}: {repeated_count} degenerate triangle(s) contain repeated vertex indices.")

    try:
        areas = np.asarray(mesh.area_faces, dtype=float)
        zero_area = int(np.count_nonzero(~np.isfinite(areas) | (areas <= 1e-15)))
        if zero_area:
            warnings.append(f"{component_id}: {zero_area} zero/invalid-area triangle(s) detected.")
    except Exception:
        warnings.append(f"{component_id}: triangle-area validation could not be completed.")

    sorted_faces = np.sort(faces, axis=1)
    if len(sorted_faces):
        duplicate_count = len(sorted_faces) - len(np.unique(sorted_faces, axis=0))
        if duplicate_count:
            warnings.append(f"{component_id}: {duplicate_count} duplicate triangle(s) detected.")

    try:
        if not bool(mesh.is_winding_consistent):
            warnings.append(f"{component_id}: face winding is not consistent.")
    except Exception:
        warnings.append(f"{component_id}: face-winding validation could not be completed.")

    try:
        if not bool(mesh.is_watertight):
            warnings.append(f"{component_id}: mesh is not watertight (acceptable for visualization, verify for FEA use).")
    except Exception:
        warnings.append(f"{component_id}: watertightness validation could not be completed.")

    try:
        body_count = int(mesh.body_count)
        if body_count > 1:
            warnings.append(f"{component_id}: mesh contains {body_count} disconnected bodies.")
    except Exception:
        pass

    return warnings


def load_geometry(path: str | Path, source_unit: str = "m", axis_mode: str = "auto") -> GeometryAsset:
    """Load a GLB or STL file and normalize geometry coordinates to SI metres.

    This loader is shared by visualization and the structural FEA pipeline so both
    consume the identical coordinates, scale, and axis convention. Loading a file
    alone does not mutate SC-MEPLS dynamics; geometry-derived mass/inertia/dimensions
    are coupled explicitly by the FEA mass-properties workflow.

    STL carries no unit metadata, so ``source_unit`` is authoritative. In ``auto``
    axis mode, GLB/glTF Y-up geometry is rotated into SC-MEPLS Z-up; STL coordinates
    are kept as stored.
    """

    source = Path(path).expanduser().resolve()
    if not source.exists() or not source.is_file():
        raise GeometryImportError(f"Geometry file does not exist: {source}")
    extension = source.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise GeometryImportError("Only .glb and .stl geometry files are supported.")
    if source_unit not in _UNIT_TO_METRE:
        raise GeometryImportError(f"Unsupported geometry unit: {source_unit}")
    if axis_mode not in _AXIS_MODES:
        raise GeometryImportError(f"Unsupported axis mode: {axis_mode}")

    scale = _UNIT_TO_METRE[source_unit]
    effective_axis_mode = (
        "gltf_y_up_to_z_up"
        if axis_mode == "auto" and extension == ".glb"
        else ("as_stored" if axis_mode == "auto" else axis_mode)
    )
    axis_transform = np.eye(4)
    if effective_axis_mode == "gltf_y_up_to_z_up":
        axis_transform[:3, :3] = np.array(
            [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
        )

    scene = _scene_from_path(source)
    nodes = list(scene.graph.nodes_geometry)
    if not nodes:
        raise GeometryImportError("The geometry file contains no renderable mesh nodes.")

    warnings: list[str] = []
    if extension == ".stl":
        warnings.append(f"STL is unitless; coordinates were interpreted as {source_unit}.")
    elif source_unit != "m":
        warnings.append(
            "glTF/GLB convention is metres; a non-metre override was applied. Verify that this file was intentionally exported with nonstandard units."
        )

    parts: list[GeometryPart] = []
    used_ids: dict[str, int] = {}
    for index, node_name in enumerate(nodes, start=1):
        transform, geometry_name = scene.graph[node_name]
        if geometry_name is None or geometry_name not in scene.geometry:
            continue
        mesh = scene.geometry[geometry_name].copy()
        if not isinstance(mesh, trimesh.Trimesh):
            continue
        mesh.apply_transform(transform)
        mesh.apply_transform(axis_transform)

        vertices = np.asarray(mesh.vertices, dtype=float) * scale
        faces = np.asarray(mesh.faces, dtype=np.int64)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
            warnings.append(f"Skipped node '{node_name}': invalid or empty vertices.")
            continue
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            warnings.append(f"Skipped node '{node_name}': only triangular surface meshes are supported.")
            continue
        if not np.all(np.isfinite(vertices)):
            raise GeometryImportError(f"Node '{node_name}' contains non-finite vertex coordinates.")

        base_id = _safe_component_id(str(node_name or geometry_name), index)
        duplicate_index = used_ids.get(base_id, 0)
        used_ids[base_id] = duplicate_index + 1
        component_id = base_id if duplicate_index == 0 else f"{base_id}_{duplicate_index + 1}"
        if duplicate_index:
            warnings.append(f"Duplicate component ID '{base_id}' was renamed to '{component_id}'.")

        warnings.extend(_validate_topology(component_id, vertices, faces, mesh))
        parts.append(
            GeometryPart(
                component_id=component_id,
                vertices_m=np.ascontiguousarray(vertices),
                faces=np.ascontiguousarray(faces),
                cell_rgb=_face_rgb(mesh),
            )
        )

    if not parts:
        raise GeometryImportError("No valid triangular mesh parts could be imported.")

    asset = GeometryAsset(source, source_unit, effective_axis_mode, scale, tuple(parts), tuple(warnings))
    dims = asset.dimensions_m
    if np.any(~np.isfinite(dims)) or np.any(dims <= 0):
        raise GeometryImportError("Imported geometry has an invalid bounding-box dimension.")
    if float(np.max(dims)) > 1000.0:
        warnings.append("Geometry exceeds 1000 m in at least one dimension; verify selected units/scale.")
    if float(np.max(dims)) < 1e-4:
        warnings.append("Geometry is smaller than 0.1 mm overall; verify selected units/scale.")
    if asset.triangle_count > 1_000_000:
        warnings.append("Geometry exceeds 1,000,000 triangles and may render slowly.")

    if tuple(warnings) != asset.warnings:
        asset = GeometryAsset(source, source_unit, effective_axis_mode, scale, tuple(parts), tuple(warnings))
    return asset
