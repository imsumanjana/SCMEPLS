from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tempfile

import numpy as np
import trimesh

from scmepls_studio.geometry import GeometryAsset, GeometryImportError, GeometryPart, load_geometry

from .mesh import TetraMesh, TetraMeshError


class VolumeMeshingError(RuntimeError):
    """Raised when a closed surface cannot be converted to a valid volume mesh."""


@dataclass(frozen=True)
class MeshingOptions:
    element_size_m: float = 0.025
    min_element_size_m: float | None = None
    max_element_size_m: float | None = None
    optimize: bool = True
    classify_angle_deg: float = 40.0

    def __post_init__(self) -> None:
        values = [self.element_size_m]
        if self.min_element_size_m is not None:
            values.append(self.min_element_size_m)
        if self.max_element_size_m is not None:
            values.append(self.max_element_size_m)
        if not all(np.isfinite(v) and v > 0 for v in values):
            raise ValueError("Finite positive mesh sizes are required.")
        if not 1.0 <= float(self.classify_angle_deg) <= 179.0:
            raise ValueError("classify_angle_deg must be between 1 and 179 degrees.")


def select_structural_part(asset: GeometryAsset, component_id: str | None = None) -> GeometryPart:
    """Select exactly one watertight imported body for structural meshing."""
    if component_id is None:
        if len(asset.parts) != 1:
            names = ", ".join(part.component_id for part in asset.parts)
            raise VolumeMeshingError(
                "Multi-part GLB assemblies require an explicit structural component. "
                f"Available parts: {names}"
            )
        part = asset.parts[0]
    else:
        matches = [part for part in asset.parts if part.component_id == component_id]
        if not matches:
            raise VolumeMeshingError(f"Structural component '{component_id}' was not found.")
        part = matches[0]

    tri = trimesh.Trimesh(vertices=part.vertices_m, faces=part.faces, process=False)
    if not bool(tri.is_watertight):
        raise VolumeMeshingError(
            f"Structural component '{part.component_id}' is not watertight; "
            "a closed manifold surface is required for 3-D FEA meshing."
        )
    if not np.isfinite(float(abs(tri.volume))) or abs(float(tri.volume)) <= 1e-15:
        raise VolumeMeshingError(f"Structural component '{part.component_id}' encloses no usable volume.")
    return part


def _gmsh_tetrahedralize(part: GeometryPart, options: MeshingOptions) -> TetraMesh:
    try:
        import gmsh
    except Exception as exc:  # pragma: no cover - platform package loading
        raise VolumeMeshingError(
            "Gmsh is required for boundary-conforming tetrahedralization. "
            "Install the project dependencies and ensure the Gmsh runtime can load."
        ) from exc

    tri = trimesh.Trimesh(vertices=part.vertices_m, faces=part.faces, process=False)
    gmsh.initialize()
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("scmepls_fea")
        with tempfile.TemporaryDirectory(prefix="scmepls_gmsh_") as directory:
            stl_path = Path(directory) / "structural_surface.stl"
            tri.export(stl_path)
            gmsh.merge(str(stl_path))

            angle = np.deg2rad(float(options.classify_angle_deg))
            # Gmsh 4.13-4.15 Python bindings expose these arguments as
            # (angle, boundary, forReparametrization, curveAngle, exportDiscrete).
            # Positional arguments keep compatibility across the supported releases.
            gmsh.model.mesh.classifySurfaces(angle, True, True, np.pi, True)
            gmsh.model.mesh.createGeometry()
            gmsh.model.geo.synchronize()
            surfaces = [tag for dim, tag in gmsh.model.getEntities(2) if dim == 2]
            if not surfaces:
                raise VolumeMeshingError("Gmsh reconstructed no closed surfaces from the imported body.")
            loop = gmsh.model.geo.addSurfaceLoop(surfaces)
            gmsh.model.geo.addVolume([loop])
            gmsh.model.geo.synchronize()

            size_min = (
                float(options.min_element_size_m)
                if options.min_element_size_m is not None
                else 0.5 * float(options.element_size_m)
            )
            size_max = (
                float(options.max_element_size_m)
                if options.max_element_size_m is not None
                else float(options.element_size_m)
            )
            gmsh.option.setNumber("Mesh.MeshSizeMin", size_min)
            gmsh.option.setNumber("Mesh.MeshSizeMax", size_max)
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
            gmsh.model.mesh.generate(3)
            if options.optimize:
                try:
                    gmsh.model.mesh.optimize("Netgen")
                except Exception:
                    gmsh.model.mesh.optimize()

            node_tags, coords, _ = gmsh.model.mesh.getNodes()
            node_tags = np.asarray(node_tags, dtype=np.int64)
            nodes = np.asarray(coords, dtype=float).reshape(-1, 3)
            if len(nodes) < 4:
                raise VolumeMeshingError("Gmsh returned fewer than four volume nodes.")
            tag_to_index = {int(tag): i for i, tag in enumerate(node_tags.tolist())}

            element_types, _, element_node_tags = gmsh.model.mesh.getElements(3)
            tetra_tags = None
            for etype, flat_nodes in zip(element_types, element_node_tags):
                props = gmsh.model.mesh.getElementProperties(int(etype))
                _, dim, order, node_count, _, _ = props
                if dim == 3 and order == 1 and node_count == 4:
                    tetra_tags = np.asarray(flat_nodes, dtype=np.int64).reshape(-1, 4)
                    break
            if tetra_tags is None or len(tetra_tags) == 0:
                raise VolumeMeshingError("Gmsh produced no first-order tetrahedral elements.")
            try:
                tets = np.vectorize(tag_to_index.__getitem__, otypes=[np.int64])(tetra_tags)
            except KeyError as exc:
                raise VolumeMeshingError("Gmsh element connectivity references an unknown node tag.") from exc
            return TetraMesh(nodes, tets, component_id=part.component_id)
    except (VolumeMeshingError, TetraMeshError):
        raise
    except Exception as exc:
        raise VolumeMeshingError(f"Volume meshing failed: {exc}") from exc
    finally:
        gmsh.finalize()


def tetrahedralize_geometry(
    path: str | Path,
    *,
    component_id: str | None = None,
    source_unit: str = "m",
    axis_mode: str = "auto",
    options: MeshingOptions | None = None,
) -> TetraMesh:
    """Create a boundary-conforming tetrahedral mesh from a closed GLB/STL body.

    GLB/STL is first normalized by the same SC-MEPLS geometry importer used by
    visualization, so FEA uses the identical SI-unit coordinates and axis convention.
    """
    try:
        asset = load_geometry(path, source_unit=source_unit, axis_mode=axis_mode)
    except GeometryImportError as exc:
        raise VolumeMeshingError(str(exc)) from exc
    part = select_structural_part(asset, component_id)
    return _gmsh_tetrahedralize(part, options or MeshingOptions())
