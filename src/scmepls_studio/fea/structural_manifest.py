from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .coupling import StructuralLoadMap
from .mesher import MeshingOptions
from .solver import IsotropicMaterial


class StructuralManifestError(ValueError):
    """Raised when a structural-analysis sidecar is incomplete or unsafe."""


@dataclass(frozen=True)
class StructuralManifest:
    path: Path
    schema_version: int
    structural_component: str | None
    mass_scope: str
    material: IsotropicMaterial
    meshing: MeshingOptions
    load_map: StructuralLoadMap
    maximum_snap_distance_m: float


def structural_manifest_path_for_geometry(path: str | Path) -> Path:
    source = Path(path)
    return source.with_suffix(".fea.json")


def _array(raw: Any, name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    try:
        array = np.asarray(raw, dtype=float)
    except Exception as exc:
        raise StructuralManifestError(f"'{name}' must contain numeric coordinates.") from exc
    if shape is not None and array.shape != shape:
        raise StructuralManifestError(f"'{name}' must have shape {shape}; got {array.shape}.")
    if array.ndim < 1 or not np.all(np.isfinite(array)):
        raise StructuralManifestError(f"'{name}' must contain finite values.")
    return array


def load_structural_manifest(path: str | Path) -> StructuralManifest:
    source = Path(path)
    if not source.exists() or not source.is_file():
        raise StructuralManifestError(f"Structural manifest does not exist: {source}")
    try:
        raw: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise StructuralManifestError(f"Could not read structural manifest: {exc}") from exc
    if not isinstance(raw, dict):
        raise StructuralManifestError("Structural manifest root must be a JSON object.")

    schema_version = int(raw.get("schema_version", 1))
    if schema_version != 1:
        raise StructuralManifestError(f"Unsupported structural schema_version: {schema_version}")
    structural_component = raw.get("structural_component")
    if structural_component is not None:
        structural_component = str(structural_component).strip() or None
    mass_scope = str(raw.get("mass_scope", "moving_assembly")).strip().lower()
    if mass_scope not in {"moving_assembly", "platform_only"}:
        raise StructuralManifestError("mass_scope must be 'moving_assembly' or 'platform_only'.")

    m = raw.get("material", {})
    if not isinstance(m, dict):
        raise StructuralManifestError("'material' must be a JSON object.")
    try:
        material = IsotropicMaterial(
            youngs_modulus_pa=float(m.get("youngs_modulus_pa", 69e9)),
            poisson_ratio=float(m.get("poisson_ratio", 0.33)),
            density_kg_m3=float(m.get("density_kg_m3", 2700.0)),
            yield_strength_pa=float(m.get("yield_strength_pa", 240e6)),
        )
    except (TypeError, ValueError) as exc:
        raise StructuralManifestError(f"Invalid material definition: {exc}") from exc

    mesh = raw.get("mesh", {})
    if not isinstance(mesh, dict):
        raise StructuralManifestError("'mesh' must be a JSON object.")
    try:
        meshing = MeshingOptions(
            element_size_m=float(mesh.get("element_size_m", 0.025)),
            min_element_size_m=(
                None if mesh.get("min_element_size_m") is None else float(mesh["min_element_size_m"])
            ),
            max_element_size_m=(
                None if mesh.get("max_element_size_m") is None else float(mesh["max_element_size_m"])
            ),
            optimize=bool(mesh.get("optimize", True)),
            classify_angle_deg=float(mesh.get("classify_angle_deg", 40.0)),
        )
    except (TypeError, ValueError) as exc:
        raise StructuralManifestError(f"Invalid mesh definition: {exc}") from exc

    mapping = raw.get("mapping")
    if not isinstance(mapping, dict):
        raise StructuralManifestError(
            "A validated structural manifest requires a 'mapping' object with explicit module, constraint, wind, propulsion, and lock points."
        )
    module_points = _array(mapping.get("module_points_m"), "mapping.module_points_m", (8, 3))
    constraint_points = _array(mapping.get("constraint_points_m"), "mapping.constraint_points_m", (3, 3))
    propulsion_point = _array(mapping.get("propulsion_point_m"), "mapping.propulsion_point_m", (3,))
    wind_point = _array(mapping.get("wind_point_m"), "mapping.wind_point_m", (3,))
    lock_points = _array(mapping.get("lock_points_m"), "mapping.lock_points_m")
    if lock_points.ndim != 2 or lock_points.shape[1] != 3 or len(lock_points) < 1:
        raise StructuralManifestError("'mapping.lock_points_m' must have shape (N, 3) with N >= 1.")
    load_map = StructuralLoadMap(
        module_points_m=module_points,
        constraint_points_m=constraint_points,
        propulsion_point_m=propulsion_point,
        wind_point_m=wind_point,
        lock_points_m=lock_points,
        source="structural_manifest",
    )

    maximum_snap_distance_m = float(raw.get("maximum_snap_distance_m", 0.02))
    if not np.isfinite(maximum_snap_distance_m) or maximum_snap_distance_m <= 0:
        raise StructuralManifestError("maximum_snap_distance_m must be finite and positive.")
    return StructuralManifest(
        path=source,
        schema_version=schema_version,
        structural_component=structural_component,
        mass_scope=mass_scope,
        material=material,
        meshing=meshing,
        load_map=load_map,
        maximum_snap_distance_m=maximum_snap_distance_m,
    )
