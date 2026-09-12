from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from .coupling import StructuralLoadMap
from .mass_properties import PayloadMassProperties
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
    payload: PayloadMassProperties | None = None


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


def _positive(raw: Any, name: str, default: float) -> float:
    try:
        value = float(default if raw is None else raw)
    except (TypeError, ValueError) as exc:
        raise StructuralManifestError(f"'{name}' must be numeric.") from exc
    if not np.isfinite(value) or value <= 0.0:
        raise StructuralManifestError(f"'{name}' must be finite and positive.")
    return value


def _payload(raw: Any, mass_scope: str) -> PayloadMassProperties | None:
    if raw is None:
        if mass_scope == "platform_only":
            raise StructuralManifestError(
                "platform_only structural coupling requires a 'payload' object with mass_kg, centroid_m, and inertia_centroid_kg_m2."
            )
        return None
    if not isinstance(raw, dict):
        raise StructuralManifestError("'payload' must be a JSON object.")
    try:
        mass = float(raw["mass_kg"])
        centroid = _array(raw["centroid_m"], "payload.centroid_m", (3,))
        inertia = _array(raw["inertia_centroid_kg_m2"], "payload.inertia_centroid_kg_m2", (3, 3))
        return PayloadMassProperties(mass, centroid, inertia)
    except KeyError as exc:
        raise StructuralManifestError(f"Missing payload field: {exc.args[0]}") from exc
    except (TypeError, ValueError) as exc:
        raise StructuralManifestError(f"Invalid payload definition: {exc}") from exc


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
    if schema_version not in {1, 2}:
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
            min_element_size_m=(None if mesh.get("min_element_size_m") is None else float(mesh["min_element_size_m"])),
            max_element_size_m=(None if mesh.get("max_element_size_m") is None else float(mesh["max_element_size_m"])),
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
    payload_support = mapping.get("payload_support_points_m")
    payload_support_points = None
    if payload_support is not None:
        payload_support_points = _array(payload_support, "mapping.payload_support_points_m")
        if payload_support_points.ndim != 2 or payload_support_points.shape[1] != 3 or len(payload_support_points) < 1:
            raise StructuralManifestError("'mapping.payload_support_points_m' must have shape (N, 3) with N >= 1.")

    load_map = StructuralLoadMap(
        module_points_m=module_points,
        constraint_points_m=constraint_points,
        propulsion_point_m=propulsion_point,
        wind_point_m=wind_point,
        lock_points_m=lock_points,
        payload_support_points_m=payload_support_points,
        module_patch_radius_m=_positive(mapping.get("module_patch_radius_m"), "mapping.module_patch_radius_m", 0.025),
        propulsion_patch_radius_m=_positive(mapping.get("propulsion_patch_radius_m"), "mapping.propulsion_patch_radius_m", 0.035),
        wind_patch_radius_m=_positive(mapping.get("wind_patch_radius_m"), "mapping.wind_patch_radius_m", 0.040),
        lock_patch_radius_m=_positive(mapping.get("lock_patch_radius_m"), "mapping.lock_patch_radius_m", 0.025),
        payload_patch_radius_m=_positive(mapping.get("payload_patch_radius_m"), "mapping.payload_patch_radius_m", 0.040),
        source="structural_manifest",
    )

    maximum_snap_distance_m = _positive(raw.get("maximum_snap_distance_m"), "maximum_snap_distance_m", 0.02)
    payload = _payload(raw.get("payload"), mass_scope)
    if payload is not None and payload.mass_kg > 0.0 and payload_support_points is None:
        raise StructuralManifestError(
            "A non-zero payload requires mapping.payload_support_points_m so payload loads can enter the platform through physical support regions."
        )

    return StructuralManifest(
        path=source,
        schema_version=schema_version,
        structural_component=structural_component,
        mass_scope=mass_scope,
        material=material,
        meshing=meshing,
        load_map=load_map,
        maximum_snap_distance_m=maximum_snap_distance_m,
        payload=payload,
    )
