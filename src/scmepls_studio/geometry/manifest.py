from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


class GeometryManifestError(ValueError):
    """Raised when a geometry sidecar manifest is malformed or inconsistent."""


_ALLOWED_DYNAMIC_GROUPS = {"world", "platform", "module", "lock"}


def _vector3(value: Any, name: str, *, allow_none: bool = False) -> tuple[float, float, float] | None:
    if value is None and allow_none:
        return None
    if value is None:
        return (0.0, 0.0, 0.0)
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise GeometryManifestError(f"Manifest '{name}' must be a three-value array [x, y, z].")
    try:
        vector = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise GeometryManifestError(f"Manifest '{name}' must contain numeric values.") from exc
    if not all(abs(v) < float("inf") for v in vector):
        raise GeometryManifestError(f"Manifest '{name}' must contain finite values.")
    return vector  # type: ignore[return-value]


@dataclass(frozen=True)
class ComponentBinding:
    component_id: str
    role: str = "visual"
    dynamic_group: str = "world"
    simulation_module: int | None = None
    pivot_m: tuple[float, float, float] | None = None


@dataclass(frozen=True)
class GeometryManifest:
    path: Path
    schema_version: int
    components: dict[str, ComponentBinding]
    required_components: tuple[str, ...]
    platform_origin_m: tuple[float, float, float] = (0.0, 0.0, 0.0)


def manifest_path_for_geometry(geometry_path: str | Path) -> Path:
    source = Path(geometry_path)
    return source.with_suffix(".manifest.json")


def load_geometry_manifest(path: str | Path, available_component_ids: set[str] | None = None) -> GeometryManifest:
    source = Path(path)
    if not source.exists():
        raise GeometryManifestError(f"Geometry manifest does not exist: {source}")
    try:
        raw: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    except Exception as exc:
        raise GeometryManifestError(f"Could not read geometry manifest: {exc}") from exc

    schema_version = int(raw.get("schema_version", 1))
    if schema_version != 1:
        raise GeometryManifestError(f"Unsupported geometry manifest schema_version: {schema_version}")

    platform_origin = _vector3(raw.get("platform_origin_m"), "platform_origin_m")
    assert platform_origin is not None

    raw_components = raw.get("components", {})
    if not isinstance(raw_components, dict):
        raise GeometryManifestError("Manifest 'components' must be an object keyed by imported component ID.")

    components: dict[str, ComponentBinding] = {}
    for component_id, spec in raw_components.items():
        if not isinstance(component_id, str) or not component_id.strip():
            raise GeometryManifestError("Manifest component IDs must be non-empty strings.")
        if spec is None:
            spec = {}
        if not isinstance(spec, dict):
            raise GeometryManifestError(f"Manifest entry '{component_id}' must be an object.")

        simulation_module = spec.get("simulation_module")
        if simulation_module is not None:
            simulation_module = int(simulation_module)
            if not 1 <= simulation_module <= 8:
                raise GeometryManifestError(
                    f"Manifest entry '{component_id}' simulation_module must be between 1 and 8."
                )

        dynamic_group = str(spec.get("dynamic_group", "world")).strip().lower()
        if dynamic_group not in _ALLOWED_DYNAMIC_GROUPS:
            raise GeometryManifestError(
                f"Manifest entry '{component_id}' dynamic_group must be one of {sorted(_ALLOWED_DYNAMIC_GROUPS)}."
            )
        pivot = _vector3(spec.get("pivot_m"), f"components.{component_id}.pivot_m", allow_none=True)
        components[component_id] = ComponentBinding(
            component_id=component_id,
            role=str(spec.get("role", "visual")).strip() or "visual",
            dynamic_group=dynamic_group,
            simulation_module=simulation_module,
            pivot_m=pivot,
        )

    raw_required = raw.get("required_components", [])
    if not isinstance(raw_required, list) or not all(isinstance(x, str) for x in raw_required):
        raise GeometryManifestError("Manifest 'required_components' must be a list of component IDs.")
    required = tuple(raw_required)

    if available_component_ids is not None:
        unknown = sorted(set(components) - available_component_ids)
        if unknown:
            raise GeometryManifestError(f"Manifest references components not found in geometry: {unknown}")
        missing_required = sorted(set(required) - available_component_ids)
        if missing_required:
            raise GeometryManifestError(f"Required geometry components are missing: {missing_required}")

    return GeometryManifest(
        path=source,
        schema_version=schema_version,
        components=components,
        required_components=required,
        platform_origin_m=platform_origin,
    )
