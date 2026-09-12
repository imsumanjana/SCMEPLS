from __future__ import annotations

from dataclasses import dataclass
import re

from ..geometry import GeometryAsset, GeometryManifest


@dataclass(frozen=True)
class SceneBinding:
    """Semantic link between one imported component and the digital-twin scene.

    ``source`` remains the sixth positional field for backward compatibility with
    the v1.2/v1.3 public helper API. New actuator-motion fields therefore have
    defaults and follow it.
    """

    component_id: str
    role: str
    dynamic_group: str
    simulation_module: int | None
    pivot_m: tuple[float, float, float]
    source: str
    motion_axis: tuple[float, float, float] | None = None
    stroke_m: float = 0.0
    motion_signal: str = "none"


@dataclass(frozen=True)
class SceneBindingRegistry:
    bindings: dict[str, SceneBinding]
    platform_origin_m: tuple[float, float, float]

    def for_component(self, component_id: str) -> SceneBinding:
        return self.bindings[component_id]

    @property
    def platform_components(self) -> tuple[str, ...]:
        return tuple(name for name,binding in self.bindings.items() if binding.dynamic_group in {"platform","module"})

    @property
    def world_components(self) -> tuple[str, ...]:
        return tuple(name for name,binding in self.bindings.items() if binding.dynamic_group in {"world","lock"})

    def components_for_module(self,module:int)->tuple[str,...]:
        return tuple(name for name,binding in self.bindings.items() if binding.simulation_module==module)


def _infer_module(component_id:str)->int|None:
    upper=component_id.upper();patterns=(r"(?:EM|PNEU|MODULE)[_-]?M?0?([1-8])(?:\D|$)",r"(?:^|\D)M0?([1-8])(?:\D|$)")
    for pattern in patterns:
        match=re.search(pattern,upper)
        if match:return int(match.group(1))
    return None


def _infer_role_and_group(component_id:str)->tuple[str,str]:
    upper=component_id.upper()
    if any(token in upper for token in ("TRACK","RAIL","GUIDEWAY","GROUND")):return "guideway","world"
    if "LOCK" in upper or "CLAMP" in upper or "PIN" in upper:return "lock","lock"
    if "PNEU" in upper or "CYLINDER" in upper:return "pneumatic","module" if _infer_module(component_id) else "platform"
    if "EM_" in upper or "MAGNET" in upper or "COIL" in upper:return "electromagnetic","module" if _infer_module(component_id) else "platform"
    if any(token in upper for token in ("PLATFORM","CHASSIS","DECK","ROCKET","CRADLE","TOWER","FLANGE")):return "structure","platform"
    return "visual","world"


def build_scene_bindings(asset:GeometryAsset,manifest:GeometryManifest|None=None)->SceneBindingRegistry:
    """Build deterministic geometry-to-scene bindings with conservative defaults."""
    platform_origin=manifest.platform_origin_m if manifest is not None else (0.0,0.0,0.0);bindings:dict[str,SceneBinding]={}
    for part in asset.parts:
        declared=manifest.components.get(part.component_id) if manifest is not None else None
        if declared is not None:
            pivot=declared.pivot_m or platform_origin;binding=SceneBinding(component_id=part.component_id,role=declared.role,dynamic_group=declared.dynamic_group,simulation_module=declared.simulation_module,pivot_m=pivot,source="manifest",motion_axis=declared.motion_axis,stroke_m=float(declared.stroke_m),motion_signal=declared.motion_signal)
        else:
            role,group=_infer_role_and_group(part.component_id);module=_infer_module(part.component_id);pivot=platform_origin if group in {"platform","module"} else (0.0,0.0,0.0);binding=SceneBinding(part.component_id,role,group,module,pivot,"inferred")
        bindings[part.component_id]=binding
    return SceneBindingRegistry(bindings=bindings,platform_origin_m=platform_origin)
