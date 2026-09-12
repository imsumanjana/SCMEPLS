from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .binding import SceneBinding
from .physics import DigitalTwinFrame, ModuleState, SimulationTimeline

ResultMetric = Literal["gap", "current", "em_force", "pressure", "pneumatic_force", "health"]


@dataclass(frozen=True)
class ResultRange:
    minimum: float
    maximum: float

    def normalize(self, value: float) -> float:
        if not np.isfinite(value):
            return 0.0
        span = self.maximum - self.minimum
        if span <= 1e-15:
            return 0.0
        return float(np.clip((value - self.minimum) / span, 0.0, 1.0))


@dataclass(frozen=True)
class ComponentResult:
    component_id: str
    module: int | None
    label: str
    value: float
    unit: str
    normalized: float
    health: float
    em_force_n: float
    pneumatic_force_n: float


_METRIC_COLUMN = {
    "gap": "gap_{n}_m",
    "current": "coil_current_{n}_a",
    "em_force": "em_force_{n}_n",
    "pressure": "pressure_{n}_pa",
    "pneumatic_force": "pneumatic_force_{n}_n",
    "health": "health_{n}",
}
_METRIC_LABEL_UNIT = {
    "gap": ("Air gap", "m"),
    "current": ("Coil current", "A"),
    "em_force": ("EM force", "N"),
    "pressure": ("Pneumatic pressure", "Pa"),
    "pneumatic_force": ("Pneumatic force", "N"),
    "health": ("Health", "p.u."),
}


class ResultField:
    """Global scale and frame extraction for module-resolved result visualization."""

    def __init__(self, timeline: SimulationTimeline, metric: ResultMetric) -> None:
        if metric not in _METRIC_COLUMN:
            raise ValueError(f"Unsupported digital-twin result metric: {metric}")
        self.timeline = timeline
        self.metric = metric
        values: list[float] = []
        template = _METRIC_COLUMN[metric]
        for module in range(1, 9):
            column = template.format(n=module)
            if column in timeline.history.columns:
                data = timeline.history[column].to_numpy(dtype=float)
                values.extend(data[np.isfinite(data)].tolist())
        if values:
            minimum = float(np.min(values))
            maximum = float(np.max(values))
        else:
            minimum, maximum = 0.0, 1.0
        if metric == "health":
            minimum = min(minimum, 0.0)
            maximum = max(maximum, 1.0)
        self.range = ResultRange(minimum, maximum)

    @property
    def label(self) -> str:
        return _METRIC_LABEL_UNIT[self.metric][0]

    @property
    def unit(self) -> str:
        return _METRIC_LABEL_UNIT[self.metric][1]

    def value(self, module_state: ModuleState) -> float:
        return {
            "gap": module_state.true_gap_m,
            "current": module_state.coil_current_a,
            "em_force": module_state.em_force_n,
            "pressure": module_state.pressure_pa,
            "pneumatic_force": module_state.pneumatic_force_n,
            "health": module_state.health,
        }[self.metric]

    def normalized(self, module_state: ModuleState) -> float:
        return self.range.normalize(self.value(module_state))


def default_metric_for_binding(binding: SceneBinding) -> ResultMetric:
    role = binding.role.lower()
    if "pneu" in role:
        return "pneumatic_force"
    if "electro" in role or "magnet" in role or "coil" in role:
        return "em_force"
    return "health"


def component_result(
    binding: SceneBinding,
    frame: DigitalTwinFrame,
    field: ResultField,
) -> ComponentResult:
    if binding.simulation_module is None:
        return ComponentResult(
            component_id=binding.component_id,
            module=None,
            label=field.label,
            value=0.0,
            unit=field.unit,
            normalized=0.0,
            health=1.0,
            em_force_n=0.0,
            pneumatic_force_n=0.0,
        )
    module_state = frame.modules[binding.simulation_module - 1]
    value = field.value(module_state)
    return ComponentResult(
        component_id=binding.component_id,
        module=binding.simulation_module,
        label=field.label,
        value=value,
        unit=field.unit,
        normalized=field.normalized(module_state),
        health=module_state.health,
        em_force_n=module_state.em_force_n,
        pneumatic_force_n=module_state.pneumatic_force_n,
    )


def force_vector_n(module_state: ModuleState, kind: Literal["em", "pneumatic", "total"] = "total") -> np.ndarray:
    if kind == "em":
        force = module_state.em_force_n
    elif kind == "pneumatic":
        force = module_state.pneumatic_force_n
    elif kind == "total":
        force = module_state.em_force_n + module_state.pneumatic_force_n
    else:
        raise ValueError(f"Unsupported force-vector kind: {kind}")
    return np.array([0.0, 0.0, float(force)], dtype=float)
