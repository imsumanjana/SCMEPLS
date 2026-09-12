from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

MODULE_COUNT = 8
_REQUIRED_COLUMNS = ("time_s", "x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad")


@dataclass(frozen=True)
class RigidBodyState:
    x_m: float
    y_m: float
    z_m: float
    roll_rad: float
    pitch_rad: float
    yaw_rad: float
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    vz_mps: float = 0.0
    p_radps: float = 0.0
    q_radps: float = 0.0
    r_radps: float = 0.0


@dataclass(frozen=True)
class ModuleState:
    module: int
    true_gap_m: float
    sensor_gap_m: float
    estimated_gap_m: float
    coil_current_a: float
    em_force_n: float
    pressure_pa: float
    pneumatic_force_n: float
    health: float
    coil_efficiency: float


@dataclass(frozen=True)
class DigitalTwinFrame:
    index: int
    time_s: float
    mode: str
    rigid_body: RigidBodyState
    modules: tuple[ModuleState, ...]
    lock_fraction: float
    unsafe: bool
    fault_severity: float
    cg_shift_x_m: float
    cg_shift_y_m: float
    wind_force_n: float


class SimulationTimeline:
    """Read-only bridge from validated SC-MEPLS history to 3D/plot consumers."""

    def __init__(self, history: pd.DataFrame) -> None:
        missing = [name for name in _REQUIRED_COLUMNS if name not in history.columns]
        if missing:
            raise ValueError(f"Simulation history is missing required digital-twin columns: {missing}")
        if len(history) < 2:
            raise ValueError("Simulation history must contain at least two rows.")
        numeric = history[list(_REQUIRED_COLUMNS)].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError("Required digital-twin history columns must contain finite numeric values.")
        time_s = numeric["time_s"].to_numpy(dtype=float)
        if np.any(np.diff(time_s) <= 0):
            raise ValueError("Simulation history time_s must be strictly increasing.")
        self.history = history.reset_index(drop=True).copy()
        self.time_s = time_s
        self.reference = self._rigid_body_from_row(self.history.iloc[0])

    def __len__(self) -> int:
        return len(self.history)

    @property
    def duration_s(self) -> float:
        return float(self.time_s[-1] - self.time_s[0])

    def nearest_index(self, time_s: float) -> int:
        value = float(np.clip(time_s, self.time_s[0], self.time_s[-1]))
        index = int(np.searchsorted(self.time_s, value, side="left"))
        if index <= 0:
            return 0
        if index >= len(self.time_s):
            return len(self.time_s) - 1
        before = index - 1
        return before if value - self.time_s[before] <= self.time_s[index] - value else index

    def frame_at_time(self, time_s: float) -> DigitalTwinFrame:
        return self.frame_at_index(self.nearest_index(time_s))

    def frame_at_index(self, index: int) -> DigitalTwinFrame:
        if not 0 <= int(index) < len(self.history):
            raise IndexError("Digital-twin frame index is out of range.")
        row = self.history.iloc[int(index)]
        modules = tuple(self._module_from_row(row, n) for n in range(1, MODULE_COUNT + 1))
        mode = str(row.get("mode_name", row.get("mode", "")))
        return DigitalTwinFrame(
            index=int(index),
            time_s=float(row["time_s"]),
            mode=mode,
            rigid_body=self._rigid_body_from_row(row),
            modules=modules,
            lock_fraction=self._value(row, "lock_fraction"),
            unsafe=bool(round(self._value(row, "unsafe_flag"))),
            fault_severity=self._value(row, "fault_severity"),
            cg_shift_x_m=self._value(row, "cg_shift_x_m"),
            cg_shift_y_m=self._value(row, "cg_shift_y_m"),
            wind_force_n=self._value(row, "wind_force_n"),
        )

    @staticmethod
    def _value(row: pd.Series, name: str, default: float = 0.0) -> float:
        value = row.get(name, default)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float(default)
        return number if np.isfinite(number) else float(default)

    @classmethod
    def _rigid_body_from_row(cls, row: pd.Series) -> RigidBodyState:
        return RigidBodyState(
            x_m=cls._value(row, "x_m"),
            y_m=cls._value(row, "y_m"),
            z_m=cls._value(row, "z_m"),
            roll_rad=cls._value(row, "roll_rad"),
            pitch_rad=cls._value(row, "pitch_rad"),
            yaw_rad=cls._value(row, "yaw_rad"),
            vx_mps=cls._value(row, "vx_mps"),
            vy_mps=cls._value(row, "vy_mps"),
            vz_mps=cls._value(row, "vz_mps"),
            p_radps=cls._value(row, "p_radps"),
            q_radps=cls._value(row, "q_radps"),
            r_radps=cls._value(row, "r_radps"),
        )

    @classmethod
    def _module_from_row(cls, row: pd.Series, module: int) -> ModuleState:
        true_gap = cls._value(row, f"gap_{module}_m")
        return ModuleState(
            module=module,
            true_gap_m=true_gap,
            sensor_gap_m=cls._value(row, f"sensor_gap_{module}_m", true_gap),
            estimated_gap_m=cls._value(row, f"estimated_gap_{module}_m", true_gap),
            coil_current_a=cls._value(row, f"coil_current_{module}_a"),
            em_force_n=cls._value(row, f"em_force_{module}_n"),
            pressure_pa=cls._value(row, f"pressure_{module}_pa"),
            pneumatic_force_n=cls._value(row, f"pneumatic_force_{module}_n"),
            health=cls._value(row, f"health_{module}", 1.0),
            coil_efficiency=cls._value(row, f"coil_efficiency_{module}", 1.0),
        )

    def relative_transform(self, frame: DigitalTwinFrame, pivot_m: tuple[float, float, float]) -> np.ndarray:
        """Return a rigid transform from the imported reference pose to ``frame``.

        Imported CAD remains the reference geometry. Translation is the simulated
        displacement from frame zero, while rotation is relative to the initial
        attitude and is applied about the manifest/platform pivot.
        """
        current = frame.rigid_body
        reference = self.reference
        rotation = _rotation_matrix(current.roll_rad, current.pitch_rad, current.yaw_rad)
        rotation_ref = _rotation_matrix(reference.roll_rad, reference.pitch_rad, reference.yaw_rad)
        relative_rotation = rotation @ rotation_ref.T
        translation = np.array(
            [current.x_m - reference.x_m, current.y_m - reference.y_m, current.z_m - reference.z_m],
            dtype=float,
        )
        pivot = np.asarray(pivot_m, dtype=float)
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = relative_rotation
        transform[:3, 3] = translation + pivot - relative_rotation @ pivot
        return transform


def _rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx
