from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy import signal

ModelType = Literal["first_order", "second_order", "mass_damper_pid"]


@dataclass(frozen=True)
class ResponseParameters:
    model_type: ModelType = "first_order"
    gain: float = 1.0
    time_constant_s: float = 1.0
    natural_frequency_rad_s: float = 2.0
    damping_ratio: float = 0.9
    mass_kg: float = 75.0
    damping_ns_m: float = 150.0
    actuator_gain: float = 1.0
    kp: float = 90.0
    ki: float = 15.0
    kd: float = 60.0


@dataclass(frozen=True)
class ResponseMetrics:
    rise_time_s: float
    settling_time_s: float
    overshoot_percent: float
    steady_state_value: float
    steady_state_error: float
    iae: float
    itae: float


def transfer_function(params: ResponseParameters) -> signal.TransferFunction:
    if params.model_type == "first_order":
        if params.time_constant_s <= 0:
            raise ValueError("Time constant must be positive.")
        return signal.TransferFunction([params.gain], [params.time_constant_s, 1.0])
    if params.model_type == "second_order":
        if params.natural_frequency_rad_s <= 0 or params.damping_ratio <= 0:
            raise ValueError("Natural frequency and damping ratio must be positive.")
        wn = params.natural_frequency_rad_s
        return signal.TransferFunction(
            [params.gain * wn**2],
            [1.0, 2.0 * params.damping_ratio * wn, wn**2],
        )
    if params.model_type == "mass_damper_pid":
        if params.mass_kg <= 0 or params.actuator_gain <= 0:
            raise ValueError("Mass and actuator gain must be positive.")
        ka = params.actuator_gain
        numerator = [ka * params.kd, ka * params.kp, ka * params.ki]
        denominator = [params.mass_kg, params.damping_ns_m + ka * params.kd, ka * params.kp, ka * params.ki]
        return signal.TransferFunction(numerator, denominator)
    raise ValueError(f"Unsupported model type: {params.model_type}")


def simulate_step(params: ResponseParameters, duration_s: float = 10.0, points: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    if duration_s <= 0 or points < 100:
        raise ValueError("Duration must be positive and points must be at least 100.")
    system = transfer_function(params)
    t = np.linspace(0.0, duration_s, points)
    tout, y = signal.step(system, T=t)
    return np.asarray(tout), np.asarray(y)


def calculate_response_metrics(t: np.ndarray, y: np.ndarray, reference: float = 1.0) -> ResponseMetrics:
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if t.size != y.size or t.size < 10:
        raise ValueError("Step response arrays must have equal length and at least 10 points.")
    final = float(np.mean(y[-max(10, y.size // 50):]))
    target_for_rise = final if abs(final) > 1e-12 else reference
    low, high = 0.1 * target_for_rise, 0.9 * target_for_rise
    try:
        t10 = float(t[np.where(y >= low)[0][0]])
        t90 = float(t[np.where(y >= high)[0][0]])
        rise = max(0.0, t90 - t10)
    except IndexError:
        rise = float("nan")
    band = 0.02 * max(abs(target_for_rise), 1e-9)
    outside = np.where(np.abs(y - target_for_rise) > band)[0]
    settling = float(t[outside[-1] + 1]) if outside.size and outside[-1] + 1 < t.size else 0.0
    peak = float(np.max(y))
    overshoot = max(0.0, 100.0 * (peak - target_for_rise) / max(abs(target_for_rise), 1e-9))
    error = reference - y
    # np.trapz is retained for the declared NumPy >=1.26 compatibility floor.
    iae = float(np.trapz(np.abs(error), t))
    itae = float(np.trapz(t * np.abs(error), t))
    return ResponseMetrics(
        rise_time_s=rise,
        settling_time_s=settling,
        overshoot_percent=overshoot,
        steady_state_value=final,
        steady_state_error=float(reference - final),
        iae=iae,
        itae=itae,
    )
