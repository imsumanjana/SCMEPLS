from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

MIN_GAP_M = 0.0015
MAX_GAP_M = 0.016
MIN_COMPLETE_RUN_TIME_S = 75.0


@dataclass(frozen=True)
class RolloutParameters:
    dt_s: float = 0.01
    end_time_s: float = 80.0
    track_length_m: float = 1.50
    platform_mass_kg: float = 25.0
    payload_mass_kg: float = 50.0
    initial_gap_m: float = 0.002
    target_gap_m: float = 0.010
    cg_shift_x_m: float = 0.035
    cg_shift_y_m: float = -0.025
    cg_shift_time_s: float = 14.0
    wind_start_s: float = 20.0
    wind_end_s: float = 50.0
    coil_fault_time_s: float = 28.0
    coil_fault_index: int = 3
    sensor_fault_start_s: float = 36.0
    sensor_fault_end_s: float = 45.0
    sensor_fault_index: int = 5
    leak_time_s: float = 56.0
    leak_index: int = 6
    seed: int = 1


@dataclass
class _State:
    x: float = 0.0
    vx: float = 0.0
    y: float = 0.0
    vy: float = 0.0
    z: float = 0.002
    vz: float = 0.0
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0
    vib2: float = 0.0
    i_actual: np.ndarray | None = None
    p_actual: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.i_actual is None:
            self.i_actual = np.zeros(8)
        if self.p_actual is None:
            self.p_actual = np.zeros(8)


def _smoothstep(a: float) -> tuple[float, float]:
    aa = float(np.clip(a, 0.0, 1.0))
    s = 3 * aa**2 - 2 * aa**3
    ds_da = 6 * aa * (1 - aa)
    return s, ds_da


def _validate_parameters(params: RolloutParameters) -> None:
    if not np.isfinite(params.dt_s) or params.dt_s <= 0:
        raise ValueError("Time step must be finite and positive.")
    if not np.isfinite(params.end_time_s) or params.end_time_s < MIN_COMPLETE_RUN_TIME_S:
        raise ValueError(
            f"Simulation duration must be at least {MIN_COMPLETE_RUN_TIME_S:g} s so transfer and hard-lock metrics are defined."
        )
    if not np.isfinite(params.track_length_m) or params.track_length_m <= 0:
        raise ValueError("Track length must be finite and positive.")
    if params.platform_mass_kg <= 0 or params.payload_mass_kg < 0:
        raise ValueError("Mass values are invalid.")
    if not (MIN_GAP_M <= params.initial_gap_m <= MAX_GAP_M):
        raise ValueError(f"Initial gap must be between {MIN_GAP_M:g} m and {MAX_GAP_M:g} m.")
    if not (MIN_GAP_M <= params.target_gap_m <= MAX_GAP_M):
        raise ValueError(f"Target gap must be between {MIN_GAP_M:g} m and {MAX_GAP_M:g} m.")
    for name, value in (
        ("coil_fault_index", params.coil_fault_index),
        ("sensor_fault_index", params.sensor_fault_index),
        ("leak_index", params.leak_index),
    ):
        if not isinstance(value, (int, np.integer)) or not 0 <= int(value) <= 8:
            raise ValueError(f"{name} must be an integer from 0 to 8 (0 disables the fault).")
    if params.wind_end_s <= params.wind_start_s:
        raise ValueError("Wind end time must be later than wind start time.")
    if params.sensor_fault_end_s <= params.sensor_fault_start_s:
        raise ValueError("Sensor fault end time must be later than sensor fault start time.")


def _scenario(t: float, p: RolloutParameters) -> dict[str, float]:
    cgx = p.cg_shift_x_m if t > p.cg_shift_time_s else 0.0
    cgy = p.cg_shift_y_m if t > p.cg_shift_time_s else 0.0
    wind = 18.0 * np.sin(0.45 * t) + 8.0 * np.sin(1.8 * t) if p.wind_start_s < t < p.wind_end_s else 0.0
    fault_index = p.coil_fault_index if t > p.coil_fault_time_s else 0
    leak_index = p.leak_index if t > p.leak_time_s else 0
    sensor_fault_index = p.sensor_fault_index if p.sensor_fault_start_s < t < p.sensor_fault_end_s else 0

    if t < 1:
        mode, gap, x_ref, v_ref, lock = 0, p.initial_gap_m, 0.0, 0.0, 0.0
    elif t < 8:
        s, _ = _smoothstep((t - 1) / 7)
        mode, gap, x_ref, v_ref, lock = 1, p.initial_gap_m + (p.target_gap_m - p.initial_gap_m) * s, 0.0, 0.0, 0.0
    elif t < 38:
        a = (t - 8) / 30
        s, ds_da = _smoothstep(a)
        mode, gap, x_ref, v_ref, lock = 2, p.target_gap_m, p.track_length_m * s, p.track_length_m * ds_da / 30, 0.0
    elif t < 48:
        mode, gap, x_ref, v_ref, lock = 3, p.target_gap_m, p.track_length_m, 0.0, 0.0
    elif t < 56:
        s, _ = _smoothstep((t - 48) / 8)
        mode, gap, x_ref, v_ref, lock = 4, p.target_gap_m, p.track_length_m, 0.0, 0.10 * s
    elif t < 72:
        s, _ = _smoothstep((t - 56) / 16)
        mode, gap, x_ref, v_ref, lock = 5, p.target_gap_m, p.track_length_m, 0.0, 0.10 + 0.90 * s
    else:
        mode, gap, x_ref, v_ref, lock = 6, p.target_gap_m, p.track_length_m, 0.0, 1.0
    return {
        "mode": mode,
        "gap_ref": gap,
        "x_ref": x_ref,
        "v_ref": v_ref,
        "lock_fraction": lock,
        "cgx": cgx,
        "cgy": cgy,
        "wind_y": wind,
        "fault_index": fault_index,
        "leak_index": leak_index,
        "sensor_fault_index": sensor_fault_index,
    }


def simulate_rollout(params: RolloutParameters) -> tuple[pd.DataFrame, dict[str, float]]:
    _validate_parameters(params)
    t_values = np.arange(0.0, params.end_time_s + params.dt_s / 2, params.dt_s)
    state = _State(z=params.initial_gap_m)
    px = np.array([-0.55, 0.0, 0.55, -0.55, 0.55, -0.55, 0.0, 0.55])
    py = np.array([0.25, 0.25, 0.25, 0.0, 0.0, -0.25, -0.25, -0.25])
    mass = params.platform_mass_kg + params.payload_mass_kg
    gravity = 9.81
    weight = mass * gravity
    rows: list[dict[str, float]] = []

    measured_gaps = np.full(8, params.initial_gap_m)
    measured_pneumatic = np.zeros(8)

    for t in t_values:
        sc = _scenario(float(t), params)
        health = np.ones(8)
        severity = 0.0
        if sc["fault_index"]:
            health[int(sc["fault_index"]) - 1] = 0.35
            severity = max(severity, 0.65)
        if sc["leak_index"]:
            severity = max(severity, 0.40)
        gaps_for_control = measured_gaps.copy()
        if sc["sensor_fault_index"]:
            idx = int(sc["sensor_fault_index"]) - 1
            gaps_for_control[idx] += 0.0015
            gaps_for_control[idx] = float(np.median(np.delete(gaps_for_control, idx)))
            severity = max(severity, 0.25)

        unsafe = int(
            np.max(np.abs(gaps_for_control - sc["gap_ref"])) > 0.003
            or np.min(gaps_for_control) < MIN_GAP_M
            or abs(state.roll) > 0.075
            or abs(state.pitch) > 0.075
        )
        if unsafe:
            severity = max(severity, 0.85)
        lock_fraction = max(sc["lock_fraction"], 0.65) if unsafe else sc["lock_fraction"]

        kz, dz = 14000.0, 600.0
        total_force_cmd = weight + kz * (sc["gap_ref"] - state.z) - dz * state.vz
        total_force_cmd = float(np.clip(total_force_cmd, 0.70 * weight, 1.12 * weight))
        mx_des = -weight * sc["cgy"] - 5000.0 * state.roll - 450.0 * state.p
        my_des = -weight * sc["cgx"] - 6000.0 * state.pitch - 500.0 * state.q
        em_health = np.maximum(health, 0.05)
        pneumatic_measured_total = float(np.sum(measured_pneumatic))
        em_total = max(0.0, total_force_cmd - pneumatic_measured_total)
        if sc["mode"] >= 6:
            em_total = min(em_total, 0.08 * weight)
        base_share = em_total * em_health / max(np.sum(em_health), 1e-9)
        den_y = np.sum(py**2 * em_health) + 1e-9
        den_x = np.sum(px**2 * em_health) + 1e-9
        force_ref = base_share + em_health * (mx_des * py / den_y + my_des * px / den_x)
        mean_gap = float(np.mean(gaps_for_control))
        force_ref += 10000.0 * (sc["gap_ref"] - gaps_for_control) + 0.25 * 10000.0 * (mean_gap - gaps_for_control)
        force_ref = np.clip(force_ref, 0.0, 320.0)

        kmag = 3e-4
        i_cmd = np.sqrt(np.maximum(force_ref * np.maximum(gaps_for_control, MIN_GAP_M) ** 2 / kmag, 0.0))
        i_cmd = np.clip(i_cmd, 0.0, 10.0)
        area_p = 1.5e-4
        target_pneumatic_total = max(0.0, lock_fraction * total_force_cmd)
        target_per_module = np.full(8, target_pneumatic_total / 8)
        leak_factor = np.ones(8)
        if sc["leak_index"]:
            leak_factor[int(sc["leak_index"]) - 1] = 0.60
        p_cmd = np.minimum(8e5, target_per_module / (area_p * np.maximum(leak_factor, 0.20)))

        fx = 520.0 * (sc["x_ref"] - state.x) + 165.0 * (sc["v_ref"] - state.vx)
        fy = -240.0 * state.y - 75.0 * state.vy - sc["wind_y"]
        mz_cmd = -180.0 * state.yaw - 45.0 * state.r
        if sc["mode"] < 2 or sc["mode"] >= 6:
            fx = 0.0
        fx = float(np.clip(fx, -140.0, 140.0))
        fy = float(np.clip(fy, -80.0, 80.0))
        mz_cmd = float(np.clip(mz_cmd, -45.0, 45.0))

        state.i_actual += (params.dt_s / 0.035) * (i_cmd - state.i_actual)
        state.i_actual = np.clip(state.i_actual, 0.0, 10.0)
        state.p_actual += (params.dt_s / 0.18) * (p_cmd - state.p_actual)
        state.p_actual = np.clip(state.p_actual, 0.0, 8e5)

        em_force = np.zeros(8)
        pneumatic_force = np.zeros(8)
        pressure_measured = np.zeros(8)
        for i in range(8):
            gap_i = max(float(gaps_for_control[i]), MIN_GAP_M)
            efficiency = 0.35 if sc["fault_index"] == i + 1 else 1.0
            em_force[i] = min(320.0, efficiency * kmag * state.i_actual[i] ** 2 / gap_i**2)
            leak = 0.60 if sc["leak_index"] == i + 1 else 1.0
            pressure_measured[i] = leak * state.p_actual[i]
            pneumatic_force[i] = max(0.0, pressure_measured[i] * area_p)

        fz = float(np.sum(em_force) + np.sum(pneumatic_force))
        mx_support = float(np.sum((em_force + pneumatic_force) * py))
        my_support = float(np.sum((em_force + pneumatic_force) * px))
        mx_gravity = weight * sc["cgy"]
        my_gravity = weight * sc["cgx"]
        mx_wind = 0.18 * sc["wind_y"]
        fz_lock = lock_fraction * (9000.0 * (sc["gap_ref"] - state.z) - 700.0 * state.vz)
        mx_lock = -lock_fraction * (2500.0 * state.roll + 450.0 * state.p)
        my_lock = -lock_fraction * (3500.0 * state.pitch + 550.0 * state.q)

        ax = (fx - 16.0 * state.vx) / mass
        ay = (fy + sc["wind_y"] - 24.0 * state.vy) / mass
        az = (fz + fz_lock - weight - 90.0 * state.vz) / mass
        pdot = (mx_support + mx_gravity + mx_wind + mx_lock - 7.0 * state.p) / 16.0
        qdot = (my_support + my_gravity + my_lock - 8.0 * state.q) / 22.0
        rdot = (mz_cmd - 5.0 * state.r) / 28.0

        state.vx += params.dt_s * ax
        state.x += params.dt_s * state.vx
        state.vy += params.dt_s * ay
        state.y += params.dt_s * state.vy
        state.vz += params.dt_s * az
        state.z += params.dt_s * state.vz
        state.p += params.dt_s * pdot
        state.roll += params.dt_s * state.p
        state.q += params.dt_s * qdot
        state.pitch += params.dt_s * state.q
        state.r += params.dt_s * rdot
        state.yaw += params.dt_s * state.r
        if state.z < MIN_GAP_M:
            state.z = MIN_GAP_M
            state.vz = max(state.vz, 0.0)
        if state.z > MAX_GAP_M:
            state.z = MAX_GAP_M
            state.vz = min(state.vz, 0.0)
        measured_gaps = np.maximum(state.z + state.roll * py + state.pitch * px, 0.0008)
        measured_pneumatic = pneumatic_force.copy()
        state.vib2 = 0.995 * state.vib2 + 0.005 * (az**2 + ax**2)
        total_support = fz + fz_lock

        row = {
            "time_s": t,
            "mode": sc["mode"],
            "x_m": state.x,
            "x_ref_m": sc["x_ref"],
            "vx_mps": state.vx,
            "z_m": state.z,
            "gap_ref_m": sc["gap_ref"],
            "mean_gap_m": float(np.mean(measured_gaps)),
            "min_gap_m": float(np.min(measured_gaps)),
            "max_gap_m": float(np.max(measured_gaps)),
            "roll_rad": state.roll,
            "pitch_rad": state.pitch,
            "yaw_rad": state.yaw,
            "ax_mps2": ax,
            "az_mps2": az,
            "vibration_rms_proxy": float(np.sqrt(state.vib2)),
            "em_force_n": float(np.sum(em_force)),
            "pneumatic_force_n": float(np.sum(pneumatic_force)),
            "total_support_force_n": total_support,
            "lock_fraction": lock_fraction,
            "fault_severity": severity,
            "unsafe_flag": unsafe,
            "wind_force_n": sc["wind_y"],
        }
        for i in range(8):
            row[f"gap_{i+1}_m"] = measured_gaps[i]
            row[f"health_{i+1}"] = health[i]
        rows.append(row)

    df = pd.DataFrame(rows)
    after_levitation = df["time_s"] > 10
    transfer = (df["time_s"] > 56) & (df["time_s"] < 72)
    hard_lock = df["time_s"] > 74
    if not after_levitation.any() or not transfer.any() or not hard_lock.any():
        raise RuntimeError("Required rollout validation phases were not sampled; reduce dt or increase end time.")
    metrics = {
        "design_weight_n": weight,
        "maximum_gap_error_after_levitation_mm": float(
            np.max(np.abs(df.loc[after_levitation, "mean_gap_m"] - df.loc[after_levitation, "gap_ref_m"])) * 1000
        ),
        "maximum_roll_deg": float(np.max(np.abs(df["roll_rad"])) * 180 / np.pi),
        "maximum_pitch_deg": float(np.max(np.abs(df["pitch_rad"])) * 180 / np.pi),
        "final_position_error_mm": float(abs(params.track_length_m - df.iloc[-1]["x_m"]) * 1000),
        "support_force_cv_during_transfer_percent": float(
            100 * df.loc[transfer, "total_support_force_n"].std() / max(df.loc[transfer, "total_support_force_n"].mean(), 1e-9)
        ),
        "mean_em_force_after_hard_lock_n": float(df.loc[hard_lock, "em_force_n"].mean()),
        "mean_pneumatic_force_after_hard_lock_n": float(df.loc[hard_lock, "pneumatic_force_n"].mean()),
        "maximum_fault_severity": float(df["fault_severity"].max()),
        "any_unsafe_flag": int(df["unsafe_flag"].max()),
        "parameters": asdict(params),
    }
    return df, metrics
