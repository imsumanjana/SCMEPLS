from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from scmepls_studio.coordinates import module_vertical_offsets_m

MIN_GAP_M = 0.0015
MAX_GAP_M = 0.016
MIN_COMPLETE_RUN_TIME_S = 75.0
MAX_TIME_STEP_S = 0.02
MODE_NAMES = {
    0: "INITIAL",
    1: "LEVITATE",
    2: "ROLLOUT",
    3: "ALIGN",
    4: "PRELOCK",
    5: "TRANSFER_SEAT",
    6: "HARD_LOCK",
}


_DEFAULT_MODULE_X = (-0.55, 0.0, 0.55, -0.55, 0.55, -0.55, 0.0, 0.55)
_DEFAULT_MODULE_Y = (0.25, 0.25, 0.25, 0.0, 0.0, -0.25, -0.25, -0.25)


@dataclass(frozen=True)
class RolloutParameters:
    # Numerical / mission definition
    dt_s: float = 0.01
    end_time_s: float = 80.0
    track_length_m: float = 1.50

    # Moving-body mass properties
    platform_mass_kg: float = 25.0
    payload_mass_kg: float = 50.0
    body_length_m: float = 1.10
    body_width_m: float = 0.50
    body_height_m: float = 0.36
    inertia_xx_kg_m2: float = 16.0
    inertia_yy_kg_m2: float = 22.0
    inertia_zz_kg_m2: float = 28.0
    inertia_xy_kg_m2: float = 0.0
    inertia_xz_kg_m2: float = 0.0
    inertia_yz_kg_m2: float = 0.0

    # Module geometry in body coordinates
    module_x_m: tuple[float, ...] = _DEFAULT_MODULE_X
    module_y_m: tuple[float, ...] = _DEFAULT_MODULE_Y

    # Gap / payload-disturbance scenario
    initial_gap_m: float = 0.002
    target_gap_m: float = 0.010
    cg_shift_x_m: float = 0.035
    cg_shift_y_m: float = -0.025
    cg_shift_time_s: float = 14.0

    # Environmental / fault scenario
    wind_start_s: float = 20.0
    wind_end_s: float = 50.0
    wind_amp1_n: float = 18.0
    wind_freq1_rad_s: float = 0.45
    wind_amp2_n: float = 8.0
    wind_freq2_rad_s: float = 1.8
    coil_fault_time_s: float = 28.0
    coil_fault_index: int = 3
    coil_fault_efficiency: float = 0.35
    sensor_fault_start_s: float = 36.0
    sensor_fault_end_s: float = 45.0
    sensor_fault_index: int = 5
    sensor_bias_m: float = 0.0015
    leak_time_s: float = 56.0
    leak_index: int = 6
    leak_retained_fraction: float = 0.60

    # Scenario phase boundaries
    initial_end_s: float = 1.0
    levitate_end_s: float = 8.0
    rollout_end_s: float = 38.0
    align_end_s: float = 48.0
    prelock_end_s: float = 56.0
    transfer_end_s: float = 68.0
    hardlock_start_s: float = 72.0

    # Electromagnetic / pneumatic actuator model
    electromagnetic_k_n_m2_a2: float = 3.0e-4
    max_module_current_a: float = 10.0
    max_module_em_force_n: float = 320.0
    current_time_constant_s: float = 0.035
    pneumatic_area_m2: float = 1.5e-4
    max_pneumatic_pressure_pa: float = 8.0e5
    pressure_time_constant_s: float = 0.18

    # Rigid-body control / damping
    vertical_k_n_m: float = 14000.0
    vertical_damping_ns_m: float = 600.0
    force_command_min_weight_fraction: float = 0.70
    force_command_max_weight_fraction: float = 1.12
    roll_k_nm_rad: float = 5000.0
    roll_damping_nms_rad: float = 450.0
    pitch_k_nm_rad: float = 6000.0
    pitch_damping_nms_rad: float = 500.0
    local_gap_k_n_m: float = 10000.0
    equalization_k_n_m: float = 2500.0
    propulsion_k_n_m: float = 520.0
    propulsion_damping_ns_m: float = 165.0
    lateral_k_n_m: float = 240.0
    lateral_damping_ns_m: float = 75.0
    yaw_k_nm_rad: float = 180.0
    yaw_damping_nms_rad: float = 45.0
    max_propulsion_force_n: float = 140.0
    max_lateral_control_force_n: float = 80.0
    max_yaw_control_moment_nm: float = 45.0
    passive_x_damping_ns_m: float = 16.0
    passive_y_damping_ns_m: float = 24.0
    passive_z_damping_ns_m: float = 90.0
    passive_roll_damping_nms_rad: float = 7.0
    passive_pitch_damping_nms_rad: float = 8.0
    passive_yaw_damping_nms_rad: float = 5.0

    # Mechanical lock model
    lock_x_k_n_m: float = 4000.0
    lock_x_damping_ns_m: float = 600.0
    lock_y_k_n_m: float = 3500.0
    lock_y_damping_ns_m: float = 500.0
    lock_z_k_n_m: float = 9000.0
    lock_z_damping_ns_m: float = 700.0
    lock_roll_k_nm_rad: float = 2500.0
    lock_roll_damping_nms_rad: float = 450.0
    lock_pitch_k_nm_rad: float = 3500.0
    lock_pitch_damping_nms_rad: float = 550.0
    lock_yaw_k_nm_rad: float = 2200.0
    lock_yaw_damping_nms_rad: float = 300.0
    lock_rate_per_s: float = 0.75
    hardlock_em_weight_fraction: float = 0.08

    # Docking/interlock envelope
    docking_x_tolerance_m: float = 0.05
    docking_y_tolerance_m: float = 0.02
    docking_z_tolerance_m: float = 0.003
    docking_roll_tolerance_rad: float = 0.05
    docking_pitch_tolerance_rad: float = 0.05
    docking_yaw_tolerance_rad: float = 0.05
    docking_vx_tolerance_mps: float = 0.03
    docking_vy_tolerance_mps: float = 0.03
    docking_vz_tolerance_mps: float = 0.02
    docking_rate_tolerance_rad_s: float = 0.05
    unsafe_gap_error_m: float = 0.003
    unsafe_attitude_rad: float = 0.075

    # Diagnostic / display filtering
    vibration_rms_time_constant_s: float = 1.0

    @property
    def inertia_tensor_kg_m2(self) -> np.ndarray:
        return np.array(
            [
                [self.inertia_xx_kg_m2, self.inertia_xy_kg_m2, self.inertia_xz_kg_m2],
                [self.inertia_xy_kg_m2, self.inertia_yy_kg_m2, self.inertia_yz_kg_m2],
                [self.inertia_xz_kg_m2, self.inertia_yz_kg_m2, self.inertia_zz_kg_m2],
            ],
            dtype=float,
        )

    @property
    def module_x_array_m(self) -> np.ndarray:
        return np.asarray(self.module_x_m, dtype=float)

    @property
    def module_y_array_m(self) -> np.ndarray:
        return np.asarray(self.module_y_m, dtype=float)


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
    lock_fraction: float = 0.0
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
    return s, 6 * aa * (1 - aa)


def _validate_parameters(params: RolloutParameters) -> None:
    finite_positive = {
        "dt_s": params.dt_s,
        "end_time_s": params.end_time_s,
        "track_length_m": params.track_length_m,
        "platform_mass_kg": params.platform_mass_kg,
        "body_length_m": params.body_length_m,
        "body_width_m": params.body_width_m,
        "body_height_m": params.body_height_m,
        "electromagnetic_k_n_m2_a2": params.electromagnetic_k_n_m2_a2,
        "max_module_current_a": params.max_module_current_a,
        "max_module_em_force_n": params.max_module_em_force_n,
        "current_time_constant_s": params.current_time_constant_s,
        "pneumatic_area_m2": params.pneumatic_area_m2,
        "max_pneumatic_pressure_pa": params.max_pneumatic_pressure_pa,
        "pressure_time_constant_s": params.pressure_time_constant_s,
        "lock_rate_per_s": params.lock_rate_per_s,
        "vibration_rms_time_constant_s": params.vibration_rms_time_constant_s,
    }
    for name, value in finite_positive.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive.")
    if params.dt_s > MAX_TIME_STEP_S:
        raise ValueError(f"Time step must be no greater than {MAX_TIME_STEP_S:g} s.")
    if params.end_time_s < MIN_COMPLETE_RUN_TIME_S:
        raise ValueError(
            f"Simulation duration must be at least {MIN_COMPLETE_RUN_TIME_S:g} s so transfer and hard-lock metrics are defined."
        )
    if params.payload_mass_kg < 0 or not np.isfinite(params.payload_mass_kg):
        raise ValueError("Payload mass must be finite and non-negative.")

    px, py = params.module_x_array_m, params.module_y_array_m
    if px.shape != (8,) or py.shape != (8,) or np.any(~np.isfinite(px)) or np.any(~np.isfinite(py)):
        raise ValueError("module_x_m and module_y_m must each contain eight finite coordinates.")
    if len(np.unique(np.column_stack((px, py)), axis=0)) < 4:
        raise ValueError("Module coordinates must define a distributed support footprint.")

    inertia = params.inertia_tensor_kg_m2
    if np.any(~np.isfinite(inertia)) or np.any(np.linalg.eigvalsh(inertia) <= 0):
        raise ValueError("Rigid-body inertia tensor must be finite and positive definite.")
    if not (MIN_GAP_M <= params.initial_gap_m <= MAX_GAP_M):
        raise ValueError(f"Initial/seated gap must be between {MIN_GAP_M:g} m and {MAX_GAP_M:g} m.")
    if not (MIN_GAP_M <= params.target_gap_m <= MAX_GAP_M):
        raise ValueError(f"Levitation gap must be between {MIN_GAP_M:g} m and {MAX_GAP_M:g} m.")
    if params.initial_gap_m >= params.target_gap_m:
        raise ValueError("Initial/seated gap must be smaller than the levitation target gap.")
    for name, value in (
        ("coil_fault_index", params.coil_fault_index),
        ("sensor_fault_index", params.sensor_fault_index),
        ("leak_index", params.leak_index),
    ):
        if not isinstance(value, (int, np.integer)) or not 0 <= int(value) <= 8:
            raise ValueError(f"{name} must be an integer from 0 to 8 (0 disables the fault).")
    if not (0.0 < params.coil_fault_efficiency <= 1.0):
        raise ValueError("coil_fault_efficiency must be in (0, 1].")
    if not (0.0 <= params.leak_retained_fraction <= 1.0):
        raise ValueError("leak_retained_fraction must be between 0 and 1.")
    if params.wind_end_s <= params.wind_start_s:
        raise ValueError("Wind end time must be later than wind start time.")
    if params.sensor_fault_end_s <= params.sensor_fault_start_s:
        raise ValueError("Sensor fault end time must be later than sensor fault start time.")
    phase_times = np.array(
        [
            params.initial_end_s,
            params.levitate_end_s,
            params.rollout_end_s,
            params.align_end_s,
            params.prelock_end_s,
            params.transfer_end_s,
            params.hardlock_start_s,
        ],
        dtype=float,
    )
    if np.any(~np.isfinite(phase_times)) or np.any(np.diff(phase_times) <= 0):
        raise ValueError("Scenario phase boundaries must be finite and strictly increasing.")
    if params.hardlock_start_s >= params.end_time_s:
        raise ValueError("hardlock_start_s must occur before end_time_s.")


def _scenario(t: float, p: RolloutParameters) -> dict[str, float]:
    cgx = p.cg_shift_x_m if t > p.cg_shift_time_s else 0.0
    cgy = p.cg_shift_y_m if t > p.cg_shift_time_s else 0.0
    wind = (
        p.wind_amp1_n * np.sin(p.wind_freq1_rad_s * t)
        + p.wind_amp2_n * np.sin(p.wind_freq2_rad_s * t)
        if p.wind_start_s < t < p.wind_end_s
        else 0.0
    )
    fault_index = p.coil_fault_index if t > p.coil_fault_time_s else 0
    leak_index = p.leak_index if t > p.leak_time_s else 0
    sensor_fault_index = p.sensor_fault_index if p.sensor_fault_start_s < t < p.sensor_fault_end_s else 0

    if t < p.initial_end_s:
        mode, gap, x_ref, v_ref, lock = 0, p.initial_gap_m, 0.0, 0.0, 0.0
    elif t < p.levitate_end_s:
        span = p.levitate_end_s - p.initial_end_s
        s, _ = _smoothstep((t - p.initial_end_s) / span)
        mode, gap, x_ref, v_ref, lock = (
            1,
            p.initial_gap_m + (p.target_gap_m - p.initial_gap_m) * s,
            0.0,
            0.0,
            0.0,
        )
    elif t < p.rollout_end_s:
        span = p.rollout_end_s - p.levitate_end_s
        s, ds_da = _smoothstep((t - p.levitate_end_s) / span)
        mode, gap, x_ref, v_ref, lock = (
            2,
            p.target_gap_m,
            p.track_length_m * s,
            p.track_length_m * ds_da / span,
            0.0,
        )
    elif t < p.align_end_s:
        mode, gap, x_ref, v_ref, lock = 3, p.target_gap_m, p.track_length_m, 0.0, 0.0
    elif t < p.prelock_end_s:
        span = p.prelock_end_s - p.align_end_s
        s, _ = _smoothstep((t - p.align_end_s) / span)
        mode, gap, x_ref, v_ref, lock = 4, p.target_gap_m, p.track_length_m, 0.0, 0.10 * s
    elif t < p.transfer_end_s:
        span = p.transfer_end_s - p.prelock_end_s
        s, _ = _smoothstep((t - p.prelock_end_s) / span)
        gap = p.target_gap_m + (p.initial_gap_m - p.target_gap_m) * s
        mode, x_ref, v_ref, lock = 5, p.track_length_m, 0.0, 0.10 + 0.75 * s
    elif t < p.hardlock_start_s:
        span = p.hardlock_start_s - p.transfer_end_s
        s, _ = _smoothstep((t - p.transfer_end_s) / span)
        mode, gap, x_ref, v_ref, lock = 5, p.initial_gap_m, p.track_length_m, 0.0, 0.85 + 0.15 * s
    else:
        mode, gap, x_ref, v_ref, lock = 6, p.initial_gap_m, p.track_length_m, 0.0, 1.0

    return {
        "mode": float(mode),
        "gap_ref": float(gap),
        "x_ref": float(x_ref),
        "v_ref": float(v_ref),
        "lock_fraction": float(lock),
        "cgx": float(cgx),
        "cgy": float(cgy),
        "wind_y": float(wind),
        "fault_index": float(fault_index),
        "leak_index": float(leak_index),
        "sensor_fault_index": float(sensor_fault_index),
    }


def _reconstruct_failed_gap(
    measured_gaps: np.ndarray,
    failed_index: int,
    px: np.ndarray,
    py: np.ndarray,
) -> float:
    """Reconstruct one failed gap from a rigid-plane fit to healthy sensors."""
    healthy = np.ones(8, dtype=bool)
    healthy[int(failed_index)] = False
    design = np.column_stack((np.ones(np.count_nonzero(healthy)), py[healthy], -px[healthy]))
    target = measured_gaps[healthy]
    coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
    query = np.array([1.0, py[failed_index], -px[failed_index]], dtype=float)
    return float(query @ coefficients)


def _docking_envelope(state: _State, gap_ref_m: float, p: RolloutParameters) -> bool:
    return bool(
        abs(state.x - p.track_length_m) <= p.docking_x_tolerance_m
        and abs(state.y) <= p.docking_y_tolerance_m
        and abs(state.z - gap_ref_m) <= p.docking_z_tolerance_m
        and abs(state.roll) <= p.docking_roll_tolerance_rad
        and abs(state.pitch) <= p.docking_pitch_tolerance_rad
        and abs(state.yaw) <= p.docking_yaw_tolerance_rad
        and abs(state.vx) <= p.docking_vx_tolerance_mps
        and abs(state.vy) <= p.docking_vy_tolerance_mps
        and abs(state.vz) <= p.docking_vz_tolerance_mps
        and max(abs(state.p), abs(state.q), abs(state.r)) <= p.docking_rate_tolerance_rad_s
    )


def _update_lock_state(
    current: float,
    scheduled_target: float,
    *,
    docking_ready: bool,
    detected_interlock: bool,
    physical_unsafe: bool,
    dt_s: float,
    rate_per_s: float,
) -> tuple[float, str]:
    """Interlocked monotonic lock engagement with no unsafe forced engagement."""
    current = float(np.clip(current, 0.0, 1.0))
    permitted = docking_ready and not detected_interlock and not physical_unsafe
    if current >= 0.98:
        target = max(current, scheduled_target if permitted else current)
        state = "ENGAGED"
    elif permitted:
        target = max(current, float(np.clip(scheduled_target, 0.0, 1.0)))
        state = "ENGAGING" if target > current + 1e-9 else ("ENGAGED" if current >= 0.98 else "READY")
    else:
        target = current
        state = "INHIBITED" if scheduled_target > current + 1e-9 else "DISENGAGED"
    delta = np.clip(target - current, -rate_per_s * dt_s, rate_per_s * dt_s)
    updated = float(np.clip(current + delta, 0.0, 1.0))
    if updated >= 0.999:
        updated = 1.0
        state = "ENGAGED"
    return updated, state


def simulate_rollout(params: RolloutParameters) -> tuple[pd.DataFrame, dict[str, float]]:
    _validate_parameters(params)
    t_values = np.arange(0.0, params.end_time_s + params.dt_s / 2, params.dt_s)
    state = _State(z=params.initial_gap_m)
    px = params.module_x_array_m.copy()
    py = params.module_y_array_m.copy()
    mass = params.platform_mass_kg + params.payload_mass_kg
    gravity = 9.81
    weight = mass * gravity
    inertia = params.inertia_tensor_kg_m2
    rows: list[dict[str, float | str]] = []

    true_gaps = np.full(8, params.initial_gap_m, dtype=float)
    measured_pneumatic = np.zeros(8, dtype=float)
    current_alpha = 1.0 - np.exp(-params.dt_s / params.current_time_constant_s)
    pressure_alpha = 1.0 - np.exp(-params.dt_s / params.pressure_time_constant_s)
    vibration_alpha = 1.0 - np.exp(-params.dt_s / params.vibration_rms_time_constant_s)

    for t in t_values:
        sc = _scenario(float(t), params)
        actual_coil_efficiency = np.ones(8, dtype=float)
        if sc["fault_index"]:
            actual_coil_efficiency[int(sc["fault_index"]) - 1] = params.coil_fault_efficiency
        health = actual_coil_efficiency.copy()
        injected_fault_severity = 1.0 - float(np.min(actual_coil_efficiency))
        if sc["leak_index"]:
            injected_fault_severity = max(injected_fault_severity, 1.0 - params.leak_retained_fraction)

        measured_gaps = true_gaps.copy()
        estimated_gaps = measured_gaps.copy()
        if sc["sensor_fault_index"]:
            idx = int(sc["sensor_fault_index"]) - 1
            measured_gaps[idx] += params.sensor_bias_m
            estimated_gaps[idx] = _reconstruct_failed_gap(measured_gaps, idx, px, py)
            injected_fault_severity = max(
                injected_fault_severity,
                min(1.0, abs(params.sensor_bias_m) / max(params.unsafe_gap_error_m, 1e-12)),
            )

        scheduled_gap_ref = float(sc["gap_ref"])
        initial_docking_ready = _docking_envelope(state, scheduled_gap_ref, params)
        effective_gap_ref = (
            scheduled_gap_ref
            if (int(sc["mode"]) < 5 or initial_docking_ready or state.lock_fraction > 0.0)
            else params.target_gap_m
        )

        physical_unsafe = bool(
            np.max(np.abs(true_gaps - effective_gap_ref)) > params.unsafe_gap_error_m
            or np.min(true_gaps) < MIN_GAP_M
            or abs(state.roll) > params.unsafe_attitude_rad
            or abs(state.pitch) > params.unsafe_attitude_rad
        )
        detected_interlock = bool(
            np.max(np.abs(estimated_gaps - effective_gap_ref)) > params.unsafe_gap_error_m
            or np.min(estimated_gaps) < MIN_GAP_M
            or abs(state.roll) > params.unsafe_attitude_rad
            or abs(state.pitch) > params.unsafe_attitude_rad
        )
        docking_ready = _docking_envelope(state, effective_gap_ref, params)
        state.lock_fraction, lock_state = _update_lock_state(
            state.lock_fraction,
            float(sc["lock_fraction"]),
            docking_ready=docking_ready,
            detected_interlock=detected_interlock,
            physical_unsafe=physical_unsafe,
            dt_s=params.dt_s,
            rate_per_s=params.lock_rate_per_s,
        )
        hard_lock_confirmed = state.lock_fraction >= 0.999 and docking_ready and not physical_unsafe
        severity = max(
            injected_fault_severity,
            0.85 if physical_unsafe else 0.0,
            0.70 if detected_interlock else 0.0,
        )

        total_force_cmd = (
            weight
            + params.vertical_k_n_m * (effective_gap_ref - state.z)
            - params.vertical_damping_ns_m * state.vz
        )
        total_force_cmd = float(
            np.clip(
                total_force_cmd,
                params.force_command_min_weight_fraction * weight,
                params.force_command_max_weight_fraction * weight,
            )
        )

        mx_gravity = -weight * float(sc["cgy"])
        my_gravity = weight * float(sc["cgx"])
        mx_des = -mx_gravity - params.roll_k_nm_rad * state.roll - params.roll_damping_nms_rad * state.p
        my_des = -my_gravity - params.pitch_k_nm_rad * state.pitch - params.pitch_damping_nms_rad * state.q

        allocator_health = np.maximum(health, 0.05)
        pneumatic_measured_total = float(np.sum(measured_pneumatic))
        em_total = max(0.0, total_force_cmd - pneumatic_measured_total)
        if hard_lock_confirmed:
            em_total = min(em_total, params.hardlock_em_weight_fraction * weight)

        base_share = em_total * allocator_health / max(np.sum(allocator_health), 1e-12)
        den_y = np.sum(py**2 * allocator_health) + 1e-12
        den_x = np.sum(px**2 * allocator_health) + 1e-12
        force_ref = base_share + allocator_health * (
            mx_des * py / den_y - my_des * px / den_x
        )
        mean_gap = float(np.mean(estimated_gaps))
        force_ref += (
            params.local_gap_k_n_m * (effective_gap_ref - estimated_gaps)
            + params.equalization_k_n_m * (mean_gap - estimated_gaps)
        )
        force_ref = np.clip(force_ref, 0.0, params.max_module_em_force_n)

        efficiency_estimate = np.maximum(health, 0.05)
        i_cmd = np.sqrt(
            np.maximum(
                force_ref
                * np.maximum(estimated_gaps, MIN_GAP_M) ** 2
                / (params.electromagnetic_k_n_m2_a2 * efficiency_estimate),
                0.0,
            )
        )
        i_cmd = np.clip(i_cmd, 0.0, params.max_module_current_a)

        target_pneumatic_total = max(0.0, state.lock_fraction * total_force_cmd)
        p_cmd = np.minimum(
            params.max_pneumatic_pressure_pa,
            np.full(8, target_pneumatic_total / 8.0) / params.pneumatic_area_m2,
        )

        fx_control = (
            params.propulsion_k_n_m * (float(sc["x_ref"]) - state.x)
            + params.propulsion_damping_ns_m * (float(sc["v_ref"]) - state.vx)
        )
        fy_control = (
            -params.lateral_k_n_m * state.y
            - params.lateral_damping_ns_m * state.vy
            - float(sc["wind_y"])
        )
        mz_control = -params.yaw_k_nm_rad * state.yaw - params.yaw_damping_nms_rad * state.r
        if int(sc["mode"]) < 2:
            fx_control = 0.0
        if hard_lock_confirmed:
            fx_control = fy_control = mz_control = 0.0
        fx_control = float(np.clip(fx_control, -params.max_propulsion_force_n, params.max_propulsion_force_n))
        fy_control = float(np.clip(fy_control, -params.max_lateral_control_force_n, params.max_lateral_control_force_n))
        mz_control = float(np.clip(mz_control, -params.max_yaw_control_moment_nm, params.max_yaw_control_moment_nm))

        assert state.i_actual is not None and state.p_actual is not None
        state.i_actual += current_alpha * (i_cmd - state.i_actual)
        state.i_actual = np.clip(state.i_actual, 0.0, params.max_module_current_a)
        state.p_actual += pressure_alpha * (p_cmd - state.p_actual)
        state.p_actual = np.clip(state.p_actual, 0.0, params.max_pneumatic_pressure_pa)

        em_force = np.zeros(8, dtype=float)
        pneumatic_force = np.zeros(8, dtype=float)
        pressure_measured = np.zeros(8, dtype=float)
        for i in range(8):
            physical_gap = max(float(true_gaps[i]), 0.0008)
            em_force[i] = min(
                params.max_module_em_force_n,
                actual_coil_efficiency[i]
                * params.electromagnetic_k_n_m2_a2
                * state.i_actual[i] ** 2
                / physical_gap**2,
            )
            leak = params.leak_retained_fraction if int(sc["leak_index"]) == i + 1 else 1.0
            pressure_measured[i] = leak * state.p_actual[i]
            pneumatic_force[i] = max(0.0, pressure_measured[i] * params.pneumatic_area_m2)

        support_force = em_force + pneumatic_force
        fz_support = float(np.sum(support_force))
        mx_support = float(np.sum(support_force * py))
        my_support = float(-np.sum(support_force * px))
        mx_wind = -0.5 * params.body_height_m * float(sc["wind_y"])

        lf = state.lock_fraction
        fx_lock = lf * (
            params.lock_x_k_n_m * (params.track_length_m - state.x)
            - params.lock_x_damping_ns_m * state.vx
        )
        fy_lock = lf * (-params.lock_y_k_n_m * state.y - params.lock_y_damping_ns_m * state.vy)
        fz_lock = lf * (
            params.lock_z_k_n_m * (effective_gap_ref - state.z)
            - params.lock_z_damping_ns_m * state.vz
        )
        mx_lock = -lf * (
            params.lock_roll_k_nm_rad * state.roll + params.lock_roll_damping_nms_rad * state.p
        )
        my_lock = -lf * (
            params.lock_pitch_k_nm_rad * state.pitch + params.lock_pitch_damping_nms_rad * state.q
        )
        mz_lock = -lf * (
            params.lock_yaw_k_nm_rad * state.yaw + params.lock_yaw_damping_nms_rad * state.r
        )

        fx_passive = -params.passive_x_damping_ns_m * state.vx
        fy_passive = -params.passive_y_damping_ns_m * state.vy
        fz_passive = -params.passive_z_damping_ns_m * state.vz
        ax = (fx_control + fx_lock + fx_passive) / mass
        ay = (fy_control + fy_lock + float(sc["wind_y"]) + fy_passive) / mass
        az = (fz_support + fz_lock - weight + fz_passive) / mass

        omega = np.array([state.p, state.q, state.r], dtype=float)
        moment = np.array(
            [
                mx_support + mx_gravity + mx_wind + mx_lock - params.passive_roll_damping_nms_rad * state.p,
                my_support + my_gravity + my_lock - params.passive_pitch_damping_nms_rad * state.q,
                mz_control + mz_lock - params.passive_yaw_damping_nms_rad * state.r,
            ],
            dtype=float,
        )
        omega_dot = np.linalg.solve(inertia, moment - np.cross(omega, inertia @ omega))
        pdot, qdot, rdot = [float(value) for value in omega_dot]

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

        true_gaps = np.maximum(
            state.z + module_vertical_offsets_m(px, py, state.roll, state.pitch, state.yaw),
            0.0008,
        )
        measured_pneumatic = pneumatic_force.copy()
        state.vib2 += vibration_alpha * ((az**2 + ax**2 + ay**2) - state.vib2)
        total_support = fz_support + fz_lock

        row: dict[str, float | str] = {
            "time_s": float(t),
            "mode": int(sc["mode"]),
            "mode_name": MODE_NAMES[int(sc["mode"])],
            "x_m": state.x,
            "x_ref_m": float(sc["x_ref"]),
            "vx_mps": state.vx,
            "y_m": state.y,
            "vy_mps": state.vy,
            "z_m": state.z,
            "vz_mps": state.vz,
            "gap_ref_m": effective_gap_ref,
            "mean_gap_m": float(np.mean(true_gaps)),
            "min_gap_m": float(np.min(true_gaps)),
            "max_gap_m": float(np.max(true_gaps)),
            "roll_rad": state.roll,
            "pitch_rad": state.pitch,
            "yaw_rad": state.yaw,
            "roll_rate_rad_s": state.p,
            "pitch_rate_rad_s": state.q,
            "yaw_rate_rad_s": state.r,
            "p_radps": state.p,
            "q_radps": state.q,
            "r_radps": state.r,
            "roll_accel_rad_s2": pdot,
            "pitch_accel_rad_s2": qdot,
            "yaw_accel_rad_s2": rdot,
            "ax_mps2": ax,
            "ay_mps2": ay,
            "az_mps2": az,
            "vibration_rms_proxy": float(np.sqrt(max(state.vib2, 0.0))),
            "em_force_n": float(np.sum(em_force)),
            "pneumatic_force_n": float(np.sum(pneumatic_force)),
            "total_support_force_n": total_support,
            "propulsion_force_x_n": fx_control,
            "lateral_control_force_y_n": fy_control,
            "passive_force_x_n": fx_passive,
            "passive_force_y_n": fy_passive,
            "passive_force_z_n": fz_passive,
            "lock_force_x_n": fx_lock,
            "lock_force_y_n": fy_lock,
            "lock_force_z_n": fz_lock,
            "support_moment_x_nm": mx_support,
            "support_moment_y_nm": my_support,
            "gravity_moment_x_nm": mx_gravity,
            "gravity_moment_y_nm": my_gravity,
            "wind_moment_x_nm": mx_wind,
            "yaw_control_moment_nm": mz_control,
            "lock_moment_x_nm": mx_lock,
            "lock_moment_y_nm": my_lock,
            "lock_moment_z_nm": mz_lock,
            "lock_fraction": state.lock_fraction,
            "lock_state": lock_state,
            "hard_lock_confirmed": int(hard_lock_confirmed),
            "docking_ready": int(docking_ready),
            "detected_interlock": int(detected_interlock),
            "physical_truth_violation": int(physical_unsafe),
            "fault_severity": severity,
            "unsafe_flag": int(physical_unsafe),
            "wind_force_n": float(sc["wind_y"]),
            "cg_shift_x_m": float(sc["cgx"]),
            "cg_shift_y_m": float(sc["cgy"]),
        }
        for i in range(8):
            n = i + 1
            row[f"module_x_{n}_m"] = px[i]
            row[f"module_y_{n}_m"] = py[i]
            row[f"gap_{n}_m"] = true_gaps[i]
            row[f"sensor_gap_{n}_m"] = measured_gaps[i]
            row[f"estimated_gap_{n}_m"] = estimated_gaps[i]
            row[f"health_{n}"] = health[i]
            row[f"coil_efficiency_{n}"] = actual_coil_efficiency[i]
            row[f"coil_current_{n}_a"] = state.i_actual[i]
            row[f"em_force_{n}_n"] = em_force[i]
            row[f"pressure_{n}_pa"] = pressure_measured[i]
            row[f"pneumatic_force_{n}_n"] = pneumatic_force[i]
        rows.append(row)

    df = pd.DataFrame(rows)
    after_levitation = df["time_s"] > max(params.levitate_end_s + 2.0, 10.0)
    transfer = (df["time_s"] > params.prelock_end_s) & (df["time_s"] < params.hardlock_start_s)
    hard_lock = df["time_s"] > min(params.hardlock_start_s + 2.0, params.end_time_s - params.dt_s)
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
        "final_lateral_error_mm": float(abs(df.iloc[-1]["y_m"]) * 1000),
        "final_yaw_error_deg": float(abs(df.iloc[-1]["yaw_rad"]) * 180 / np.pi),
        "final_mean_gap_mm": float(df.iloc[-1]["mean_gap_m"] * 1000),
        "support_force_cv_during_transfer_percent": float(
            100
            * df.loc[transfer, "total_support_force_n"].std()
            / max(df.loc[transfer, "total_support_force_n"].mean(), 1e-9)
        ),
        "mean_em_force_after_hard_lock_n": float(df.loc[hard_lock, "em_force_n"].mean()),
        "mean_pneumatic_force_after_hard_lock_n": float(df.loc[hard_lock, "pneumatic_force_n"].mean()),
        "maximum_fault_severity": float(df["fault_severity"].max()),
        "any_unsafe_flag": int(df["unsafe_flag"].max()),
        "any_detected_interlock": int(df["detected_interlock"].max()),
        "hard_lock_confirmed": int(df.iloc[-1]["hard_lock_confirmed"]),
        "module_coordinates_source": "configured",
        "parameters": asdict(params),
    }
    return df, metrics
