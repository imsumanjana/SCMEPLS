import numpy as np
import pandas as pd

from scmepls_studio.coordinates import module_vertical_offsets_m
from scmepls_studio.digital_twin.physics import SimulationTimeline
from scmepls_studio.models.rollout_sim import RolloutParameters, _update_lock_state, simulate_rollout


def test_coordinate_convention_matches_right_handed_pitch_and_roll():
    x = np.array([1.0, 0.0])
    y = np.array([0.0, 1.0])
    offsets = module_vertical_offsets_m(x, y, roll_rad=1e-4, pitch_rad=2e-4)
    assert offsets[0] < 0.0  # +pitch about +Y lowers the +X point
    assert offsets[1] > 0.0  # +roll about +X raises the +Y point
    assert offsets[0] == pytest.approx(-2e-4, rel=1e-5)
    assert offsets[1] == pytest.approx(1e-4, rel=1e-5)


def test_timeline_reads_native_rollout_angular_rate_names():
    frame = {
        "time_s": [0.0, 1.0],
        "x_m": [0.0, 0.0],
        "y_m": [0.0, 0.0],
        "z_m": [0.002, 0.002],
        "roll_rad": [0.0, 0.0],
        "pitch_rad": [0.0, 0.0],
        "yaw_rad": [0.0, 0.0],
        "roll_rate_rad_s": [0.1, 0.2],
        "pitch_rate_rad_s": [0.3, 0.4],
        "yaw_rate_rad_s": [0.5, 0.6],
    }
    state = SimulationTimeline(pd.DataFrame(frame)).frame_at_index(1).rigid_body
    assert state.p_radps == pytest.approx(0.2)
    assert state.q_radps == pytest.approx(0.4)
    assert state.r_radps == pytest.approx(0.6)


def test_unsafe_condition_never_increases_lock_engagement():
    updated, status = _update_lock_state(
        0.35,
        0.90,
        docking_ready=True,
        detected_interlock=False,
        physical_unsafe=True,
        dt_s=0.02,
        rate_per_s=0.75,
    )
    assert updated == pytest.approx(0.35)
    assert status == "INHIBITED"


def test_module_positions_are_parameters_and_exported():
    px = (-0.3, 0.0, 0.3, -0.3, 0.3, -0.3, 0.0, 0.3)
    py = (0.2, 0.2, 0.2, 0.0, 0.0, -0.2, -0.2, -0.2)
    df, _ = simulate_rollout(RolloutParameters(dt_s=0.02, module_x_m=px, module_y_m=py))
    assert tuple(df.iloc[0][f"module_x_{i}_m"] for i in range(1, 9)) == pytest.approx(px)
    assert tuple(df.iloc[0][f"module_y_{i}_m"] for i in range(1, 9)) == pytest.approx(py)
    assert df.iloc[-1]["hard_lock_confirmed"] == 1


import pytest
