import numpy as np
import pytest

from scmepls_studio.models.rollout_sim import RolloutParameters, simulate_rollout


def test_rollout_rejects_coarse_time_step():
    with pytest.raises(ValueError):
        simulate_rollout(RolloutParameters(dt_s=0.03))


def test_rollout_exports_physical_docking_state():
    df, _ = simulate_rollout(RolloutParameters(dt_s=0.02))
    assert "docking_ready" in df.columns
    assert np.isfinite(df["vibration_rms_proxy"]).all()
    assert df["lock_fraction"].max() <= 1.0


def test_sensor_fault_does_not_make_true_gap_nonfinite():
    df, _ = simulate_rollout(RolloutParameters(dt_s=0.02, sensor_fault_index=5))
    gap_columns = [f"gap_{i}_m" for i in range(1, 9)]
    assert np.isfinite(df[gap_columns].to_numpy()).all()
