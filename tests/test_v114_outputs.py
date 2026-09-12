import numpy as np

from scmepls_studio.models.rollout_sim import RolloutParameters, simulate_rollout


def test_module_level_validation_outputs_exist():
    df, _ = simulate_rollout(RolloutParameters(dt_s=0.02))
    required = [
        "mode_name", "y_m", "vy_mps", "vz_mps", "ay_mps2",
        "roll_rate_rad_s", "pitch_rate_rad_s", "yaw_rate_rad_s",
    ]
    for i in range(1, 9):
        required.extend([
            f"gap_{i}_m", f"sensor_gap_{i}_m", f"estimated_gap_{i}_m",
            f"coil_current_{i}_a", f"em_force_{i}_n",
            f"pressure_{i}_pa", f"pneumatic_force_{i}_n",
        ])
    missing = [name for name in required if name not in df.columns]
    assert not missing
    assert np.isfinite(df[[c for c in required if c != "mode_name"]].to_numpy()).all()
