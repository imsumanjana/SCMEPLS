from scmepls_studio.models.rollout_sim import RolloutParameters, simulate_rollout


def test_rollout_runs_and_returns_metrics():
    df, metrics = simulate_rollout(RolloutParameters(dt_s=0.02, end_time_s=80.0))
    assert len(df) > 3000
    assert metrics["design_weight_n"] > 0
    assert "total_support_force_n" in df.columns
    assert df["lock_fraction"].iloc[-1] == 1.0
    assert df["mode_name"].iloc[-1] == "HARD_LOCK"
