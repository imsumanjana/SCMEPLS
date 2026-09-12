from scmepls_studio.models.rollout_sim import RolloutParameters, simulate_rollout


def test_hard_lock_seats_and_constrains_final_state():
    params = RolloutParameters(dt_s=0.02)
    df, metrics = simulate_rollout(params)
    final = df.iloc[-1]
    assert final["lock_fraction"] == 1.0
    assert abs(final["mean_gap_m"] - params.initial_gap_m) < 0.003
    assert metrics["final_position_error_mm"] < 50.0
    assert metrics["final_lateral_error_mm"] < 20.0
    assert metrics["final_yaw_error_deg"] < 3.0


def test_seating_reference_returns_to_initial_gap():
    params = RolloutParameters(dt_s=0.02)
    df, _ = simulate_rollout(params)
    assert abs(float(df.iloc[-1]["gap_ref_m"]) - params.initial_gap_m) < 1e-12
