from scmepls_studio.models.control_response import ResponseParameters, calculate_response_metrics, simulate_step


def test_first_order_response():
    t, y = simulate_step(ResponseParameters(model_type="first_order", time_constant_s=1.0), 10, 2000)
    metrics = calculate_response_metrics(t, y)
    assert 2.0 < metrics.rise_time_s < 2.4
    assert 3.8 < metrics.settling_time_s < 4.2
    assert metrics.overshoot_percent < 0.1
