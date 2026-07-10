import numpy as np
from scmepls_studio.models.vibration import calculate_metrics, generate_synthetic_timeseries, metrics_from_dataframe


def test_rms_known_sine():
    t = np.linspace(0, 10, 10001)
    a = np.sqrt(2) * np.sin(2*np.pi*t)
    m = calculate_metrics(t, a)
    assert abs(m.rms - 1.0) < 1e-3


def test_default_vibration_generation():
    df = generate_synthetic_timeseries(5, 100, {"A": 1.0, "B": 0.5})
    metrics = metrics_from_dataframe(df)
    assert len(metrics) == 2
    assert abs(metrics.loc[metrics.system == "A", "rms"].iloc[0] - 1.0) < 1e-6
