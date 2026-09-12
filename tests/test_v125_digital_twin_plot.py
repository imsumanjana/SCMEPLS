import pandas as pd

from scmepls_studio.digital_twin import SceneBinding, SimulationTimeline, digital_twin_history_figure


def test_module_history_plot_contains_metric_and_gap_axes():
    history = pd.DataFrame([
        {"time_s":0.0,"x_m":0.0,"y_m":0.0,"z_m":0.0,"roll_rad":0.0,"pitch_rad":0.0,"yaw_rad":0.0,"em_force_1_n":10.0,"gap_1_m":0.002},
        {"time_s":1.0,"x_m":0.1,"y_m":0.0,"z_m":0.01,"roll_rad":0.0,"pitch_rad":0.0,"yaw_rad":0.0,"em_force_1_n":20.0,"gap_1_m":0.010},
    ])
    timeline = SimulationTimeline(history)
    binding = SceneBinding("EM_M01", "electromagnetic", "module", 1, (0,0,0), "test")
    fig = digital_twin_history_figure(timeline, binding, "em_force", 0.5)
    assert len(fig.axes) == 2
    assert fig.axes[0].get_ylabel() == "EM force (N)"
    assert fig.axes[1].get_ylabel() == "Gap (mm)"
