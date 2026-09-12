import pandas as pd

from scmepls_studio.digital_twin import ResultField, SceneBinding, SimulationTimeline, component_result


def test_module_result_field_uses_global_scale_and_binding():
    history = pd.DataFrame([
        {"time_s":0.0,"x_m":0.0,"y_m":0.0,"z_m":0.0,"roll_rad":0.0,"pitch_rad":0.0,"yaw_rad":0.0,"em_force_1_n":10.0,"health_1":1.0},
        {"time_s":1.0,"x_m":0.0,"y_m":0.0,"z_m":0.0,"roll_rad":0.0,"pitch_rad":0.0,"yaw_rad":0.0,"em_force_1_n":30.0,"health_1":0.5},
    ])
    timeline = SimulationTimeline(history)
    field = ResultField(timeline, "em_force")
    binding = SceneBinding("EM_M01", "electromagnetic", "module", 1, (0,0,0), "test")
    result = component_result(binding, timeline.frame_at_index(1), field)
    assert result.value == 30.0
    assert result.unit == "N"
    assert result.normalized == 1.0
    assert result.module == 1
