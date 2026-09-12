import numpy as np
import pandas as pd

from scmepls_studio.digital_twin import SimulationTimeline


def test_timeline_maps_pose_and_module_results_without_feedback_to_physics():
    history = pd.DataFrame(
        [
            {"time_s": 0.0, "x_m": 0.0, "y_m": 0.0, "z_m": 0.002, "roll_rad": 0.0, "pitch_rad": 0.0, "yaw_rad": 0.0,
             "gap_1_m": 0.002, "coil_current_1_a": 2.0, "em_force_1_n": 100.0},
            {"time_s": 1.0, "x_m": 1.0, "y_m": 0.2, "z_m": 0.012, "roll_rad": 0.0, "pitch_rad": 0.0, "yaw_rad": 0.0,
             "gap_1_m": 0.010, "coil_current_1_a": 3.0, "em_force_1_n": 120.0, "mode_name": "ROLLOUT"},
        ]
    )
    timeline = SimulationTimeline(history)
    frame = timeline.frame_at_time(0.9)
    assert frame.index == 1
    assert frame.mode == "ROLLOUT"
    assert frame.modules[0].coil_current_a == 3.0
    transform = timeline.relative_transform(frame, (0.0, 0.0, 0.0))
    np.testing.assert_allclose(transform[:3, 3], [1.0, 0.2, 0.01], atol=1e-12)
    np.testing.assert_allclose(transform[:3, :3], np.eye(3), atol=1e-12)
