from pathlib import Path

import numpy as np
import pandas as pd

from scmepls_studio.digital_twin import PlaybackController, build_scene_bindings, component_transforms
from scmepls_studio.geometry import GeometryAsset, GeometryPart
from scmepls_studio.digital_twin.physics import SimulationTimeline


def _asset() -> GeometryAsset:
    part_track = GeometryPart("TRACK", np.array([[0,0,0],[1,0,0],[0,1,0]], float), np.array([[0,1,2]]), None)
    part_platform = GeometryPart("PLATFORM", np.array([[0,0,0],[1,0,0],[0,1,0]], float), np.array([[0,1,2]]), None)
    return GeometryAsset(Path("a.glb"), "m", "as_stored", 1.0, (part_track, part_platform), ())


def test_playback_and_component_transforms():
    history = pd.DataFrame([
        {"time_s":0.0,"x_m":0.0,"y_m":0.0,"z_m":0.0,"roll_rad":0.0,"pitch_rad":0.0,"yaw_rad":0.0},
        {"time_s":1.0,"x_m":1.0,"y_m":0.0,"z_m":0.0,"roll_rad":0.0,"pitch_rad":0.0,"yaw_rad":0.0},
    ])
    timeline = SimulationTimeline(history)
    controller = PlaybackController(timeline, speed=2.0)
    controller.play()
    frame = controller.advance(0.5)
    assert frame.index == 1
    asset = _asset()
    registry = build_scene_bindings(asset)
    transforms = component_transforms(registry, timeline, frame)
    np.testing.assert_allclose(transforms["TRACK"], np.eye(4))
    np.testing.assert_allclose(transforms["PLATFORM"][:3, 3], [1.0, 0.0, 0.0])
