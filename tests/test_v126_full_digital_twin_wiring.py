from pathlib import Path

import numpy as np

from scmepls_studio.digital_twin import MeshRegistry, PlaybackController, ResultField, SimulationTimeline, build_scene_bindings, component_transforms, digital_twin_history_figure
from scmepls_studio.geometry import GeometryAsset, GeometryPart
from scmepls_studio.models.rollout_sim import RolloutParameters, simulate_rollout


def _part(name: str) -> GeometryPart:
    return GeometryPart(name,np.array([[0.0,0.0,0.0],[0.2,0.0,0.0],[0.0,0.2,0.0]]),np.array([[0,1,2]],dtype=np.int64),None)


def test_geometry_mesh_physics_animation_results_and_plotting_are_composable():
    history,_ = simulate_rollout(RolloutParameters(dt_s=0.02,end_time_s=75.0))
    timeline = SimulationTimeline(history); controller = PlaybackController(timeline); controller.seek_time(30.0)
    asset = GeometryAsset(Path("assembly.glb"),"m","as_stored",1.0,(_part("TRACK"),_part("PLATFORM"),_part("EM_M01")),())
    bindings = build_scene_bindings(asset); meshes = MeshRegistry.from_asset(asset,bindings); frame = timeline.frame_at_time(30.0)
    transforms = component_transforms(bindings,timeline,frame); field = ResultField(timeline,"em_force")
    assert meshes.triangle_count == 3; assert np.allclose(transforms["TRACK"],np.eye(4)); assert not np.allclose(transforms["PLATFORM"],np.eye(4)); assert field.range.maximum >= field.range.minimum
    fig = digital_twin_history_figure(timeline,bindings.for_component("EM_M01"),"em_force",30.0); assert len(fig.axes) == 2
