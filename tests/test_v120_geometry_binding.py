from pathlib import Path

import numpy as np

from scmepls_studio.digital_twin import build_scene_bindings
from scmepls_studio.geometry import GeometryAsset, GeometryPart
from scmepls_studio.geometry.manifest import load_geometry_manifest


def _part(name: str) -> GeometryPart:
    return GeometryPart(
        component_id=name,
        vertices_m=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        cell_rgb=None,
    )


def test_manifest_binding_allows_multiple_parts_for_one_module(tmp_path):
    asset = GeometryAsset(Path("assembly.glb"), "m", "as_stored", 1.0, (_part("EM_M01"), _part("PNEU_M01")), ())
    manifest_path = tmp_path / "assembly.manifest.json"
    manifest_path.write_text(
        '{"schema_version":1,"platform_origin_m":[0.1,0.2,0.3],"components":'
        '{"EM_M01":{"role":"electromagnetic","dynamic_group":"module","simulation_module":1},'
        '"PNEU_M01":{"role":"pneumatic","dynamic_group":"module","simulation_module":1}}}',
        encoding="utf-8",
    )
    manifest = load_geometry_manifest(manifest_path, {"EM_M01", "PNEU_M01"})
    registry = build_scene_bindings(asset, manifest)
    assert registry.components_for_module(1) == ("EM_M01", "PNEU_M01")
    assert registry.for_component("EM_M01").pivot_m == (0.1, 0.2, 0.3)
    assert registry.for_component("EM_M01").source == "manifest"


def test_conservative_name_inference_keeps_track_stationary():
    asset = GeometryAsset(Path("assembly.glb"), "m", "as_stored", 1.0, (_part("TRACK"), _part("PLATFORM")), ())
    registry = build_scene_bindings(asset)
    assert registry.for_component("TRACK").dynamic_group == "world"
    assert registry.for_component("PLATFORM").dynamic_group == "platform"
