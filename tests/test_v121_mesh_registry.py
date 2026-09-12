from pathlib import Path

import numpy as np

from scmepls_studio.digital_twin import MeshRegistry, actor_style, build_scene_bindings
from scmepls_studio.geometry import GeometryAsset, GeometryPart


def test_mesh_registry_preserves_geometry_and_component_identity():
    part = GeometryPart(
        component_id="PLATFORM",
        vertices_m=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        faces=np.array([[0, 1, 2]], dtype=np.int64),
        cell_rgb=np.array([[10, 20, 30]], dtype=np.uint8),
    )
    asset = GeometryAsset(Path("assembly.glb"), "m", "as_stored", 1.0, (part,), ())
    registry = MeshRegistry.from_asset(asset, build_scene_bindings(asset))
    record = registry.for_component("PLATFORM")
    assert record.polydata.n_points == 3
    assert record.polydata.n_cells == 1
    assert registry.component_by_index[1] == "PLATFORM"
    assert tuple(record.polydata.field_data["scmepls_component_id"])[0] == "PLATFORM"
    np.testing.assert_allclose(record.base_centroid_m, [1 / 3, 1 / 3, 0.0])
    assert actor_style(record, "wireframe")["style"] == "wireframe"
