from __future__ import annotations

import numpy as np
import pytest
import trimesh

from scmepls_studio.geometry import GeometryImportError, load_geometry


def test_stl_import_respects_explicit_mm_units(tmp_path):
    path = tmp_path / "platform.stl"
    trimesh.creation.box(extents=(1000.0, 500.0, 250.0)).export(path)
    asset = load_geometry(path, source_unit="mm")
    assert len(asset.parts) == 1
    np.testing.assert_allclose(asset.dimensions_m, [1.0, 0.5, 0.25], rtol=0, atol=1e-9)
    assert any("unitless" in warning for warning in asset.warnings)


def test_glb_import_preserves_named_scene_node(tmp_path):
    path = tmp_path / "assembly.glb"
    scene = trimesh.Scene()
    scene.add_geometry(
        trimesh.creation.box(extents=(1.0, 0.5, 0.2)),
        node_name="PLATFORM",
        geom_name="platform_mesh",
    )
    path.write_bytes(scene.export(file_type="glb"))
    asset = load_geometry(path, source_unit="m", axis_mode="as_stored")
    assert [part.component_id for part in asset.parts] == ["PLATFORM"]
    np.testing.assert_allclose(asset.dimensions_m, [1.0, 0.5, 0.2], rtol=0, atol=1e-8)


def test_geometry_import_rejects_unsupported_extension(tmp_path):
    path = tmp_path / "model.obj"
    path.write_text("# not used", encoding="utf-8")
    with pytest.raises(GeometryImportError):
        load_geometry(path)
