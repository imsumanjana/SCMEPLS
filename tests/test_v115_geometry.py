import json

import numpy as np
import pytest
import trimesh

from scmepls_studio.geometry import GeometryManifestError, load_geometry, load_geometry_manifest


def test_glb_auto_axis_conversion_y_up_to_z_up(tmp_path):
    path = tmp_path / "axis.glb"
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.creation.box(extents=(1.0, 2.0, 3.0)), node_name="PLATFORM")
    path.write_bytes(scene.export(file_type="glb"))
    asset = load_geometry(path, source_unit="m", axis_mode="auto")
    np.testing.assert_allclose(asset.dimensions_m, [1.0, 3.0, 2.0], atol=1e-8)


def test_geometry_manifest_validates_component_bindings(tmp_path):
    manifest_path = tmp_path / "assembly.manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "required_components": ["PLATFORM", "EM_M01"],
        "components": {
            "PLATFORM": {"role": "platform", "dynamic_group": "platform"},
            "EM_M01": {"role": "em_module", "simulation_module": 1},
        },
    }), encoding="utf-8")
    manifest = load_geometry_manifest(manifest_path, {"PLATFORM", "EM_M01"})
    assert manifest.components["EM_M01"].simulation_module == 1


def test_geometry_manifest_rejects_missing_required_component(tmp_path):
    manifest_path = tmp_path / "assembly.manifest.json"
    manifest_path.write_text(json.dumps({
        "schema_version": 1,
        "required_components": ["PLATFORM", "EM_M08"],
        "components": {"PLATFORM": {"role": "platform"}},
    }), encoding="utf-8")
    with pytest.raises(GeometryManifestError):
        load_geometry_manifest(manifest_path, {"PLATFORM"})
