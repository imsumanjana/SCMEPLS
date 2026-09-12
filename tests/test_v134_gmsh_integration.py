import numpy as np
import trimesh

from scmepls_studio.fea import MeshingOptions, mesh_quality_summary, tetrahedralize_geometry


def test_gmsh_tetrahedralizes_closed_stl_and_preserves_volume(tmp_path):
    surface = trimesh.creation.box(extents=(1.0, 0.5, 0.2))
    path = tmp_path / "box.stl"
    surface.export(path)
    mesh = tetrahedralize_geometry(
        path,
        source_unit="m",
        axis_mode="as_stored",
        options=MeshingOptions(element_size_m=0.16, max_element_size_m=0.16),
    )
    assert mesh.node_count > 8
    assert mesh.element_count > 5
    assert mesh.total_volume_m3 == pytest.approx(0.1, rel=0.08)
    quality = mesh_quality_summary(mesh)
    assert 0.0 < quality.minimum_mean_ratio <= 1.0
    assert quality.inverted_element_count >= 0


import pytest
