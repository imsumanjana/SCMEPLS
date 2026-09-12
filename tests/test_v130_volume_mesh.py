import numpy as np
import pytest

from scmepls_studio.fea import TetraMesh, TetraMeshError


def test_single_tetra_has_true_volume_boundary_and_nearest_nodes():
    mesh = TetraMesh(
        nodes_m=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        tetrahedra=np.array([[0, 1, 2, 3]], dtype=np.int64),
        component_id="PLATFORM",
    )
    assert mesh.node_count == 4
    assert mesh.element_count == 1
    assert mesh.total_volume_m3 == pytest.approx(1.0 / 6.0)
    assert mesh.boundary_faces.shape == (4, 3)
    idx, distance = mesh.nearest_nodes(np.array([[0.01, 0.0, 0.0]]))
    assert int(idx[0]) == 0
    assert float(distance[0]) == pytest.approx(0.01)


def test_zero_volume_tetra_is_rejected():
    with pytest.raises(TetraMeshError, match="zero-volume"):
        TetraMesh(
            nodes_m=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [1.0, 1.0, 0.0],
                ]
            ),
            tetrahedra=np.array([[0, 1, 2, 3]], dtype=np.int64),
        )
