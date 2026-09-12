import numpy as np

from scmepls_studio.fea import (
    IsotropicMaterial,
    StaticElasticSolver,
    TetraMesh,
    constrained_dofs,
)


def test_node_by_node_tetra_fea_solves_displacement_stress_and_reactions():
    mesh = TetraMesh(
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ),
        np.array([[0, 1, 2, 3]], dtype=np.int64),
    )
    material = IsotropicMaterial(
        youngs_modulus_pa=70e9,
        poisson_ratio=0.3,
        density_kg_m3=2700.0,
        yield_strength_pa=250e6,
    )
    fixed = constrained_dofs(
        {
            0: (True, True, True),
            1: (False, True, True),
            2: (False, False, True),
        },
        mesh.node_count,
    )
    solver = StaticElasticSolver(mesh, material, fixed)
    loads = np.zeros((mesh.node_count, 3))
    loads[3, 0] = 1000.0
    result = solver.solve(loads, body_acceleration_mps2=(0.0, 0.0, 0.0))

    assert result.displacement_m.shape == (4, 3)
    assert result.element_stress_pa.shape == (1, 6)
    assert result.nodal_von_mises_pa.shape == (4,)
    assert result.max_displacement_m > 0.0
    assert result.max_von_mises_pa > 0.0
    assert np.all(np.isfinite(result.displacement_m))
    assert np.all(np.isfinite(result.element_stress_pa))
    assert np.linalg.norm(np.sum(result.reaction_n, axis=0) + np.sum(result.nodal_force_n, axis=0)) < 1e-6
