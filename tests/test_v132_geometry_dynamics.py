import numpy as np
import pytest

from scmepls_studio.fea import (
    IsotropicMaterial,
    TetraMesh,
    rollout_parameters_from_structure,
    structural_mass_properties,
)
from scmepls_studio.models.rollout_sim import RolloutParameters, simulate_rollout


def test_tetra_mass_centroid_and_inertia_are_integrated_from_volume():
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
    material = IsotropicMaterial(density_kg_m3=600.0)
    props = structural_mass_properties(mesh, material)
    assert props.volume_m3 == pytest.approx(1.0 / 6.0)
    assert props.mass_kg == pytest.approx(100.0)
    assert props.centroid_m == pytest.approx(np.array([0.25, 0.25, 0.25]))
    assert np.all(np.linalg.eigvalsh(props.inertia_centroid_kg_m2) > 0.0)


def test_geometry_properties_can_drive_rollout_mass_inertia_and_dimensions():
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
    props = structural_mass_properties(mesh, IsotropicMaterial(density_kg_m3=600.0))
    base = RolloutParameters(payload_mass_kg=0.0, dt_s=0.02, end_time_s=75.0)
    coupled = rollout_parameters_from_structure(base, props, mass_scope="moving_assembly")
    assert coupled.platform_mass_kg == pytest.approx(100.0)
    assert coupled.body_length_m == pytest.approx(1.0)
    assert coupled.inertia_xx_kg_m2 == pytest.approx(props.inertia_centroid_kg_m2[0, 0])
    history, metrics = simulate_rollout(coupled)
    assert len(history) > 100
    assert metrics["parameters"]["body_length_m"] == pytest.approx(1.0)
