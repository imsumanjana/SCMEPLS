import json

import numpy as np
import pandas as pd

from scmepls_studio.digital_twin import SimulationTimeline
from scmepls_studio.fea import (
    IsotropicMaterial,
    StructuralTwinSolver,
    TetraMesh,
    auto_structural_load_map,
    load_structural_manifest,
)
from scmepls_studio.models.rollout_sim import RolloutParameters


def _cube_mesh() -> TetraMesh:
    nodes = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
        ],
        dtype=float,
    )
    tets = np.array(
        [
            [0, 1, 3, 4],
            [1, 2, 3, 6],
            [1, 4, 5, 6],
            [3, 4, 6, 7],
            [1, 3, 4, 6],
        ],
        dtype=np.int64,
    )
    return TetraMesh(nodes, tets, component_id="PLATFORM")


def _timeline() -> SimulationTimeline:
    rows = []
    for i, t in enumerate((0.0, 0.02)):
        row = {
            "time_s": t,
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": 0.01,
            "roll_rad": 0.0,
            "pitch_rad": 0.0,
            "yaw_rad": 0.0,
            "vx_mps": 0.0,
            "vy_mps": 0.0,
            "vz_mps": 0.0,
            "ax_mps2": 0.0,
            "ay_mps2": 0.0,
            "az_mps2": 0.0,
            "roll_rate_rad_s": 0.0,
            "pitch_rate_rad_s": 0.0,
            "yaw_rate_rad_s": 0.0,
            "gap_ref_m": 0.01,
            "lock_fraction": 0.0,
            "wind_force_n": 0.0,
            "mode_name": "LEVITATE",
        }
        for module in range(1, 9):
            row[f"em_force_{module}_n"] = 12.0 + i
            row[f"pneumatic_force_{module}_n"] = 0.0
        rows.append(row)
    return SimulationTimeline(pd.DataFrame(rows))


def test_structural_twin_maps_rollout_forces_to_nodes_and_solves_frame():
    mesh = _cube_mesh()
    material = IsotropicMaterial(
        youngs_modulus_pa=5e8,
        poisson_ratio=0.30,
        density_kg_m3=100.0,
        yield_strength_pa=2e8,
    )
    load_map = auto_structural_load_map(mesh)
    twin = StructuralTwinSolver(
        mesh,
        material,
        _timeline(),
        RolloutParameters(platform_mass_kg=100.0, payload_mass_kg=0.0),
        load_map,
        maximum_snap_distance_m=0.2,
    )
    result = twin.solve_frame(0)
    assert result.fea.displacement_m.shape == (mesh.node_count, 3)
    assert result.fea.nodal_von_mises_pa.shape == (mesh.node_count,)
    assert result.mapping.module_nodes.shape == (8,)
    assert result.equilibrium_residual_norm_n < 1e-6
    grid = result.to_pyvista(mesh, deformation_scale=10.0)
    assert "von_mises_pa" in grid.point_data
    assert "element_von_mises_pa" in grid.cell_data


def test_structural_manifest_requires_explicit_validated_mapping(tmp_path):
    path = tmp_path / "assembly.fea.json"
    payload = {
        "schema_version": 1,
        "structural_component": "PLATFORM",
        "mass_scope": "moving_assembly",
        "material": {
            "youngs_modulus_pa": 69e9,
            "poisson_ratio": 0.33,
            "density_kg_m3": 2700,
            "yield_strength_pa": 240e6,
        },
        "mesh": {"element_size_m": 0.05},
        "maximum_snap_distance_m": 0.05,
        "mapping": {
            "module_points_m": [[0, 0, 0]] * 8,
            "constraint_points_m": [[0, 0, 0], [1, 0, 0], [0, 1, 0]],
            "propulsion_point_m": [0, 0, 0],
            "wind_point_m": [0, 1, 1],
            "lock_points_m": [[1, 0, 0], [1, 1, 0]],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    manifest = load_structural_manifest(path)
    assert manifest.structural_component == "PLATFORM"
    assert manifest.material.density_kg_m3 == 2700
    assert manifest.load_map.module_points_m.shape == (8, 3)
