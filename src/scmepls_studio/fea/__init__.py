"""Volumetric meshing and finite-element analysis for imported SC-MEPLS geometry."""

from .coupling import (
    NodeMapping,
    StructuralFrameResult,
    StructuralLoadMap,
    StructuralTwinSolver,
    auto_structural_load_map,
    map_structural_locations,
    validate_structural_mapping,
)
from .load_distribution import (
    LoadPatch,
    add_patch_force,
    add_wrench_on_nodes,
    boundary_load_patch,
    patch_union,
)
from .mass_properties import (
    PayloadMassProperties,
    StructuralMassProperties,
    combine_platform_and_payload,
    rollout_parameters_from_structure,
    structural_mass_properties,
)
from .mesh import TetraMesh, TetraMeshError
from .mesher import MeshingOptions, VolumeMeshingError, select_structural_part, tetrahedralize_geometry
from .solver import (
    FEAResult,
    IsotropicMaterial,
    StaticElasticSolver,
    constrained_dofs,
    elasticity_matrix,
    three_point_kinematic_constraints,
    von_mises_from_stress,
)
from .structural_manifest import (
    StructuralManifest,
    StructuralManifestError,
    load_structural_manifest,
    structural_manifest_path_for_geometry,
)
from .validation import (
    MeshConvergencePoint,
    MeshQualitySummary,
    StructuralCheckpoint,
    StructuralValidationReport,
    critical_frame_indices,
    mesh_quality_summary,
    run_structural_validation,
    tetra_mean_ratio_quality,
)

__all__ = [
    "FEAResult",
    "IsotropicMaterial",
    "LoadPatch",
    "MeshConvergencePoint",
    "MeshQualitySummary",
    "MeshingOptions",
    "NodeMapping",
    "PayloadMassProperties",
    "StaticElasticSolver",
    "StructuralCheckpoint",
    "StructuralFrameResult",
    "StructuralLoadMap",
    "StructuralManifest",
    "StructuralManifestError",
    "StructuralMassProperties",
    "StructuralTwinSolver",
    "StructuralValidationReport",
    "TetraMesh",
    "TetraMeshError",
    "VolumeMeshingError",
    "add_patch_force",
    "add_wrench_on_nodes",
    "auto_structural_load_map",
    "boundary_load_patch",
    "combine_platform_and_payload",
    "constrained_dofs",
    "critical_frame_indices",
    "elasticity_matrix",
    "load_structural_manifest",
    "map_structural_locations",
    "mesh_quality_summary",
    "patch_union",
    "rollout_parameters_from_structure",
    "run_structural_validation",
    "select_structural_part",
    "structural_manifest_path_for_geometry",
    "structural_mass_properties",
    "tetra_mean_ratio_quality",
    "three_point_kinematic_constraints",
    "tetrahedralize_geometry",
    "validate_structural_mapping",
    "von_mises_from_stress",
]
