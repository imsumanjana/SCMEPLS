from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scmepls_studio.digital_twin import SimulationTimeline
from scmepls_studio.models.rollout_sim import RolloutParameters, simulate_rollout

from .coupling import StructuralFrameResult, StructuralTwinSolver
from .mass_properties import StructuralMassProperties, rollout_parameters_from_structure, structural_mass_properties
from .mesh import TetraMesh
from .mesher import tetrahedralize_geometry
from .structural_manifest import StructuralManifest, load_structural_manifest


@dataclass(frozen=True)
class MeshQualitySummary:
    minimum_mean_ratio: float
    median_mean_ratio: float
    maximum_mean_ratio: float
    poor_element_count: int
    inverted_element_count: int


@dataclass(frozen=True)
class StructuralCheckpoint:
    label: str
    frame_index: int
    time_s: float
    max_displacement_m: float
    max_von_mises_pa: float
    minimum_safety_factor: float
    equilibrium_residual_norm_n: float


@dataclass(frozen=True)
class StructuralValidationReport:
    geometry_path: str
    structural_manifest_path: str
    structural_component: str | None
    mass_scope: str
    node_count: int
    element_count: int
    volume_m3: float
    mass_kg: float
    centroid_m: tuple[float, float, float]
    inertia_centroid_kg_m2: tuple[tuple[float, float, float], ...]
    dimensions_m: tuple[float, float, float]
    mapping_max_snap_distance_m: float
    mesh_quality: MeshQualitySummary
    checkpoints: tuple[StructuralCheckpoint, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def tetra_mean_ratio_quality(mesh: TetraMesh) -> np.ndarray:
    """Return 0..1 tetrahedral mean-ratio shape quality (1 = regular tetrahedron)."""
    p = mesh.nodes_m[mesh.tetrahedra]
    edge_pairs = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    sum_edge_sq = np.zeros(mesh.element_count, dtype=float)
    for a, b in edge_pairs:
        d = p[:, a] - p[:, b]
        sum_edge_sq += np.einsum("ij,ij->i", d, d)
    volumes = mesh.element_volumes_m3
    numerator = 12.0 * np.power(3.0 * volumes, 2.0 / 3.0)
    return np.divide(numerator, sum_edge_sq, out=np.zeros_like(numerator), where=sum_edge_sq > 0)


def mesh_quality_summary(mesh: TetraMesh, poor_threshold: float = 0.10) -> MeshQualitySummary:
    quality = tetra_mean_ratio_quality(mesh)
    signed = mesh.signed_element_volumes_m3
    return MeshQualitySummary(
        minimum_mean_ratio=float(np.min(quality)),
        median_mean_ratio=float(np.median(quality)),
        maximum_mean_ratio=float(np.max(quality)),
        poor_element_count=int(np.count_nonzero(quality < poor_threshold)),
        inverted_element_count=int(np.count_nonzero(signed < 0.0)),
    )


def critical_frame_indices(history: pd.DataFrame) -> dict[str, int]:
    if history.empty:
        raise ValueError("Simulation history is empty.")
    labels: dict[str, int] = {"initial": 0, "final": len(history) - 1}

    def max_abs(column: str) -> int | None:
        if column not in history.columns:
            return None
        values = np.abs(pd.to_numeric(history[column], errors="coerce").to_numpy(dtype=float))
        if len(values) == 0 or not np.isfinite(values).any():
            return None
        return int(np.nanargmax(values))

    for label, column in (
        ("peak_support", "total_support_force_n"),
        ("peak_ax", "ax_mps2"),
        ("peak_ay", "ay_mps2"),
        ("peak_az", "az_mps2"),
        ("peak_fault", "fault_severity"),
    ):
        idx = max_abs(column)
        if idx is not None:
            labels[label] = idx
    if "lock_fraction" in history.columns:
        lock = pd.to_numeric(history["lock_fraction"], errors="coerce").to_numpy(dtype=float)
        transfer = np.flatnonzero((lock > 0.05) & (lock < 0.95))
        if len(transfer):
            labels["load_transfer"] = int(transfer[len(transfer) // 2])
    return labels


def _tuple3(values: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(values, dtype=float).ravel()
    return (float(v[0]), float(v[1]), float(v[2]))


def _matrix3(values: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    m = np.asarray(values, dtype=float)
    return tuple(_tuple3(row) for row in m)


def run_structural_validation(
    geometry_path: str | Path,
    structural_manifest_path: str | Path,
    *,
    base_rollout_parameters: RolloutParameters | None = None,
    source_unit: str = "m",
    axis_mode: str = "auto",
    rerun_geometry_coupled_dynamics: bool = True,
) -> tuple[StructuralValidationReport, TetraMesh, pd.DataFrame, dict[str, StructuralFrameResult]]:
    """Run the geometry→volume mesh→dynamics→node FEA validation chain.

    The structural manifest is intentionally mandatory. Real-model validation is
    rejected unless module, lock, wind, propulsion and kinematic constraint points
    are explicitly defined in the imported geometry coordinate system.
    """
    manifest: StructuralManifest = load_structural_manifest(structural_manifest_path)
    mesh = tetrahedralize_geometry(
        geometry_path,
        component_id=manifest.structural_component,
        source_unit=source_unit,
        axis_mode=axis_mode,
        options=manifest.meshing,
    )
    properties: StructuralMassProperties = structural_mass_properties(mesh, manifest.material)
    base = base_rollout_parameters or RolloutParameters()
    coupled = rollout_parameters_from_structure(base, properties, mass_scope=manifest.mass_scope)
    if rerun_geometry_coupled_dynamics:
        history, _ = simulate_rollout(coupled)
    else:
        history, _ = simulate_rollout(base)
    timeline = SimulationTimeline(history)
    twin = StructuralTwinSolver(
        mesh,
        manifest.material,
        timeline,
        coupled if rerun_geometry_coupled_dynamics else base,
        manifest.load_map,
        maximum_snap_distance_m=manifest.maximum_snap_distance_m,
    )

    frame_results: dict[str, StructuralFrameResult] = {}
    checkpoints: list[StructuralCheckpoint] = []
    for label, index in critical_frame_indices(history).items():
        result = twin.solve_frame(index)
        frame_results[label] = result
        checkpoints.append(
            StructuralCheckpoint(
                label=label,
                frame_index=index,
                time_s=result.time_s,
                max_displacement_m=result.fea.max_displacement_m,
                max_von_mises_pa=result.fea.max_von_mises_pa,
                minimum_safety_factor=result.fea.safety_factor_min,
                equilibrium_residual_norm_n=result.equilibrium_residual_norm_n,
            )
        )

    quality = mesh_quality_summary(mesh)
    warnings: list[str] = []
    if quality.inverted_element_count:
        warnings.append(f"{quality.inverted_element_count} tetrahedral elements have negative orientation.")
    if quality.poor_element_count:
        warnings.append(
            f"{quality.poor_element_count} tetrahedral elements have mean-ratio quality below 0.10; refine/repair before trusting local stress peaks."
        )
    total_mass_model = float(coupled.platform_mass_kg + coupled.payload_mass_kg)
    mass_error = abs(total_mass_model - properties.mass_kg) / max(properties.mass_kg, 1e-12)
    if manifest.mass_scope == "moving_assembly" and mass_error > 1e-9:
        warnings.append(f"Geometry-coupled rigid-body mass differs from integrated mesh mass by {100*mass_error:.6g}%.")
    if twin.mapping.maximum_distance_m > 0.5 * manifest.maximum_snap_distance_m:
        warnings.append(
            "At least one application point snaps more than half of the permitted mapping distance; inspect node placement locally."
        )

    report = StructuralValidationReport(
        geometry_path=str(Path(geometry_path)),
        structural_manifest_path=str(Path(structural_manifest_path)),
        structural_component=manifest.structural_component,
        mass_scope=manifest.mass_scope,
        node_count=mesh.node_count,
        element_count=mesh.element_count,
        volume_m3=properties.volume_m3,
        mass_kg=properties.mass_kg,
        centroid_m=_tuple3(properties.centroid_m),
        inertia_centroid_kg_m2=_matrix3(properties.inertia_centroid_kg_m2),
        dimensions_m=_tuple3(properties.dimensions_m),
        mapping_max_snap_distance_m=twin.mapping.maximum_distance_m,
        mesh_quality=quality,
        checkpoints=tuple(checkpoints),
        warnings=tuple(warnings),
    )
    return report, mesh, history, frame_results
