from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scmepls_studio.digital_twin import SimulationTimeline
from scmepls_studio.models.rollout_sim import RolloutParameters, simulate_rollout

from .coupling import StructuralFrameResult, StructuralTwinSolver
from .mass_properties import StructuralMassProperties, rollout_parameters_from_structure, structural_mass_properties
from .mesh import TetraMesh
from .mesher import MeshingOptions, tetrahedralize_geometry
from .structural_manifest import StructuralManifest, load_structural_manifest


@dataclass(frozen=True)
class MeshQualitySummary:
    minimum_mean_ratio: float
    median_mean_ratio: float
    maximum_mean_ratio: float
    poor_element_count: int
    nonpositive_jacobian_count: int

    @property
    def inverted_element_count(self) -> int:
        """Backward-compatible alias; connectivity orientation is normalized first."""
        return self.nonpositive_jacobian_count


@dataclass(frozen=True)
class StructuralCheckpoint:
    label: str
    frame_index: int
    time_s: float
    max_displacement_m: float
    max_von_mises_pa: float
    minimum_safety_factor: float
    pre_fea_force_residual_norm_n: float
    pre_fea_moment_residual_norm_nm: float
    equilibrium_residual_norm_n: float


@dataclass(frozen=True)
class MeshConvergencePoint:
    element_size_m: float
    node_count: int
    element_count: int
    max_displacement_m: float
    max_von_mises_pa: float
    minimum_safety_factor: float


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
    module_points_m: tuple[tuple[float, float, float], ...]
    mapping_max_snap_distance_m: float
    mesh_quality: MeshQualitySummary
    checkpoints: tuple[StructuralCheckpoint, ...]
    convergence: tuple[MeshConvergencePoint, ...]
    convergence_displacement_change_percent: float | None
    convergence_stress_change_percent: float | None
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
        nonpositive_jacobian_count=int(np.count_nonzero(signed <= 0.0)),
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


def _convergence_options(base: MeshingOptions, element_size_m: float) -> MeshingOptions:
    size = float(element_size_m)
    ratio_min = None
    ratio_max = None
    if base.min_element_size_m is not None:
        ratio_min = size * float(base.min_element_size_m) / float(base.element_size_m)
    if base.max_element_size_m is not None:
        ratio_max = size * float(base.max_element_size_m) / float(base.element_size_m)
    return replace(base, element_size_m=size, min_element_size_m=ratio_min, max_element_size_m=ratio_max)


def _run_convergence(
    geometry_path: str | Path,
    manifest: StructuralManifest,
    history: pd.DataFrame,
    rollout_parameters: RolloutParameters,
    representative_index: int,
    *,
    source_unit: str,
    axis_mode: str,
) -> tuple[tuple[MeshConvergencePoint, ...], float | None, float | None]:
    base_size = float(manifest.meshing.element_size_m)
    sizes = (1.35 * base_size, base_size, 0.75 * base_size)
    points: list[MeshConvergencePoint] = []
    for size in sizes:
        mesh = tetrahedralize_geometry(
            geometry_path,
            component_id=manifest.structural_component,
            source_unit=source_unit,
            axis_mode=axis_mode,
            options=_convergence_options(manifest.meshing, size),
        )
        timeline = SimulationTimeline(history)
        twin = StructuralTwinSolver(
            mesh,
            manifest.material,
            timeline,
            rollout_parameters,
            manifest.load_map,
            maximum_snap_distance_m=manifest.maximum_snap_distance_m,
            payload=manifest.payload,
        )
        result = twin.solve_frame(representative_index)
        points.append(
            MeshConvergencePoint(
                element_size_m=size,
                node_count=mesh.node_count,
                element_count=mesh.element_count,
                max_displacement_m=result.fea.max_displacement_m,
                max_von_mises_pa=result.fea.max_von_mises_pa,
                minimum_safety_factor=result.fea.safety_factor_min,
            )
        )
    fine_prev, fine = points[-2], points[-1]
    displacement_change = 100.0 * abs(fine.max_displacement_m - fine_prev.max_displacement_m) / max(abs(fine.max_displacement_m), 1e-15)
    stress_change = 100.0 * abs(fine.max_von_mises_pa - fine_prev.max_von_mises_pa) / max(abs(fine.max_von_mises_pa), 1e-9)
    return tuple(points), float(displacement_change), float(stress_change)


def run_structural_validation(
    geometry_path: str | Path,
    structural_manifest_path: str | Path,
    *,
    base_rollout_parameters: RolloutParameters | None = None,
    source_unit: str = "m",
    axis_mode: str = "auto",
    rerun_geometry_coupled_dynamics: bool = True,
    run_mesh_convergence: bool = False,
) -> tuple[StructuralValidationReport, TetraMesh, pd.DataFrame, dict[str, StructuralFrameResult]]:
    """Run geometry → volume mesh → coupled dynamics → node FEA validation.

    The structural manifest is mandatory. Geometry-derived dynamics use integrated
    mass/inertia and the exact manifest M1–M8 XY coordinates. ``platform_only``
    analysis additionally requires explicit payload mass properties and payload
    support patches. Optional three-level mesh convergence is available for final
    quantitative structural claims.
    """
    manifest = load_structural_manifest(structural_manifest_path)
    mesh = tetrahedralize_geometry(
        geometry_path,
        component_id=manifest.structural_component,
        source_unit=source_unit,
        axis_mode=axis_mode,
        options=manifest.meshing,
    )
    properties: StructuralMassProperties = structural_mass_properties(mesh, manifest.material)
    base = base_rollout_parameters or RolloutParameters()
    coupled = rollout_parameters_from_structure(
        base,
        properties,
        mass_scope=manifest.mass_scope,
        module_points_m=manifest.load_map.module_points_m,
        payload=manifest.payload,
    )
    history, _ = simulate_rollout(coupled if rerun_geometry_coupled_dynamics else base)
    active_parameters = coupled if rerun_geometry_coupled_dynamics else base
    timeline = SimulationTimeline(history)
    twin = StructuralTwinSolver(
        mesh,
        manifest.material,
        timeline,
        active_parameters,
        manifest.load_map,
        maximum_snap_distance_m=manifest.maximum_snap_distance_m,
        payload=manifest.payload,
    )

    frame_results: dict[str, StructuralFrameResult] = {}
    checkpoints: list[StructuralCheckpoint] = []
    indices = critical_frame_indices(history)
    for label, index in indices.items():
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
                pre_fea_force_residual_norm_n=result.pre_fea_force_residual_norm_n,
                pre_fea_moment_residual_norm_nm=result.pre_fea_moment_residual_norm_nm,
                equilibrium_residual_norm_n=result.equilibrium_residual_norm_n,
            )
        )

    convergence: tuple[MeshConvergencePoint, ...] = ()
    convergence_disp: float | None = None
    convergence_stress: float | None = None
    if run_mesh_convergence:
        representative_index = indices.get("peak_support", indices.get("load_transfer", len(history) - 1))
        convergence, convergence_disp, convergence_stress = _run_convergence(
            geometry_path,
            manifest,
            history,
            active_parameters,
            representative_index,
            source_unit=source_unit,
            axis_mode=axis_mode,
        )

    quality = mesh_quality_summary(mesh)
    warnings: list[str] = []
    if quality.nonpositive_jacobian_count:
        warnings.append(
            f"{quality.nonpositive_jacobian_count} tetrahedral elements have non-positive Jacobian after orientation normalization."
        )
    if quality.poor_element_count:
        warnings.append(
            f"{quality.poor_element_count} tetrahedral elements have mean-ratio quality below 0.10; refine/repair before trusting local stress peaks."
        )
    if twin.mapping.maximum_distance_m > 0.5 * manifest.maximum_snap_distance_m:
        warnings.append(
            "At least one application point snaps more than half of the permitted mapping distance; inspect node placement locally."
        )
    if not run_mesh_convergence:
        warnings.append(
            "Mesh convergence was not run. Do not use local stress maxima as final quantitative design evidence until a coarse/base/fine convergence study is completed."
        )
    elif convergence_stress is not None and convergence_stress > 10.0:
        warnings.append(
            f"Fine-grid maximum von Mises stress changed by {convergence_stress:.3g}% from the base grid; structural stress is not mesh-converged."
        )
    if convergence_disp is not None and convergence_disp > 5.0:
        warnings.append(
            f"Fine-grid maximum displacement changed by {convergence_disp:.3g}% from the base grid; displacement is not mesh-converged."
        )

    max_force_residual = max((cp.pre_fea_force_residual_norm_n for cp in checkpoints), default=0.0)
    max_moment_residual = max((cp.pre_fea_moment_residual_norm_nm for cp in checkpoints), default=0.0)
    design_weight = float(active_parameters.platform_mass_kg + active_parameters.payload_mass_kg) * 9.81
    characteristic_moment = max(design_weight * max(float(np.max(mesh.dimensions_m)), 1e-6), 1e-6)
    if max_force_residual > 0.05 * max(design_weight, 1e-6):
        warnings.append(
            f"Pre-FEA force balance residual reaches {max_force_residual:.6g} N (>5% of design weight). Inspect load mapping/model completeness."
        )
    if max_moment_residual > 0.05 * characteristic_moment:
        warnings.append(
            f"Pre-FEA moment balance residual reaches {max_moment_residual:.6g} N·m (>5% characteristic moment). Inspect generalized load transfer."
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
        module_points_m=tuple(_tuple3(row) for row in manifest.load_map.module_points_m),
        mapping_max_snap_distance_m=twin.mapping.maximum_distance_m,
        mesh_quality=quality,
        checkpoints=tuple(checkpoints),
        convergence=convergence,
        convergence_displacement_change_percent=convergence_disp,
        convergence_stress_change_percent=convergence_stress,
        warnings=tuple(warnings),
    )
    return report, mesh, history, frame_results
