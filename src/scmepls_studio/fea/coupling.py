from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from scmepls_studio.coordinates import world_vector_to_body
from scmepls_studio.digital_twin.physics import SimulationTimeline
from scmepls_studio.models.rollout_sim import RolloutParameters

from .load_distribution import LoadPatch, add_patch_force, add_wrench_on_nodes, boundary_load_patch, patch_union
from .mass_properties import PayloadMassProperties, StructuralMassProperties, structural_mass_properties
from .mesh import TetraMesh
from .solver import FEAResult, IsotropicMaterial, StaticElasticSolver, three_point_kinematic_constraints

_MODULE_X_FRACTION = np.array([-1.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 1.0])
_MODULE_Y_FRACTION = np.array([1.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, -1.0])


@dataclass(frozen=True)
class StructuralLoadMap:
    """Physical load/constraint locations and patch radii in the geometry frame."""

    module_points_m: np.ndarray
    constraint_points_m: np.ndarray
    propulsion_point_m: np.ndarray
    wind_point_m: np.ndarray
    lock_points_m: np.ndarray
    payload_support_points_m: np.ndarray | None = None
    module_patch_radius_m: float = 0.025
    propulsion_patch_radius_m: float = 0.035
    wind_patch_radius_m: float = 0.040
    lock_patch_radius_m: float = 0.025
    payload_patch_radius_m: float = 0.040
    source: str = "explicit"

    def __post_init__(self) -> None:
        expected = {
            "module_points_m": (8, 3),
            "constraint_points_m": (3, 3),
            "propulsion_point_m": (3,),
            "wind_point_m": (3,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} must have shape {shape} and finite coordinates.")
            object.__setattr__(self, name, np.ascontiguousarray(value))
        locks = np.asarray(self.lock_points_m, dtype=float)
        if locks.ndim != 2 or locks.shape[1] != 3 or len(locks) < 1 or not np.all(np.isfinite(locks)):
            raise ValueError("lock_points_m must contain one or more finite XYZ points.")
        object.__setattr__(self, "lock_points_m", np.ascontiguousarray(locks))
        if self.payload_support_points_m is not None:
            payload = np.asarray(self.payload_support_points_m, dtype=float)
            if payload.ndim != 2 or payload.shape[1] != 3 or len(payload) < 1 or not np.all(np.isfinite(payload)):
                raise ValueError("payload_support_points_m must have shape (N, 3) with N >= 1.")
            object.__setattr__(self, "payload_support_points_m", np.ascontiguousarray(payload))
        for name in (
            "module_patch_radius_m",
            "propulsion_patch_radius_m",
            "wind_patch_radius_m",
            "lock_patch_radius_m",
            "payload_patch_radius_m",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")


@dataclass(frozen=True)
class NodeMapping:
    module_patches: tuple[LoadPatch, ...]
    propulsion_patch: LoadPatch
    wind_patch: LoadPatch
    lock_patches: tuple[LoadPatch, ...]
    payload_patches: tuple[LoadPatch, ...]
    constraint_nodes: np.ndarray
    constraint_distance_m: np.ndarray

    @property
    def module_nodes(self) -> np.ndarray:
        return np.array([patch.node_indices[int(np.argmax(patch.weights))] for patch in self.module_patches], dtype=np.int64)

    @property
    def module_distance_m(self) -> np.ndarray:
        return np.array([patch.nearest_distance_m for patch in self.module_patches], dtype=float)

    @property
    def propulsion_node(self) -> int:
        return int(self.propulsion_patch.node_indices[int(np.argmax(self.propulsion_patch.weights))])

    @property
    def propulsion_distance_m(self) -> float:
        return float(self.propulsion_patch.nearest_distance_m)

    @property
    def wind_node(self) -> int:
        return int(self.wind_patch.node_indices[int(np.argmax(self.wind_patch.weights))])

    @property
    def wind_distance_m(self) -> float:
        return float(self.wind_patch.nearest_distance_m)

    @property
    def lock_nodes(self) -> np.ndarray:
        return np.array([patch.node_indices[int(np.argmax(patch.weights))] for patch in self.lock_patches], dtype=np.int64)

    @property
    def lock_distance_m(self) -> np.ndarray:
        return np.array([patch.nearest_distance_m for patch in self.lock_patches], dtype=float)

    @property
    def maximum_distance_m(self) -> float:
        values = [
            *(patch.nearest_distance_m for patch in self.module_patches),
            self.propulsion_patch.nearest_distance_m,
            self.wind_patch.nearest_distance_m,
            *(patch.nearest_distance_m for patch in self.lock_patches),
            *(patch.nearest_distance_m for patch in self.payload_patches),
            *np.asarray(self.constraint_distance_m, dtype=float).tolist(),
        ]
        return float(max(values)) if values else 0.0


@dataclass(frozen=True)
class StructuralFrameResult:
    time_s: float
    frame_index: int
    fea: FEAResult
    mapping: NodeMapping
    applied_external_force_n: np.ndarray
    inertial_body_force_n: np.ndarray
    pre_fea_force_residual_n: np.ndarray
    pre_fea_moment_residual_nm: np.ndarray
    pre_fea_force_residual_norm_n: float
    pre_fea_moment_residual_norm_nm: float
    wrench_distribution_residual: float
    equilibrium_residual_n: np.ndarray
    equilibrium_residual_norm_n: float

    def to_pyvista(self, mesh: TetraMesh, deformation_scale: float = 1.0):
        grid = mesh.to_pyvista()
        displacement = np.asarray(self.fea.displacement_m, dtype=float)
        grid.point_data["displacement_m"] = displacement
        grid.point_data["displacement_magnitude_m"] = np.linalg.norm(displacement, axis=1)
        grid.point_data["von_mises_pa"] = np.asarray(self.fea.nodal_von_mises_pa, dtype=float)
        grid.cell_data["element_von_mises_pa"] = np.asarray(self.fea.element_von_mises_pa, dtype=float)
        if deformation_scale != 0.0:
            grid.points = np.asarray(grid.points) + float(deformation_scale) * displacement
        return grid


def _preview_radius_for_minimum_boundary_nodes(
    mesh: TetraMesh,
    points_m: np.ndarray,
    *,
    minimum_nodes: int,
    base_radius_m: float,
) -> float:
    """Choose a preview-only patch radius that can carry a 3-D wrench.

    Quantitative FEA never uses this helper: real analyses must provide explicit
    ``.fea.json`` patch radii.  The automatic mapping exists only for software
    tests and visual preview, where very coarse meshes may otherwise collapse a
    load region to one or two nodes and make the six-equation wrench distributor
    mathematically underdetermined.
    """
    boundary = mesh.nodes_m[mesh.boundary_node_indices]
    count = int(minimum_nodes)
    if count < 1 or len(boundary) < count:
        raise ValueError("Preview mesh does not contain enough boundary nodes for the requested load patch.")
    points = np.atleast_2d(np.asarray(points_m, dtype=float))
    required = float(base_radius_m)
    for point in points:
        distances = np.linalg.norm(boundary - point[None, :], axis=1)
        kth = float(np.partition(distances, count - 1)[count - 1])
        required = max(required, np.nextafter(kth, np.inf))
    return required


def auto_structural_load_map(mesh: TetraMesh) -> StructuralLoadMap:
    """Generate deterministic fallback locations for software tests/preview only.

    Preview patch sizes adapt to the boundary-node spacing so moment-carrying
    propulsion and lock regions still contain at least three nodes on deliberately
    coarse test meshes.  This does *not* relax the strict explicit-manifest checks
    used by the real structural-validation workflow.
    """
    lower, upper = mesh.bounds_m
    center = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower)
    modules = np.column_stack(
        (
            center[0] + _MODULE_X_FRACTION * half[0],
            center[1] + _MODULE_Y_FRACTION * half[1],
            np.full(8, lower[2]),
        )
    )
    constraints = np.array(
        [
            [lower[0], lower[1], lower[2]],
            [upper[0], lower[1], lower[2]],
            [lower[0], upper[1], lower[2]],
        ],
        dtype=float,
    )
    propulsion = np.array([lower[0], center[1], lower[2]], dtype=float)
    wind = np.array([center[0], upper[1], upper[2]], dtype=float)
    locks = np.array(
        [[upper[0], lower[1], lower[2]], [upper[0], upper[1], lower[2]]],
        dtype=float,
    )
    characteristic = max(float(np.max(mesh.dimensions_m)), 1e-6)
    radius = 0.08 * characteristic
    propulsion_radius = _preview_radius_for_minimum_boundary_nodes(
        mesh,
        propulsion,
        minimum_nodes=3,
        base_radius_m=1.5 * radius,
    )
    lock_radius = _preview_radius_for_minimum_boundary_nodes(
        mesh,
        locks,
        minimum_nodes=3,
        base_radius_m=radius,
    )
    return StructuralLoadMap(
        modules,
        constraints,
        propulsion,
        wind,
        locks,
        module_patch_radius_m=radius,
        propulsion_patch_radius_m=propulsion_radius,
        wind_patch_radius_m=1.5 * radius,
        lock_patch_radius_m=lock_radius,
        source="auto_bounds_preview_only",
    )


def _nearest_boundary_nodes(mesh: TetraMesh, points_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.atleast_2d(np.asarray(points_m, dtype=float))
    boundary = mesh.boundary_node_indices
    if len(boundary) == 0:
        raise ValueError("The tetrahedral mesh has no boundary nodes.")
    tree = cKDTree(mesh.nodes_m[boundary])
    distance, local_index = tree.query(points, k=1)
    return boundary[np.asarray(local_index, dtype=np.int64)], np.asarray(distance, dtype=float)


def map_structural_locations(
    mesh: TetraMesh,
    load_map: StructuralLoadMap,
    *,
    maximum_snap_distance_m: float | None = None,
) -> NodeMapping:
    tolerance = (
        float(maximum_snap_distance_m)
        if maximum_snap_distance_m is not None
        else 0.08 * float(np.max(mesh.dimensions_m))
    )
    module_patches = tuple(
        boundary_load_patch(mesh, point, load_map.module_patch_radius_m, maximum_snap_distance_m=tolerance)
        for point in load_map.module_points_m
    )
    propulsion_patch = boundary_load_patch(
        mesh,
        load_map.propulsion_point_m,
        load_map.propulsion_patch_radius_m,
        maximum_snap_distance_m=tolerance,
    )
    wind_patch = boundary_load_patch(
        mesh,
        load_map.wind_point_m,
        load_map.wind_patch_radius_m,
        maximum_snap_distance_m=tolerance,
    )
    lock_patches = tuple(
        boundary_load_patch(mesh, point, load_map.lock_patch_radius_m, maximum_snap_distance_m=tolerance)
        for point in load_map.lock_points_m
    )
    payload_patches = tuple(
        boundary_load_patch(mesh, point, load_map.payload_patch_radius_m, maximum_snap_distance_m=tolerance)
        for point in (load_map.payload_support_points_m if load_map.payload_support_points_m is not None else [])
    )
    constraint_nodes, constraint_distance = _nearest_boundary_nodes(mesh, load_map.constraint_points_m)
    return NodeMapping(
        module_patches=module_patches,
        propulsion_patch=propulsion_patch,
        wind_patch=wind_patch,
        lock_patches=lock_patches,
        payload_patches=payload_patches,
        constraint_nodes=np.asarray(constraint_nodes, dtype=np.int64),
        constraint_distance_m=np.asarray(constraint_distance, dtype=float),
    )


def validate_structural_mapping(
    mesh: TetraMesh,
    load_map: StructuralLoadMap,
    *,
    maximum_snap_distance_m: float,
) -> NodeMapping:
    if not np.isfinite(maximum_snap_distance_m) or maximum_snap_distance_m <= 0:
        raise ValueError("maximum_snap_distance_m must be finite and positive.")
    mapping = map_structural_locations(mesh, load_map, maximum_snap_distance_m=maximum_snap_distance_m)
    if mapping.maximum_distance_m > float(maximum_snap_distance_m):
        raise ValueError(
            "At least one structural application point is too far from the volume-mesh boundary: "
            f"maximum {mapping.maximum_distance_m:.6g} m > allowed {maximum_snap_distance_m:.6g} m."
        )
    if len(np.unique(mapping.constraint_nodes)) != 3:
        raise ValueError("Structural constraint points must map to three distinct boundary nodes.")
    if float(np.max(mapping.constraint_distance_m)) > float(maximum_snap_distance_m):
        raise ValueError("At least one kinematic constraint point is too far from the volume-mesh boundary.")
    if len(patch_union(mapping.lock_patches)) < 3:
        raise ValueError("Lock load patches must contain at least three distinct boundary nodes for moment transfer.")
    if len(mapping.propulsion_patch.node_indices) < 3:
        raise ValueError("Propulsion patch must contain at least three boundary nodes for yaw-moment transfer.")
    return mapping


def _row_value(row, name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


def _nodal_resultant(mesh: TetraMesh, nodal_forces_n: np.ndarray, about_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    forces = np.asarray(nodal_forces_n, dtype=float)
    force = np.sum(forces, axis=0)
    r = mesh.nodes_m - np.asarray(about_m, dtype=float)[None, :]
    moment = np.sum(np.cross(r, forces), axis=0)
    return force, moment


class StructuralTwinSolver:
    """Quasi-static node FEA driven by one SC-MEPLS simulation frame.

    Surface point loads are intentionally avoided: module, propulsion, wind, lock,
    and optional payload transfer loads are distributed over validated boundary-node
    patches. World-frame forces/accelerations are rotated into the structural body
    frame before assembly. A 3-2-1 constraint set removes rigid modes; its reactions
    are numerical supports, not physical maglev restraints.
    """

    def __init__(
        self,
        mesh: TetraMesh,
        material: IsotropicMaterial,
        timeline: SimulationTimeline,
        rollout_parameters: RolloutParameters,
        load_map: StructuralLoadMap,
        *,
        maximum_snap_distance_m: float | None = None,
        payload: PayloadMassProperties | None = None,
    ) -> None:
        self.mesh = mesh
        self.material = material
        self.timeline = timeline
        self.rollout_parameters = rollout_parameters
        self.load_map = load_map
        self.payload = payload
        tolerance = (
            float(maximum_snap_distance_m)
            if maximum_snap_distance_m is not None
            else 0.08 * float(np.max(mesh.dimensions_m))
        )
        self.mapping = validate_structural_mapping(mesh, load_map, maximum_snap_distance_m=tolerance)
        fixed_dofs = three_point_kinematic_constraints(mesh, load_map.constraint_points_m)
        self.solver = StaticElasticSolver(mesh, material, fixed_dofs)
        self.mass_properties: StructuralMassProperties = structural_mass_properties(mesh, material)
        if payload is not None and not self.mapping.payload_patches:
            raise ValueError("Payload mass properties require payload_support_points_m in the structural mapping.")

    def _rigid_kinematics(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        history = self.timeline.history
        row = history.iloc[index]
        frame = self.timeline.frame_at_index(index)
        body = frame.rigid_body
        acceleration_world = np.array(
            [_row_value(row, "ax_mps2"), _row_value(row, "ay_mps2"), _row_value(row, "az_mps2")],
            dtype=float,
        )
        acceleration_body = world_vector_to_body(
            acceleration_world, body.roll_rad, body.pitch_rad, body.yaw_rad
        )
        omega = np.array([body.p_radps, body.q_radps, body.r_radps], dtype=float)
        explicit_alpha = np.array(
            [
                _row_value(row, "roll_accel_rad_s2", np.nan),
                _row_value(row, "pitch_accel_rad_s2", np.nan),
                _row_value(row, "yaw_accel_rad_s2", np.nan),
            ],
            dtype=float,
        )
        if np.all(np.isfinite(explicit_alpha)):
            alpha = explicit_alpha
        else:
            time = self.timeline.time_s
            rates = np.column_stack(
                [
                    history.get("roll_rate_rad_s", history.get("p_radps", 0.0)),
                    history.get("pitch_rate_rad_s", history.get("q_radps", 0.0)),
                    history.get("yaw_rate_rad_s", history.get("r_radps", 0.0)),
                ]
            ).astype(float)
            edge_order = 2 if len(history) >= 3 else 1
            alpha = np.asarray(np.gradient(rates, time, axis=0, edge_order=edge_order)[index], dtype=float)
        gravity_body = world_vector_to_body(
            np.array([0.0, 0.0, -9.81]), body.roll_rad, body.pitch_rad, body.yaw_rad
        )
        return acceleration_body, omega, alpha, gravity_body

    def _distributed_dynamic_body_forces(self, index: int) -> np.ndarray:
        acceleration, omega, alpha, gravity = self._rigid_kinematics(index)
        centroid = self.mass_properties.centroid_m
        nodal = np.zeros((self.mesh.node_count, 3), dtype=float)
        density = float(self.material.density_kg_m3)
        for tet, volume, element_centroid in zip(
            self.mesh.tetrahedra,
            self.mesh.element_volumes_m3,
            self.mesh.element_centroids_m,
        ):
            r = element_centroid - centroid
            rigid_acceleration = acceleration + np.cross(alpha, r) + np.cross(omega, np.cross(omega, r))
            equivalent_acceleration = gravity - rigid_acceleration
            share = density * float(volume) * equivalent_acceleration / 4.0
            nodal[tet] += share
        return nodal

    def _payload_transfer_forces(self, index: int) -> tuple[np.ndarray, float]:
        nodal = np.zeros((self.mesh.node_count, 3), dtype=float)
        if self.payload is None:
            return nodal, 0.0
        acceleration, omega, alpha, gravity = self._rigid_kinematics(index)
        centroid = np.asarray(self.payload.centroid_m, dtype=float)
        r = centroid - self.mass_properties.centroid_m
        payload_accel = acceleration + np.cross(alpha, r) + np.cross(omega, np.cross(omega, r))
        force = float(self.payload.mass_kg) * (gravity - payload_accel)
        inertia = np.asarray(self.payload.inertia_centroid_kg_m2, dtype=float)
        moment = -(inertia @ alpha + np.cross(omega, inertia @ omega))
        residual = add_wrench_on_nodes(
            nodal,
            self.mesh,
            patch_union(self.mapping.payload_patches),
            force,
            moment,
            about_point_m=centroid,
        )
        return nodal, residual

    def _external_nodal_forces(self, index: int) -> tuple[np.ndarray, float]:
        row = self.timeline.history.iloc[index]
        frame = self.timeline.frame_at_index(index)
        body = frame.rigid_body
        nodal = np.zeros((self.mesh.node_count, 3), dtype=float)
        wrench_residual = 0.0

        for module, patch in enumerate(self.mapping.module_patches, start=1):
            support = _row_value(row, f"em_force_{module}_n") + _row_value(row, f"pneumatic_force_{module}_n")
            force_body = world_vector_to_body(
                np.array([0.0, 0.0, support]), body.roll_rad, body.pitch_rad, body.yaw_rad
            )
            add_patch_force(nodal, patch, force_body)

        propulsion_world = np.array(
            [
                _row_value(row, "propulsion_force_x_n"),
                _row_value(row, "lateral_control_force_y_n"),
                0.0,
            ],
            dtype=float,
        )
        passive_world = np.array(
            [
                _row_value(row, "passive_force_x_n"),
                _row_value(row, "passive_force_y_n"),
                _row_value(row, "passive_force_z_n"),
            ],
            dtype=float,
        )
        propulsion_body = world_vector_to_body(
            propulsion_world + passive_world, body.roll_rad, body.pitch_rad, body.yaw_rad
        )
        yaw_moment = np.array([0.0, 0.0, _row_value(row, "yaw_control_moment_nm")], dtype=float)
        wrench_residual += add_wrench_on_nodes(
            nodal,
            self.mesh,
            self.mapping.propulsion_patch.node_indices,
            propulsion_body,
            yaw_moment,
            about_point_m=self.mapping.propulsion_patch.center_m,
        )

        wind_world = np.array([0.0, _row_value(row, "wind_force_n"), 0.0], dtype=float)
        wind_body = world_vector_to_body(wind_world, body.roll_rad, body.pitch_rad, body.yaw_rad)
        add_patch_force(nodal, self.mapping.wind_patch, wind_body)

        lock_world = np.array(
            [
                _row_value(row, "lock_force_x_n"),
                _row_value(row, "lock_force_y_n"),
                _row_value(row, "lock_force_z_n"),
            ],
            dtype=float,
        )
        lock_body = world_vector_to_body(lock_world, body.roll_rad, body.pitch_rad, body.yaw_rad)
        lock_moment = np.array(
            [
                _row_value(row, "lock_moment_x_nm"),
                _row_value(row, "lock_moment_y_nm"),
                _row_value(row, "lock_moment_z_nm"),
            ],
            dtype=float,
        )
        # The scenario CG shift is a generalized gravity moment not represented by
        # a fixed homogeneous mesh mass distribution. Transfer it through the same
        # physical support/lock structure instead of letting artificial constraints
        # absorb it silently.
        cg_gravity_moment = np.array(
            [
                _row_value(row, "gravity_moment_x_nm"),
                _row_value(row, "gravity_moment_y_nm"),
                0.0,
            ],
            dtype=float,
        )
        wrench_residual += add_wrench_on_nodes(
            nodal,
            self.mesh,
            patch_union(self.mapping.lock_patches),
            lock_body,
            lock_moment + cg_gravity_moment,
            about_point_m=self.mass_properties.centroid_m,
        )

        payload_nodal, payload_residual = self._payload_transfer_forces(index)
        nodal += payload_nodal
        wrench_residual += payload_residual
        return nodal, wrench_residual

    def solve_frame(self, index: int) -> StructuralFrameResult:
        index = int(index)
        if not 0 <= index < len(self.timeline):
            raise IndexError("Structural frame index is outside the simulation timeline.")
        external, wrench_residual = self._external_nodal_forces(index)
        inertial = self._distributed_dynamic_body_forces(index)
        combined = external + inertial
        pre_force, pre_moment = _nodal_resultant(self.mesh, combined, self.mass_properties.centroid_m)
        fea = self.solver.solve(combined, body_acceleration_mps2=(0.0, 0.0, 0.0))
        reaction_total = np.sum(fea.reaction_n, axis=0)
        total_applied = np.sum(combined, axis=0)
        residual = reaction_total + total_applied
        return StructuralFrameResult(
            time_s=float(self.timeline.time_s[index]),
            frame_index=index,
            fea=fea,
            mapping=self.mapping,
            applied_external_force_n=np.sum(external, axis=0),
            inertial_body_force_n=np.sum(inertial, axis=0),
            pre_fea_force_residual_n=pre_force,
            pre_fea_moment_residual_nm=pre_moment,
            pre_fea_force_residual_norm_n=float(np.linalg.norm(pre_force)),
            pre_fea_moment_residual_norm_nm=float(np.linalg.norm(pre_moment)),
            wrench_distribution_residual=float(wrench_residual),
            equilibrium_residual_n=residual,
            equilibrium_residual_norm_n=float(np.linalg.norm(residual)),
        )

    def solve_time(self, time_s: float) -> StructuralFrameResult:
        return self.solve_frame(self.timeline.nearest_index(time_s))
