from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from scmepls_studio.digital_twin.physics import SimulationTimeline
from scmepls_studio.models.rollout_sim import RolloutParameters

from .mass_properties import StructuralMassProperties, structural_mass_properties
from .mesh import TetraMesh
from .solver import FEAResult, IsotropicMaterial, StaticElasticSolver, three_point_kinematic_constraints

_MODULE_X_FRACTION = np.array([-1.0, 0.0, 1.0, -1.0, 1.0, -1.0, 0.0, 1.0])
_MODULE_Y_FRACTION = np.array([1.0, 1.0, 1.0, 0.0, 0.0, -1.0, -1.0, -1.0])


@dataclass(frozen=True)
class StructuralLoadMap:
    """Physical load/constraint locations in the imported geometry frame."""

    module_points_m: np.ndarray
    constraint_points_m: np.ndarray
    propulsion_point_m: np.ndarray
    wind_point_m: np.ndarray
    lock_points_m: np.ndarray
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


@dataclass(frozen=True)
class NodeMapping:
    module_nodes: np.ndarray
    module_distance_m: np.ndarray
    propulsion_node: int
    propulsion_distance_m: float
    wind_node: int
    wind_distance_m: float
    lock_nodes: np.ndarray
    lock_distance_m: np.ndarray

    @property
    def maximum_distance_m(self) -> float:
        values = np.concatenate(
            (
                np.asarray(self.module_distance_m, dtype=float).ravel(),
                np.array([self.propulsion_distance_m, self.wind_distance_m]),
                np.asarray(self.lock_distance_m, dtype=float).ravel(),
            )
        )
        return float(np.max(values)) if len(values) else 0.0


@dataclass(frozen=True)
class StructuralFrameResult:
    time_s: float
    frame_index: int
    fea: FEAResult
    mapping: NodeMapping
    applied_external_force_n: np.ndarray
    inertial_body_force_n: np.ndarray
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


def auto_structural_load_map(mesh: TetraMesh) -> StructuralLoadMap:
    """Generate deterministic fallback locations from mesh bounds.

    This is suitable for software tests and first visualization only. It is not a
    substitute for an explicit model-specific structural mapping during validation.
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
        [
            [upper[0], lower[1], lower[2]],
            [upper[0], upper[1], lower[2]],
        ],
        dtype=float,
    )
    return StructuralLoadMap(modules, constraints, propulsion, wind, locks, source="auto_bounds")


def _nearest_boundary_nodes(mesh: TetraMesh, points_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = np.atleast_2d(np.asarray(points_m, dtype=float))
    boundary = mesh.boundary_node_indices
    if len(boundary) == 0:
        raise ValueError("The tetrahedral mesh has no boundary nodes.")
    tree = cKDTree(mesh.nodes_m[boundary])
    distance, local_index = tree.query(points, k=1)
    return boundary[np.asarray(local_index, dtype=np.int64)], np.asarray(distance, dtype=float)


def map_structural_locations(mesh: TetraMesh, load_map: StructuralLoadMap) -> NodeMapping:
    module_nodes, module_distance = _nearest_boundary_nodes(mesh, load_map.module_points_m)
    propulsion_nodes, propulsion_distance = _nearest_boundary_nodes(mesh, load_map.propulsion_point_m)
    wind_nodes, wind_distance = _nearest_boundary_nodes(mesh, load_map.wind_point_m)
    lock_nodes, lock_distance = _nearest_boundary_nodes(mesh, load_map.lock_points_m)
    return NodeMapping(
        module_nodes=np.asarray(module_nodes, dtype=np.int64),
        module_distance_m=np.asarray(module_distance, dtype=float),
        propulsion_node=int(propulsion_nodes[0]),
        propulsion_distance_m=float(propulsion_distance[0]),
        wind_node=int(wind_nodes[0]),
        wind_distance_m=float(wind_distance[0]),
        lock_nodes=np.asarray(lock_nodes, dtype=np.int64),
        lock_distance_m=np.asarray(lock_distance, dtype=float),
    )


def validate_structural_mapping(
    mesh: TetraMesh,
    load_map: StructuralLoadMap,
    *,
    maximum_snap_distance_m: float,
) -> NodeMapping:
    if not np.isfinite(maximum_snap_distance_m) or maximum_snap_distance_m <= 0:
        raise ValueError("maximum_snap_distance_m must be finite and positive.")
    mapping = map_structural_locations(mesh, load_map)
    if mapping.maximum_distance_m > float(maximum_snap_distance_m):
        raise ValueError(
            "At least one structural application point is too far from the volume-mesh boundary: "
            f"maximum {mapping.maximum_distance_m:.6g} m > allowed {maximum_snap_distance_m:.6g} m."
        )
    constraint_nodes, constraint_distance = _nearest_boundary_nodes(mesh, load_map.constraint_points_m)
    if len(np.unique(constraint_nodes)) != 3:
        raise ValueError("Structural constraint points must map to three distinct boundary nodes.")
    if float(np.max(constraint_distance)) > float(maximum_snap_distance_m):
        raise ValueError("At least one kinematic constraint point is too far from the volume-mesh boundary.")
    return mapping


def _row_value(row, name: str, default: float = 0.0) -> float:
    value = row.get(name, default)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)


class StructuralTwinSolver:
    """Quasi-static node-by-node FEA driven by an SC-MEPLS simulation frame.

    The solver applies the eight module support forces, wind, propulsion/lateral
    resultant, and lock resultant to mapped boundary nodes. Gravity plus rigid-body
    translational/rotational inertia are assembled as distributed tetrahedral body
    forces. A minimal 3-2-1 constraint removes rigid modes without treating the
    maglev supports as fixed structural restraints.
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
    ) -> None:
        self.mesh = mesh
        self.material = material
        self.timeline = timeline
        self.rollout_parameters = rollout_parameters
        self.load_map = load_map
        tolerance = (
            float(maximum_snap_distance_m)
            if maximum_snap_distance_m is not None
            else 0.08 * float(np.max(mesh.dimensions_m))
        )
        self.mapping = validate_structural_mapping(mesh, load_map, maximum_snap_distance_m=tolerance)
        fixed_dofs = three_point_kinematic_constraints(mesh, load_map.constraint_points_m)
        self.solver = StaticElasticSolver(mesh, material, fixed_dofs)
        self.mass_properties: StructuralMassProperties = structural_mass_properties(mesh, material)

    def _rigid_kinematics(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        history = self.timeline.history
        row = history.iloc[index]
        acceleration = np.array(
            [_row_value(row, "ax_mps2"), _row_value(row, "ay_mps2"), _row_value(row, "az_mps2")],
            dtype=float,
        )
        omega = np.array(
            [
                _row_value(row, "roll_rate_rad_s"),
                _row_value(row, "pitch_rate_rad_s"),
                _row_value(row, "yaw_rate_rad_s"),
            ],
            dtype=float,
        )
        time = self.timeline.time_s
        rates = np.column_stack(
            [
                history.get("roll_rate_rad_s", 0.0),
                history.get("pitch_rate_rad_s", 0.0),
                history.get("yaw_rate_rad_s", 0.0),
            ]
        ).astype(float)
        if len(history) >= 3:
            angular_acceleration_all = np.gradient(rates, time, axis=0, edge_order=2)
        else:
            angular_acceleration_all = np.gradient(rates, time, axis=0, edge_order=1)
        alpha = np.asarray(angular_acceleration_all[index], dtype=float)
        return acceleration, omega, alpha

    def _distributed_dynamic_body_forces(self, index: int) -> np.ndarray:
        acceleration, omega, alpha = self._rigid_kinematics(index)
        gravity = np.array([0.0, 0.0, -9.81], dtype=float)
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

    def _external_nodal_forces(self, index: int) -> np.ndarray:
        row = self.timeline.history.iloc[index]
        nodal = np.zeros((self.mesh.node_count, 3), dtype=float)
        for module, node in enumerate(self.mapping.module_nodes, start=1):
            support = _row_value(row, f"em_force_{module}_n") + _row_value(row, f"pneumatic_force_{module}_n")
            nodal[int(node), 2] += support

        mass = float(self.rollout_parameters.platform_mass_kg + self.rollout_parameters.payload_mass_kg)
        lock = _row_value(row, "lock_fraction")
        x = _row_value(row, "x_m")
        vx = _row_value(row, "vx_mps")
        y = _row_value(row, "y_m")
        vy = _row_value(row, "vy_mps")
        z = _row_value(row, "z_m")
        vz = _row_value(row, "vz_mps")
        gap_ref = _row_value(row, "gap_ref_m", self.rollout_parameters.target_gap_m)
        fx_lock = lock * (4000.0 * (self.rollout_parameters.track_length_m - x) - 600.0 * vx)
        fy_lock = lock * (-3500.0 * y - 500.0 * vy)
        fz_lock = lock * (9000.0 * (gap_ref - z) - 700.0 * vz)

        ax = _row_value(row, "ax_mps2")
        ay = _row_value(row, "ay_mps2")
        wind = _row_value(row, "wind_force_n")
        fx_propulsion = mass * ax - fx_lock + 16.0 * vx
        fy_control = mass * ay - fy_lock - wind + 24.0 * vy
        nodal[self.mapping.propulsion_node] += np.array([fx_propulsion, fy_control, 0.0])
        nodal[self.mapping.wind_node, 1] += wind
        lock_share = np.array([fx_lock, fy_lock, fz_lock], dtype=float) / len(self.mapping.lock_nodes)
        for node in self.mapping.lock_nodes:
            nodal[int(node)] += lock_share
        return nodal

    def solve_frame(self, index: int) -> StructuralFrameResult:
        index = int(index)
        if not 0 <= index < len(self.timeline):
            raise IndexError("Structural frame index is outside the simulation timeline.")
        external = self._external_nodal_forces(index)
        inertial = self._distributed_dynamic_body_forces(index)
        combined = external + inertial
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
            equilibrium_residual_n=residual,
            equilibrium_residual_norm_n=float(np.linalg.norm(residual)),
        )

    def solve_time(self, time_s: float) -> StructuralFrameResult:
        return self.solve_frame(self.timeline.nearest_index(time_s))
