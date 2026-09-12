from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import splu

from .mesh import TetraMesh


@dataclass(frozen=True)
class IsotropicMaterial:
    youngs_modulus_pa: float = 69.0e9
    poisson_ratio: float = 0.33
    density_kg_m3: float = 2700.0
    yield_strength_pa: float = 240.0e6

    def __post_init__(self) -> None:
        if not np.isfinite(self.youngs_modulus_pa) or self.youngs_modulus_pa <= 0:
            raise ValueError("Young's modulus must be finite and positive.")
        if not np.isfinite(self.poisson_ratio) or not (-0.99 < self.poisson_ratio < 0.499):
            raise ValueError("Poisson ratio must lie between -0.99 and 0.499.")
        if not np.isfinite(self.density_kg_m3) or self.density_kg_m3 <= 0:
            raise ValueError("Density must be finite and positive.")
        if not np.isfinite(self.yield_strength_pa) or self.yield_strength_pa <= 0:
            raise ValueError("Yield strength must be finite and positive.")


@dataclass(frozen=True)
class FEAResult:
    displacement_m: np.ndarray
    nodal_force_n: np.ndarray
    reaction_n: np.ndarray
    element_strain: np.ndarray
    element_stress_pa: np.ndarray
    element_von_mises_pa: np.ndarray
    nodal_von_mises_pa: np.ndarray
    max_displacement_m: float
    max_von_mises_pa: float
    safety_factor_min: float


def elasticity_matrix(material: IsotropicMaterial) -> np.ndarray:
    e = float(material.youngs_modulus_pa)
    nu = float(material.poisson_ratio)
    scale = e / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return scale * np.array(
        [
            [1 - nu, nu, nu, 0, 0, 0],
            [nu, 1 - nu, nu, 0, 0, 0],
            [nu, nu, 1 - nu, 0, 0, 0],
            [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
        ],
        dtype=float,
    )


def tetra_b_matrix(points_m: np.ndarray) -> tuple[np.ndarray, float]:
    points = np.asarray(points_m, dtype=float)
    if points.shape != (4, 3):
        raise ValueError("A first-order tetrahedron requires four XYZ nodes.")
    interpolation = np.column_stack((np.ones(4), points))
    determinant = float(np.linalg.det(interpolation))
    volume = abs(determinant) / 6.0
    if not np.isfinite(volume) or volume <= 1e-15:
        raise ValueError("Degenerate tetrahedron cannot be used for FEA.")
    inv = np.linalg.inv(interpolation)
    gradients = inv[1:, :].T
    b = np.zeros((6, 12), dtype=float)
    for i, (dx, dy, dz) in enumerate(gradients):
        j = 3 * i
        b[0, j] = dx
        b[1, j + 1] = dy
        b[2, j + 2] = dz
        b[3, j] = dy
        b[3, j + 1] = dx
        b[4, j + 1] = dz
        b[4, j + 2] = dy
        b[5, j] = dz
        b[5, j + 2] = dx
    return b, volume


def von_mises_from_stress(stress_pa: np.ndarray) -> np.ndarray:
    s = np.asarray(stress_pa, dtype=float)
    sx, sy, sz, txy, tyz, tzx = [s[..., i] for i in range(6)]
    return np.sqrt(
        0.5 * ((sx - sy) ** 2 + (sy - sz) ** 2 + (sz - sx) ** 2)
        + 3.0 * (txy**2 + tyz**2 + tzx**2)
    )


def constrained_dofs(node_components: dict[int, tuple[bool, bool, bool]], node_count: int) -> np.ndarray:
    dofs: list[int] = []
    for node, components in node_components.items():
        if not 0 <= int(node) < int(node_count):
            raise ValueError(f"Constraint node {node} is outside the mesh.")
        if len(components) != 3:
            raise ValueError("Constraint components must contain XYZ booleans.")
        for axis, fixed in enumerate(components):
            if fixed:
                dofs.append(3 * int(node) + axis)
    return np.unique(np.asarray(dofs, dtype=np.int64))


def three_point_kinematic_constraints(mesh: TetraMesh, anchor_points_m: np.ndarray) -> np.ndarray:
    """Create a 3-2-1 constraint set that removes six rigid-body modes."""
    points = np.asarray(anchor_points_m, dtype=float)
    if points.shape != (3, 3):
        raise ValueError("Three XYZ anchor points are required for 3-2-1 constraints.")
    nodes, _ = mesh.nearest_nodes(points)
    if len(np.unique(nodes)) != 3:
        raise ValueError("The three kinematic anchor points must map to three different mesh nodes.")
    a, b, c = [int(v) for v in nodes]
    return constrained_dofs(
        {
            a: (True, True, True),
            b: (False, True, True),
            c: (False, False, True),
        },
        mesh.node_count,
    )


class StaticElasticSolver:
    """Small-strain linear-elastic tetrahedral FEA with reusable stiffness factorization."""

    def __init__(
        self,
        mesh: TetraMesh,
        material: IsotropicMaterial,
        fixed_dofs: np.ndarray,
    ) -> None:
        self.mesh = mesh
        self.material = material
        self.fixed_dofs = np.unique(np.asarray(fixed_dofs, dtype=np.int64))
        self.ndof = 3 * mesh.node_count
        if np.any(self.fixed_dofs < 0) or np.any(self.fixed_dofs >= self.ndof):
            raise ValueError("At least one constrained degree of freedom is outside the mesh.")
        if len(self.fixed_dofs) < 6:
            raise ValueError("At least six independent constraints are required to suppress rigid-body modes.")
        self._d = elasticity_matrix(material)
        self._b: list[np.ndarray] = []
        self._volumes: list[float] = []
        self._element_dofs: list[np.ndarray] = []
        self.stiffness = self._assemble_stiffness()
        mask = np.ones(self.ndof, dtype=bool)
        mask[self.fixed_dofs] = False
        self.free_dofs = np.flatnonzero(mask)
        if len(self.free_dofs) == 0:
            raise ValueError("All structural degrees of freedom are constrained.")
        kff = csc_matrix(self.stiffness[self.free_dofs][:, self.free_dofs])
        try:
            self._factor = splu(kff)
        except RuntimeError as exc:
            raise ValueError(
                "Structural stiffness is singular. Check mesh connectivity and boundary constraints."
            ) from exc

    def _assemble_stiffness(self):
        rows: list[int] = []
        cols: list[int] = []
        values: list[float] = []
        for tet in self.mesh.tetrahedra:
            points = self.mesh.nodes_m[tet]
            b, volume = tetra_b_matrix(points)
            ke = b.T @ self._d @ b * volume
            dofs = np.array([[3 * n, 3 * n + 1, 3 * n + 2] for n in tet], dtype=np.int64).ravel()
            rr, cc = np.meshgrid(dofs, dofs, indexing="ij")
            rows.extend(rr.ravel().tolist())
            cols.extend(cc.ravel().tolist())
            values.extend(ke.ravel().tolist())
            self._b.append(b)
            self._volumes.append(volume)
            self._element_dofs.append(dofs)
        return coo_matrix((values, (rows, cols)), shape=(self.ndof, self.ndof)).tocsr()

    def body_force_vector(self, acceleration_mps2: tuple[float, float, float]) -> np.ndarray:
        acceleration = np.asarray(acceleration_mps2, dtype=float)
        if acceleration.shape != (3,) or not np.all(np.isfinite(acceleration)):
            raise ValueError("Body acceleration must be a finite XYZ vector.")
        force = np.zeros((self.mesh.node_count, 3), dtype=float)
        density = float(self.material.density_kg_m3)
        for tet, volume in zip(self.mesh.tetrahedra, self._volumes):
            share = density * volume * acceleration / 4.0
            force[tet] += share
        return force

    def solve(
        self,
        nodal_forces_n: np.ndarray | None = None,
        *,
        body_acceleration_mps2: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ) -> FEAResult:
        if nodal_forces_n is None:
            nodal = np.zeros((self.mesh.node_count, 3), dtype=float)
        else:
            nodal = np.asarray(nodal_forces_n, dtype=float)
            if nodal.shape != (self.mesh.node_count, 3) or not np.all(np.isfinite(nodal)):
                raise ValueError("nodal_forces_n must have shape (node_count, 3) and finite values.")
            nodal = nodal.copy()
        nodal += self.body_force_vector(body_acceleration_mps2)
        force = nodal.ravel()

        displacement = np.zeros(self.ndof, dtype=float)
        displacement[self.free_dofs] = self._factor.solve(force[self.free_dofs])
        reaction = self.stiffness @ displacement - force

        strain = np.zeros((self.mesh.element_count, 6), dtype=float)
        stress = np.zeros_like(strain)
        for e, (b, dofs) in enumerate(zip(self._b, self._element_dofs)):
            strain[e] = b @ displacement[dofs]
            stress[e] = self._d @ strain[e]
        element_vm = von_mises_from_stress(stress)

        nodal_vm_sum = np.zeros(self.mesh.node_count, dtype=float)
        nodal_weight = np.zeros(self.mesh.node_count, dtype=float)
        for tet, volume, vm in zip(self.mesh.tetrahedra, self._volumes, element_vm):
            nodal_vm_sum[tet] += float(vm) * volume
            nodal_weight[tet] += volume
        nodal_vm = np.divide(
            nodal_vm_sum,
            nodal_weight,
            out=np.zeros_like(nodal_vm_sum),
            where=nodal_weight > 0,
        )

        displacement_xyz = displacement.reshape(-1, 3)
        reaction_xyz = reaction.reshape(-1, 3)
        magnitude = np.linalg.norm(displacement_xyz, axis=1)
        max_vm = float(np.max(element_vm)) if len(element_vm) else 0.0
        safety = np.inf if max_vm <= 0.0 else float(self.material.yield_strength_pa / max_vm)
        return FEAResult(
            displacement_m=displacement_xyz,
            nodal_force_n=nodal,
            reaction_n=reaction_xyz,
            element_strain=strain,
            element_stress_pa=stress,
            element_von_mises_pa=element_vm,
            nodal_von_mises_pa=nodal_vm,
            max_displacement_m=float(np.max(magnitude)),
            max_von_mises_pa=max_vm,
            safety_factor_min=safety,
        )
