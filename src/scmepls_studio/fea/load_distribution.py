from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from .mesh import TetraMesh


@dataclass(frozen=True)
class LoadPatch:
    """A normalized set of boundary nodes used to apply a physical load patch."""

    center_m: np.ndarray
    node_indices: np.ndarray
    weights: np.ndarray
    nearest_distance_m: float

    def __post_init__(self) -> None:
        center = np.asarray(self.center_m, dtype=float)
        nodes = np.asarray(self.node_indices, dtype=np.int64)
        weights = np.asarray(self.weights, dtype=float)
        if center.shape != (3,) or np.any(~np.isfinite(center)):
            raise ValueError("Load-patch center must be a finite XYZ vector.")
        if nodes.ndim != 1 or len(nodes) < 1:
            raise ValueError("A load patch requires at least one node.")
        if weights.shape != nodes.shape or np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("Load-patch weights must be finite, non-negative, and match node indices.")
        total = float(np.sum(weights))
        if total <= 0.0:
            raise ValueError("Load-patch weights must have a positive sum.")
        object.__setattr__(self, "center_m", center)
        object.__setattr__(self, "node_indices", nodes)
        object.__setattr__(self, "weights", weights / total)


def boundary_load_patch(
    mesh: TetraMesh,
    center_m: np.ndarray,
    radius_m: float,
    *,
    maximum_snap_distance_m: float,
) -> LoadPatch:
    """Map a physical application region to nearby boundary nodes.

    Nodes inside ``radius_m`` receive a smooth compact weight. If no boundary node
    lies inside the radius, the nearest boundary node is used only when it remains
    inside the validated snap tolerance.
    """
    center = np.asarray(center_m, dtype=float)
    if center.shape != (3,) or np.any(~np.isfinite(center)):
        raise ValueError("Load-patch center must be a finite XYZ vector.")
    if not np.isfinite(radius_m) or radius_m <= 0.0:
        raise ValueError("Load-patch radius must be finite and positive.")
    if not np.isfinite(maximum_snap_distance_m) or maximum_snap_distance_m <= 0.0:
        raise ValueError("maximum_snap_distance_m must be finite and positive.")

    boundary = mesh.boundary_node_indices
    if len(boundary) == 0:
        raise ValueError("The tetrahedral mesh has no boundary nodes.")
    points = mesh.nodes_m[boundary]
    distance = np.linalg.norm(points - center[None, :], axis=1)
    nearest = float(np.min(distance))
    if nearest > maximum_snap_distance_m:
        raise ValueError(
            f"Load application point is {nearest:.6g} m from the volume-mesh boundary, "
            f"exceeding the allowed {maximum_snap_distance_m:.6g} m."
        )
    inside = distance <= float(radius_m)
    if not np.any(inside):
        local = int(np.argmin(distance))
        nodes = np.array([boundary[local]], dtype=np.int64)
        weights = np.array([1.0], dtype=float)
    else:
        nodes = boundary[inside]
        d = distance[inside]
        # Smooth compact kernel: center nodes receive more load without a singular
        # all-force-at-one-node spike. A small floor prevents zero total weight.
        q = np.clip(d / float(radius_m), 0.0, 1.0)
        weights = np.maximum((1.0 - q * q) ** 2, 1e-12)
    return LoadPatch(center, nodes, weights, nearest)


def add_patch_force(nodal_forces_n: np.ndarray, patch: LoadPatch, force_n: np.ndarray) -> None:
    force = np.asarray(force_n, dtype=float)
    if force.shape != (3,) or np.any(~np.isfinite(force)):
        raise ValueError("Patch force must be a finite XYZ vector.")
    nodal_forces_n[patch.node_indices] += patch.weights[:, None] * force[None, :]


def add_wrench_on_nodes(
    nodal_forces_n: np.ndarray,
    mesh: TetraMesh,
    node_indices: np.ndarray,
    force_n: np.ndarray,
    moment_nm: np.ndarray,
    *,
    about_point_m: np.ndarray,
) -> float:
    """Distribute a force/moment wrench with a minimum-norm nodal solution.

    Returns the Euclidean residual of the six equilibrium equations. This is useful
    as a pre-FEA audit metric and is independent of the subsequent stiffness solve.
    """
    nodes = np.unique(np.asarray(node_indices, dtype=np.int64))
    force = np.asarray(force_n, dtype=float)
    moment = np.asarray(moment_nm, dtype=float)
    origin = np.asarray(about_point_m, dtype=float)
    if len(nodes) < 3:
        raise ValueError("At least three distinct nodes are required to distribute a wrench.")
    if force.shape != (3,) or moment.shape != (3,) or origin.shape != (3,):
        raise ValueError("Force, moment, and about_point_m must be XYZ vectors.")
    if np.any(~np.isfinite(force)) or np.any(~np.isfinite(moment)) or np.any(~np.isfinite(origin)):
        raise ValueError("Wrench inputs must be finite.")

    a = np.zeros((6, 3 * len(nodes)), dtype=float)
    for i, node in enumerate(nodes):
        j = 3 * i
        a[:3, j : j + 3] = np.eye(3)
        rx, ry, rz = mesh.nodes_m[int(node)] - origin
        a[3:, j : j + 3] = np.array(
            [[0.0, -rz, ry], [rz, 0.0, -rx], [-ry, rx, 0.0]],
            dtype=float,
        )
    target = np.concatenate((force, moment))
    solution, *_ = np.linalg.lstsq(a, target, rcond=None)
    achieved = a @ solution
    for i, node in enumerate(nodes):
        nodal_forces_n[int(node)] += solution[3 * i : 3 * i + 3]
    return float(np.linalg.norm(achieved - target))


def patch_union(patches: tuple[LoadPatch, ...]) -> np.ndarray:
    if not patches:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate([patch.node_indices for patch in patches]))
