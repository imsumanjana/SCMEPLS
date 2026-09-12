from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


class TetraMeshError(ValueError):
    """Raised when a tetrahedral finite-element mesh is invalid."""


@dataclass(frozen=True)
class TetraMesh:
    """First-order tetrahedral mesh in SI units.

    ``nodes_m`` stores nodal coordinates in the structural body frame and
    ``tetrahedra`` stores zero-based 4-node connectivity. Connectivity orientation
    is normalized on construction so a negative signed volume is not incorrectly
    reported later as a physically inverted element.
    """

    nodes_m: np.ndarray
    tetrahedra: np.ndarray
    component_id: str = "STRUCTURE"

    def __post_init__(self) -> None:
        nodes = np.asarray(self.nodes_m, dtype=float)
        tets = np.asarray(self.tetrahedra, dtype=np.int64).copy()
        if nodes.ndim != 2 or nodes.shape[1] != 3 or len(nodes) < 4:
            raise TetraMeshError("nodes_m must have shape (N, 3) with at least four nodes.")
        if tets.ndim != 2 or tets.shape[1] != 4 or len(tets) < 1:
            raise TetraMeshError("tetrahedra must have shape (M, 4) with at least one element.")
        if not np.all(np.isfinite(nodes)):
            raise TetraMeshError("Mesh nodes contain non-finite coordinates.")
        if np.any(tets < 0) or np.any(tets >= len(nodes)):
            raise TetraMeshError("Tetrahedral connectivity contains out-of-range node indices.")
        if np.any(np.apply_along_axis(lambda row: len(set(row.tolist())) < 4, 1, tets)):
            raise TetraMeshError("At least one tetrahedron repeats a node index.")

        # Normalize element orientation. A negative determinant can be caused solely
        # by node ordering and is not, by itself, evidence of an inverted physical
        # element. Swapping two local nodes makes the reference Jacobian positive.
        p = nodes[tets]
        signed = np.einsum(
            "ij,ij->i",
            np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
            p[:, 3] - p[:, 0],
        ) / 6.0
        if np.any(~np.isfinite(signed)) or np.any(np.abs(signed) <= 1e-15):
            raise TetraMeshError("Mesh contains zero-volume or invalid tetrahedra.")
        negative = signed < 0.0
        if np.any(negative):
            tmp = tets[negative, 1].copy()
            tets[negative, 1] = tets[negative, 2]
            tets[negative, 2] = tmp

        object.__setattr__(self, "nodes_m", np.ascontiguousarray(nodes))
        object.__setattr__(self, "tetrahedra", np.ascontiguousarray(tets))
        volumes = self.element_volumes_m3
        if np.any(~np.isfinite(volumes)) or np.any(volumes <= 1e-15):
            raise TetraMeshError("Mesh contains zero-volume or invalid tetrahedra.")

    @property
    def node_count(self) -> int:
        return int(self.nodes_m.shape[0])

    @property
    def element_count(self) -> int:
        return int(self.tetrahedra.shape[0])

    @property
    def bounds_m(self) -> np.ndarray:
        return np.vstack((np.min(self.nodes_m, axis=0), np.max(self.nodes_m, axis=0)))

    @property
    def dimensions_m(self) -> np.ndarray:
        bounds = self.bounds_m
        return bounds[1] - bounds[0]

    @property
    def signed_element_volumes_m3(self) -> np.ndarray:
        p = self.nodes_m[self.tetrahedra]
        a = p[:, 1] - p[:, 0]
        b = p[:, 2] - p[:, 0]
        c = p[:, 3] - p[:, 0]
        return np.einsum("ij,ij->i", np.cross(a, b), c) / 6.0

    @property
    def element_volumes_m3(self) -> np.ndarray:
        return np.abs(self.signed_element_volumes_m3)

    @property
    def element_centroids_m(self) -> np.ndarray:
        return np.mean(self.nodes_m[self.tetrahedra], axis=1)

    @property
    def total_volume_m3(self) -> float:
        return float(np.sum(self.element_volumes_m3))

    @property
    def boundary_faces(self) -> np.ndarray:
        """Return triangular faces that occur on exactly one tetrahedron."""
        t = self.tetrahedra
        faces = np.vstack(
            (
                t[:, [0, 2, 1]],
                t[:, [0, 1, 3]],
                t[:, [1, 2, 3]],
                t[:, [2, 0, 3]],
            )
        )
        keys = np.sort(faces, axis=1)
        _, first, counts = np.unique(keys, axis=0, return_index=True, return_counts=True)
        return np.ascontiguousarray(faces[first[counts == 1]])

    @property
    def boundary_node_indices(self) -> np.ndarray:
        return np.unique(self.boundary_faces.ravel())

    def nearest_nodes(
        self,
        points_m: np.ndarray,
        *,
        max_distance_m: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        points = np.atleast_2d(np.asarray(points_m, dtype=float))
        if points.shape[1] != 3 or not np.all(np.isfinite(points)):
            raise TetraMeshError("Query points must be finite XYZ coordinates.")
        tree = cKDTree(self.nodes_m)
        distance, index = tree.query(points, k=1)
        if max_distance_m is not None and np.any(distance > float(max_distance_m)):
            raise TetraMeshError(
                f"No mesh node is within {float(max_distance_m):g} m of at least one requested point."
            )
        return np.asarray(index, dtype=np.int64), np.asarray(distance, dtype=float)

    def to_pyvista(self):
        """Convert the volume mesh to a PyVista UnstructuredGrid."""
        try:
            import pyvista as pv
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("PyVista is required to visualize/export a tetrahedral mesh.") from exc
        cells = np.column_stack(
            (np.full(self.element_count, 4, dtype=np.int64), self.tetrahedra)
        ).ravel()
        cell_types = np.full(self.element_count, pv.CellType.TETRA, dtype=np.uint8)
        return pv.UnstructuredGrid(cells, cell_types, self.nodes_m)

    def save_vtu(self, path: str) -> None:
        self.to_pyvista().save(path)
