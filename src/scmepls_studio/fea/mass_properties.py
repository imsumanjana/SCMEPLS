from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from scmepls_studio.models.rollout_sim import RolloutParameters

from .mesh import TetraMesh
from .solver import IsotropicMaterial


@dataclass(frozen=True)
class StructuralMassProperties:
    volume_m3: float
    mass_kg: float
    centroid_m: np.ndarray
    inertia_centroid_kg_m2: np.ndarray
    bounds_m: np.ndarray
    dimensions_m: np.ndarray


def structural_mass_properties(
    mesh: TetraMesh,
    material: IsotropicMaterial,
) -> StructuralMassProperties:
    """Integrate mass, centroid, and full inertia tensor from tetrahedral volume elements."""
    density = float(material.density_kg_m3)
    total_volume = 0.0
    first_moment = np.zeros(3, dtype=float)
    second_moment = np.zeros((3, 3), dtype=float)

    for tet, volume in zip(mesh.tetrahedra, mesh.element_volumes_m3):
        vertices = mesh.nodes_m[tet]
        vertex_sum = np.sum(vertices, axis=0)
        sum_outer = sum(np.outer(v, v) for v in vertices)
        q = float(volume) / 20.0 * (sum_outer + np.outer(vertex_sum, vertex_sum))
        total_volume += float(volume)
        first_moment += float(volume) * vertex_sum / 4.0
        second_moment += q

    if total_volume <= 0.0:
        raise ValueError("Structural volume must be positive.")
    mass = density * total_volume
    centroid = first_moment / total_volume
    q_mass = density * second_moment
    inertia_origin = np.trace(q_mass) * np.eye(3) - q_mass
    shift = mass * (
        float(np.dot(centroid, centroid)) * np.eye(3) - np.outer(centroid, centroid)
    )
    inertia_centroid = 0.5 * ((inertia_origin - shift) + (inertia_origin - shift).T)
    eig = np.linalg.eigvalsh(inertia_centroid)
    if np.any(~np.isfinite(eig)) or np.any(eig <= 0.0):
        raise ValueError("Integrated structural inertia tensor is not positive definite.")

    bounds = mesh.bounds_m
    return StructuralMassProperties(
        volume_m3=float(total_volume),
        mass_kg=float(mass),
        centroid_m=np.asarray(centroid, dtype=float),
        inertia_centroid_kg_m2=np.asarray(inertia_centroid, dtype=float),
        bounds_m=np.asarray(bounds, dtype=float),
        dimensions_m=np.asarray(bounds[1] - bounds[0], dtype=float),
    )


def rollout_parameters_from_structure(
    base: RolloutParameters,
    properties: StructuralMassProperties,
    *,
    mass_scope: str = "moving_assembly",
) -> RolloutParameters:
    """Return rollout parameters driven by integrated imported-mesh properties.

    ``moving_assembly`` means the volume mesh represents the complete moving body,
    including payload. ``platform_only`` means the mesh represents only the platform;
    platform mass and dimensions are updated, while the pre-existing total inertia is
    retained because payload inertia is not identifiable from platform geometry alone.
    """
    scope = str(mass_scope).strip().lower()
    dims = properties.dimensions_m
    if np.any(dims <= 0.0):
        raise ValueError("Geometry dimensions must be positive.")

    common = dict(
        body_length_m=float(dims[0]),
        body_width_m=float(dims[1]),
        body_height_m=float(dims[2]),
    )
    if scope == "platform_only":
        return replace(base, platform_mass_kg=float(properties.mass_kg), **common)
    if scope != "moving_assembly":
        raise ValueError("mass_scope must be 'moving_assembly' or 'platform_only'.")
    platform_mass = float(properties.mass_kg) - float(base.payload_mass_kg)
    if platform_mass <= 0.0:
        raise ValueError(
            "For moving_assembly coupling, integrated mesh mass must exceed the configured payload mass."
        )
    inertia = properties.inertia_centroid_kg_m2
    return replace(
        base,
        platform_mass_kg=platform_mass,
        inertia_xx_kg_m2=float(inertia[0, 0]),
        inertia_yy_kg_m2=float(inertia[1, 1]),
        inertia_zz_kg_m2=float(inertia[2, 2]),
        inertia_xy_kg_m2=float(inertia[0, 1]),
        inertia_xz_kg_m2=float(inertia[0, 2]),
        inertia_yz_kg_m2=float(inertia[1, 2]),
        **common,
    )
