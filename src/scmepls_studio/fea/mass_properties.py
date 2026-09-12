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


@dataclass(frozen=True)
class PayloadMassProperties:
    """Payload mass properties in the same body coordinate frame as the platform mesh."""

    mass_kg: float
    centroid_m: np.ndarray
    inertia_centroid_kg_m2: np.ndarray

    def __post_init__(self) -> None:
        mass = float(self.mass_kg)
        centroid = np.asarray(self.centroid_m, dtype=float)
        inertia = np.asarray(self.inertia_centroid_kg_m2, dtype=float)
        if not np.isfinite(mass) or mass < 0.0:
            raise ValueError("Payload mass must be finite and non-negative.")
        if centroid.shape != (3,) or np.any(~np.isfinite(centroid)):
            raise ValueError("Payload centroid must be a finite XYZ vector.")
        if inertia.shape != (3, 3) or np.any(~np.isfinite(inertia)):
            raise ValueError("Payload inertia must be a finite 3x3 tensor.")
        if mass > 0.0 and np.any(np.linalg.eigvalsh(0.5 * (inertia + inertia.T)) <= 0.0):
            raise ValueError("Payload inertia tensor must be positive definite for non-zero payload mass.")
        object.__setattr__(self, "centroid_m", centroid)
        object.__setattr__(self, "inertia_centroid_kg_m2", 0.5 * (inertia + inertia.T))


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


def _parallel_axis(inertia_centroid: np.ndarray, mass_kg: float, offset_m: np.ndarray) -> np.ndarray:
    offset = np.asarray(offset_m, dtype=float)
    return np.asarray(inertia_centroid, dtype=float) + float(mass_kg) * (
        float(np.dot(offset, offset)) * np.eye(3) - np.outer(offset, offset)
    )


def combine_platform_and_payload(
    platform: StructuralMassProperties,
    payload: PayloadMassProperties,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return total mass, combined CG and inertia about the combined CG."""
    mp = float(platform.mass_kg)
    ml = float(payload.mass_kg)
    total = mp + ml
    if total <= 0.0:
        raise ValueError("Combined moving mass must be positive.")
    cp = np.asarray(platform.centroid_m, dtype=float)
    cl = np.asarray(payload.centroid_m, dtype=float)
    cg = (mp * cp + ml * cl) / total
    inertia = _parallel_axis(platform.inertia_centroid_kg_m2, mp, cp - cg)
    if ml > 0.0:
        inertia += _parallel_axis(payload.inertia_centroid_kg_m2, ml, cl - cg)
    inertia = 0.5 * (inertia + inertia.T)
    if np.any(np.linalg.eigvalsh(inertia) <= 0.0):
        raise ValueError("Combined moving-body inertia is not positive definite.")
    return total, cg, inertia


def rollout_parameters_from_structure(
    base: RolloutParameters,
    properties: StructuralMassProperties,
    *,
    mass_scope: str = "moving_assembly",
    module_points_m: np.ndarray | None = None,
    payload: PayloadMassProperties | None = None,
) -> RolloutParameters:
    """Return reduced-order parameters derived from validated structural properties.

    ``moving_assembly`` means the tetrahedral body already represents the complete
    moving assembly. ``platform_only`` means the volume mesh represents only the
    platform and therefore requires explicit payload mass properties whenever the
    configured payload mass is non-zero. Module X/Y locations can be supplied from
    the structural manifest so the dynamics and FEA use the same physical footprint.
    """
    scope = str(mass_scope).strip().lower()
    dims = np.asarray(properties.dimensions_m, dtype=float)
    if np.any(dims <= 0.0):
        raise ValueError("Geometry dimensions must be positive.")

    common: dict[str, object] = {
        "body_length_m": float(dims[0]),
        "body_width_m": float(dims[1]),
        "body_height_m": float(dims[2]),
    }
    if module_points_m is not None:
        module_points = np.asarray(module_points_m, dtype=float)
        if module_points.shape != (8, 3) or np.any(~np.isfinite(module_points)):
            raise ValueError("module_points_m must have shape (8, 3) with finite coordinates.")
        common["module_x_m"] = tuple(float(v) for v in module_points[:, 0])
        common["module_y_m"] = tuple(float(v) for v in module_points[:, 1])

    if scope == "moving_assembly":
        total_mass = float(properties.mass_kg)
        platform_mass = total_mass - float(base.payload_mass_kg)
        if platform_mass <= 0.0:
            raise ValueError(
                "For moving_assembly coupling, integrated mesh mass must exceed the configured payload mass."
            )
        inertia = np.asarray(properties.inertia_centroid_kg_m2, dtype=float)
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

    if scope != "platform_only":
        raise ValueError("mass_scope must be 'moving_assembly' or 'platform_only'.")

    if payload is None:
        if float(base.payload_mass_kg) > 0.0:
            raise ValueError(
                "platform_only coupling requires explicit payload mass, centroid, and inertia properties; "
                "retaining stale total inertia is not scientifically valid."
            )
        payload = PayloadMassProperties(0.0, np.asarray(properties.centroid_m, dtype=float), np.zeros((3, 3)))

    _, _, inertia = combine_platform_and_payload(properties, payload)
    return replace(
        base,
        platform_mass_kg=float(properties.mass_kg),
        payload_mass_kg=float(payload.mass_kg),
        inertia_xx_kg_m2=float(inertia[0, 0]),
        inertia_yy_kg_m2=float(inertia[1, 1]),
        inertia_zz_kg_m2=float(inertia[2, 2]),
        inertia_xy_kg_m2=float(inertia[0, 1]),
        inertia_xz_kg_m2=float(inertia[0, 2]),
        inertia_yz_kg_m2=float(inertia[1, 2]),
        **common,
    )
