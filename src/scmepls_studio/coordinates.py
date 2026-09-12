from __future__ import annotations

import numpy as np


WORLD_AXES = {
    "x": "roll-out",
    "y": "lateral",
    "z": "vertical",
}


def rotation_matrix_body_to_world(roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Return the right-handed ZYX body-to-world rotation matrix.

    SC-MEPLS uses X = roll-out, Y = lateral and Z = vertical. Positive roll is
    about +X, positive pitch about +Y and positive yaw about +Z.
    """
    roll = float(roll_rad)
    pitch = float(pitch_rad)
    yaw = float(yaw_rad)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def world_vector_to_body(vector_xyz: np.ndarray, roll_rad: float, pitch_rad: float, yaw_rad: float) -> np.ndarray:
    """Rotate a world-frame vector into the instantaneous body frame."""
    vector = np.asarray(vector_xyz, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("A finite XYZ vector is required.")
    return rotation_matrix_body_to_world(roll_rad, pitch_rad, yaw_rad).T @ vector


def module_vertical_offsets_m(
    x_m: np.ndarray,
    y_m: np.ndarray,
    roll_rad: float,
    pitch_rad: float,
    yaw_rad: float = 0.0,
) -> np.ndarray:
    """Return exact vertical offsets of module points relative to the body origin.

    Module reference points are assumed to lie in the body XY plane. This uses the
    same ZYX rotation convention as the 3-D viewer. For small angles it reduces to
    ``roll*y - pitch*x``; the negative pitch sign is intentional and follows the
    standard right-handed +Y pitch convention.
    """
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    if x.shape != y.shape or np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("Module X/Y coordinates must be finite arrays with equal shape.")
    rotation = rotation_matrix_body_to_world(roll_rad, pitch_rad, yaw_rad)
    points = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size, dtype=float)))
    offsets = (points @ rotation.T)[:, 2]
    return offsets.reshape(x.shape)


def small_angle_module_gaps_m(
    center_gap_m: float,
    x_m: np.ndarray,
    y_m: np.ndarray,
    roll_rad: float,
    pitch_rad: float,
) -> np.ndarray:
    """Small-angle gap relation retained for controller estimation and diagnostics."""
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    return float(center_gap_m) + float(roll_rad) * y - float(pitch_rad) * x
