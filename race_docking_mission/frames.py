"""SE(3) pose composition, in plain (position, quaternion) tuples.

No ROS dependency. Used to re-express the AprilTag fuser's base_link-
relative dock measurement as auv_dock_point-relative (station's dock
point relative to the AUV's *own* dock point, not its base_link) --
compose(invert(base_to_dock), base_to_station).

Quaternions are (x, y, z, w) throughout, matching geometry_msgs.
"""

from __future__ import annotations

import math

import numpy as np


def quat_normalize(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return (0.0, 0.0, 0.0, 1.0)
    return (x / n, y / n, z / n, w / n)


def quat_mul(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Hamilton product q1 * q2 (apply q2 first, then q1), xyzw convention."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return (x, y, z, w)


def quat_rotate_vector(
    q: tuple[float, float, float, float], v: np.ndarray,
) -> np.ndarray:
    """Rotate a 3-vector by quaternion q (xyzw)."""
    x, y, z, w = q
    qv = np.array([x, y, z], dtype=float)
    v = np.asarray(v, dtype=float)
    uv = np.cross(qv, v)
    uuv = np.cross(qv, uv)
    return v + 2.0 * (w * uv + uuv)


def compose(
    pos_a: tuple[float, float, float], quat_a: tuple[float, float, float, float],
    pos_b: tuple[float, float, float], quat_b: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Compose two poses: T_a * T_b (b expressed in a's frame -> result in
    a's parent frame).
    """
    pos_a = np.asarray(pos_a, dtype=float)
    pos = pos_a + quat_rotate_vector(quat_a, pos_b)
    quat = quat_normalize(quat_mul(quat_a, quat_b))
    return pos, quat


def invert(
    pos: tuple[float, float, float], quat: tuple[float, float, float, float],
) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Invert an SE(3) pose T=(pos, quat): returns T^-1 such that
    compose(*invert(pos, quat), pos, quat) is the identity transform.
    """
    quat_inv = quat_normalize(quat)
    quat_inv = (-quat_inv[0], -quat_inv[1], -quat_inv[2], quat_inv[3])
    pos_inv = -quat_rotate_vector(quat_inv, np.asarray(pos, dtype=float))
    return pos_inv, quat_inv
