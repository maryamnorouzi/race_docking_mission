"""Pure math helpers for the search mission node. No ROS dependency.

Includes the body-frame -> odom-consistent twist conversion ported from
race_controller.cpp's euler_rate_transform_()/build_odom_state_() (read,
not modified -- simple_controller_pkg is untouched). This is what lets
race_docking_mission publish actual_odometry.twist already in the same
representation race_docking_control's backstepping law needs (odom-frame
linear velocity + roll/pitch/yaw *rates*, not raw body angular velocity),
so the control package needs no attitude-dependent rotation for z2.
"""

from __future__ import annotations

import math

import numpy as np


def triangle_wave(elapsed: float, amplitude: float, rate: float) -> tuple[float, float]:
    """Position and velocity of a triangle wave starting at 0, ramping up
    to +amplitude, back down to -amplitude, and back to 0, at a constant
    |rate|, repeating forever. Used for HOLD's tag-reacquire search motion
    (a slow z wobble + yaw sweep around the pose that last saw the tag).

    Returns (position, velocity).
    """
    if amplitude <= 0.0 or rate <= 0.0:
        return 0.0, 0.0
    quarter = amplitude / rate
    period = 4.0 * quarter
    t = elapsed % period
    if t < quarter:
        return rate * t, rate
    if t < 3.0 * quarter:
        return amplitude - rate * (t - quarter), -rate
    return -amplitude + rate * (t - 3.0 * quarter), rate


def wrap_to_pi(angle: float) -> float:
    """Wrap an angle (radians) into (-pi, pi]."""
    return math.atan2(math.sin(angle), math.cos(angle))


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    """Extract yaw (radians) from a quaternion, ignoring roll/pitch."""
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def rpy_from_quaternion(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
    """(roll, pitch, yaw), radians, REP-103 convention."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    sinp = max(-1.0, min(1.0, sinp))
    pitch = math.asin(sinp)

    yaw = yaw_from_quaternion(x, y, z, w)
    return roll, pitch, yaw


def quaternion_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Standard ZYX (yaw-pitch-roll) intrinsic-rotation quaternion, REP-103
    roll/pitch/yaw convention. Returns (x, y, z, w).
    """
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (x, y, z, w)


def rotation_matrix_from_quaternion(x: float, y: float, z: float, w: float) -> np.ndarray:
    """3x3 rotation matrix (odom_from_body) for quaternion (x,y,z,w)."""
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=float)


def euler_rate_transform(roll: float, pitch: float) -> np.ndarray:
    """3x3 matrix mapping body angular velocity (p, q, r) to Euler angle
    rates (roll_dot, pitch_dot, yaw_dot). Ported verbatim (read-only, not
    modified) from race_controller.cpp's euler_rate_transform_().
    """
    cos_pitch = math.cos(pitch)
    if -1.0e-4 < cos_pitch < 1.0e-4:
        cos_pitch = 1.0e-4
    tan_pitch = max(-1000.0, min(1000.0, math.tan(pitch)))

    t = np.zeros((3, 3), dtype=float)
    t[0, 0] = 1.0
    t[0, 1] = math.sin(roll) * tan_pitch
    t[0, 2] = math.cos(roll) * tan_pitch
    t[1, 1] = math.cos(roll)
    t[1, 2] = -math.sin(roll)
    t[2, 1] = math.sin(roll) / cos_pitch
    t[2, 2] = math.cos(roll) / cos_pitch
    return t


def invert_euler_rate_transform(roll: float, pitch: float) -> np.ndarray:
    """True inverse (not transpose -- T_theta isn't orthogonal) of
    euler_rate_transform(roll, pitch). Maps Euler angle rates back to body
    angular velocity. Used for the angular block of J^-1 in the Z1/z2
    tracking-error law (race_docking_control's J^-1 * z1 / beta step).
    """
    return np.linalg.inv(euler_rate_transform(roll, pitch))


def quaternion_error(
    q_desired: tuple[float, float, float, float],
    q_actual: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Small-angle roll/pitch/yaw error from two quaternions: q_error =
    q_desired^-1 * q_actual, normalized, sign-flipped so w>=0 (shortest
    rotation), error = 2*(x, y, z) of q_error. Ported verbatim from
    docking_control_node.cpp's compute_tracking_errors_() convention.
    """
    dx, dy, dz, dw = q_desired
    q_desired_inv = (-dx, -dy, -dz, dw)  # unit quaternion: inverse = conjugate
    ex, ey, ez, ew = _quat_mul(q_desired_inv, q_actual)
    norm = math.sqrt(ex * ex + ey * ey + ez * ez + ew * ew)
    if norm > 1.0e-12:
        ex, ey, ez, ew = ex / norm, ey / norm, ez / norm, ew / norm
    if ew < 0.0:
        ex, ey, ez = -ex, -ey, -ez
    return 2.0 * ex, 2.0 * ey, 2.0 * ez


def _quat_mul(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    """Hamilton product q1*q2, both (x, y, z, w)."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return (
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    )


def transport_linear_velocity(
    v_body: tuple[float, float, float],
    omega_body: tuple[float, float, float],
    offset_pos: tuple[float, float, float],
) -> np.ndarray:
    """Rigid-body transport theorem: linear velocity of a point rigidly
    offset from the body origin by offset_pos (expressed in body axes),
    itself still expressed in body axes: v_point = v_body + omega_body x
    offset_pos. Used to get the auv_dock_point's velocity from base_link's
    raw EKF twist -- angular velocity needs no such correction (it's the
    same for every point on a rigid body).
    """
    return np.asarray(v_body, dtype=float) + np.cross(
        np.asarray(omega_body, dtype=float), np.asarray(offset_pos, dtype=float))


def body_twist_to_odom_consistent(
    orientation: tuple[float, float, float, float],
    linear_body: tuple[float, float, float],
    angular_body: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Convert a raw body-frame twist (as published by the EKF: linear/angular
    velocity in child_frame_id=base_link) into the odom-consistent
    representation race_docking_control's backstepping law expects:
    odom-frame linear velocity + roll/pitch/yaw *rates* (not raw body
    angular velocity). Ported from build_odom_state_() in
    race_controller.cpp (read-only reference, not modified).

    Returns (linear_odom, euler_rates) -- each a 3-tuple.
    """
    x, y, z, w = orientation
    roll, pitch, _yaw = rpy_from_quaternion(x, y, z, w)
    r_odom_from_body = rotation_matrix_from_quaternion(x, y, z, w)
    linear_odom = r_odom_from_body @ np.asarray(linear_body, dtype=float)
    euler_rates = euler_rate_transform(roll, pitch) @ np.asarray(angular_body, dtype=float)
    return tuple(linear_odom), tuple(euler_rates)
