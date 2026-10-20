"""Unit tests for race_docking_mission's pure-math helpers (no ROS dependency)."""

import math

import numpy as np
import pytest

from race_docking_mission.geometry_utils import (
    body_twist_to_odom_consistent,
    euler_rate_transform,
    invert_euler_rate_transform,
    quaternion_error,
    quaternion_from_rpy,
    rpy_from_quaternion,
    transport_linear_velocity,
    triangle_wave,
    wrap_to_pi,
    yaw_from_quaternion,
)

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)


def test_quaternion_rpy_round_trip():
    for roll_deg, pitch_deg, yaw_deg in (
        (0.0, 0.0, 0.0), (10.0, -20.0, 170.0), (-5.0, 5.0, -179.0),
    ):
        q = quaternion_from_rpy(
            math.radians(roll_deg), math.radians(pitch_deg), math.radians(yaw_deg))
        roll, pitch, yaw = rpy_from_quaternion(*q)
        assert roll == pytest.approx(math.radians(roll_deg), abs=1e-9)
        assert pitch == pytest.approx(math.radians(pitch_deg), abs=1e-9)
        assert yaw == pytest.approx(math.radians(yaw_deg), abs=1e-9)


def test_wrap_to_pi():
    assert wrap_to_pi(0.0) == pytest.approx(0.0)
    assert wrap_to_pi(2.0 * math.pi) == pytest.approx(0.0, abs=1e-9)
    assert wrap_to_pi(math.pi + 0.1) == pytest.approx(-math.pi + 0.1, abs=1e-9)


def test_euler_rate_transform_is_identity_at_level_attitude():
    # roll=pitch=0 -> Euler rates equal body angular velocity exactly.
    t = euler_rate_transform(0.0, 0.0)
    assert np.allclose(t, np.eye(3), atol=1e-9)


def test_body_twist_to_odom_consistent_level_attitude_passthrough():
    # At identity orientation, odom-frame linear velocity == body linear
    # velocity, and Euler rates == body angular velocity.
    linear_odom, euler_rates = body_twist_to_odom_consistent(
        IDENTITY_QUAT, (1.0, 2.0, 3.0), (0.1, 0.2, 0.3),
    )
    assert linear_odom == pytest.approx((1.0, 2.0, 3.0))
    assert euler_rates == pytest.approx((0.1, 0.2, 0.3))


def test_body_twist_to_odom_consistent_rotates_linear_by_yaw():
    q = quaternion_from_rpy(0.0, 0.0, math.pi / 2.0)  # 90 deg yaw
    linear_odom, _ = body_twist_to_odom_consistent(q, (1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    # Forward (body +x) at 90 deg yaw points along odom +y.
    assert linear_odom == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_yaw_from_quaternion_matches_rpy():
    q = quaternion_from_rpy(0.1, -0.2, 0.7)
    assert yaw_from_quaternion(*q) == pytest.approx(0.7, abs=1e-9)


def test_transport_linear_velocity_no_offset_is_passthrough():
    v = transport_linear_velocity((1.0, 2.0, 3.0), (0.1, 0.2, 0.3), (0.0, 0.0, 0.0))
    assert tuple(v) == pytest.approx((1.0, 2.0, 3.0))


def test_transport_linear_velocity_pure_rotation_about_offset_point():
    # Body spinning about its own +z (yaw rate 1 rad/s) with zero linear
    # velocity: a point 1 m along body +x should appear to move at 1 m/s
    # along body +y (omega x r, standard rigid-body rotation).
    v = transport_linear_velocity((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0))
    assert tuple(v) == pytest.approx((0.0, 1.0, 0.0), abs=1e-9)


def test_triangle_wave_zero_amplitude_or_rate_is_flat():
    assert triangle_wave(5.0, 0.0, 1.0) == (0.0, 0.0)
    assert triangle_wave(5.0, 1.0, 0.0) == (0.0, 0.0)


def test_triangle_wave_key_points():
    amplitude, rate = 0.3, 0.1
    quarter = amplitude / rate  # 3.0s: 0 -> +amplitude
    pos, vel = triangle_wave(0.0, amplitude, rate)
    assert (pos, vel) == pytest.approx((0.0, rate))
    pos, vel = triangle_wave(quarter, amplitude, rate)
    assert (pos, vel) == pytest.approx((amplitude, -rate))
    pos, vel = triangle_wave(3.0 * quarter, amplitude, rate)
    assert (pos, vel) == pytest.approx((-amplitude, rate))
    pos, vel = triangle_wave(4.0 * quarter, amplitude, rate)
    assert (pos, vel) == pytest.approx((0.0, rate))


def test_triangle_wave_repeats_with_period():
    amplitude, rate = 0.3, 0.1
    period = 4.0 * amplitude / rate
    a = triangle_wave(1.7, amplitude, rate)
    b = triangle_wave(1.7 + period, amplitude, rate)
    assert a == pytest.approx(b)


def test_invert_euler_rate_transform_is_true_inverse():
    for roll_deg, pitch_deg in ((0.0, 0.0), (15.0, -20.0), (-30.0, 10.0)):
        roll, pitch = math.radians(roll_deg), math.radians(pitch_deg)
        t = euler_rate_transform(roll, pitch)
        t_inv = invert_euler_rate_transform(roll, pitch)
        assert np.allclose(t @ t_inv, np.eye(3), atol=1e-9)
        assert np.allclose(t_inv @ t, np.eye(3), atol=1e-9)


def test_invert_euler_rate_transform_is_not_the_transpose_when_tilted():
    # At nonzero roll/pitch, T_theta isn't orthogonal, so transpose != inverse.
    roll, pitch = math.radians(20.0), math.radians(15.0)
    t = euler_rate_transform(roll, pitch)
    t_inv = invert_euler_rate_transform(roll, pitch)
    assert not np.allclose(t_inv, t.T, atol=1e-6)


def test_quaternion_error_zero_when_equal():
    q = quaternion_from_rpy(0.1, -0.2, 0.7)
    err = quaternion_error(q, q)
    assert err == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)


def test_quaternion_error_small_yaw_offset_matches_angle():
    q_desired = IDENTITY_QUAT
    yaw = math.radians(10.0)
    q_actual = quaternion_from_rpy(0.0, 0.0, yaw)
    _, _, yaw_err = quaternion_error(q_desired, q_actual)
    # Small-angle: error[2] = 2*sin(yaw/2) ~= yaw for small yaw.
    assert yaw_err == pytest.approx(2.0 * math.sin(yaw / 2.0), abs=1e-9)


def test_quaternion_error_takes_shortest_path():
    # 350 deg actual yaw vs 0 deg desired: error should reflect the short
    # (-10 deg) way around, not the long (+350 deg) way.
    q_desired = IDENTITY_QUAT
    q_actual = quaternion_from_rpy(0.0, 0.0, math.radians(-10.0))
    err_short = quaternion_error(q_desired, q_actual)
    q_actual_long_way = quaternion_from_rpy(0.0, 0.0, math.radians(350.0))
    err_long = quaternion_error(q_desired, q_actual_long_way)
    assert err_short == pytest.approx(err_long, abs=1e-9)
