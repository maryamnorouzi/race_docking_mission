"""Unit tests for the quintic trajectory module (no ROS dependency)."""

import math

import pytest

from race_docking_mission import quintic


def test_rest_to_rest_boundary_conditions():
    coeffs = quintic.solve_quintic(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, T=4.0)

    p0, v0, a0 = quintic.evaluate(coeffs, 0.0)
    assert p0 == pytest.approx(0.0)
    assert v0 == pytest.approx(0.0)
    assert a0 == pytest.approx(0.0)

    pT, vT, aT = quintic.evaluate(coeffs, 4.0)
    assert pT == pytest.approx(1.0)
    assert vT == pytest.approx(0.0, abs=1e-9)
    assert aT == pytest.approx(0.0, abs=1e-9)


def test_rest_to_rest_peak_velocity_matches_closed_form():
    delta, T = 2.0, 5.0
    coeffs = quintic.solve_quintic(0.0, 0.0, 0.0, delta, 0.0, 0.0, T=T)
    expected_peak = 15.0 * abs(delta) / (8.0 * T)
    assert quintic.peak_abs_velocity(coeffs) == pytest.approx(expected_peak, rel=1e-3)


def test_nonzero_initial_conditions_are_honored():
    coeffs = quintic.solve_quintic(p0=1.0, v0=0.5, a0=-0.2, pT=3.0, vT=0.0, aT=0.0, T=3.0)
    p, v, a = quintic.evaluate(coeffs, 0.0)
    assert (p, v, a) == pytest.approx((1.0, 0.5, -0.2))
    p, v, a = quintic.evaluate(coeffs, 3.0)
    assert (p, v, a) == pytest.approx((3.0, 0.0, 0.0), abs=1e-9)


def test_c2_handoff_between_two_segments():
    """The velocity/acceleration of segment A at the handoff instant become
    segment B's initial conditions -- this is the C2 continuity the mission
    node relies on for receding-horizon replanning."""
    seg_a = quintic.solve_quintic(0.0, 0.0, 0.0, 2.0, 0.0, 0.0, T=4.0)
    t_handoff = 1.5
    p_h, v_h, a_h = quintic.evaluate(seg_a, t_handoff)

    seg_b = quintic.solve_quintic(p_h, v_h, a_h, 5.0, 0.0, 0.0, T=6.0)
    p0, v0, a0 = quintic.evaluate(seg_b, 0.0)
    assert (p0, v0, a0) == pytest.approx((p_h, v_h, a_h))


def test_evaluate_clamps_past_duration_by_default():
    coeffs = quintic.solve_quintic(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, T=2.0)
    at_T = quintic.evaluate(coeffs, 2.0)
    past_T = quintic.evaluate(coeffs, 10.0)
    assert past_T == pytest.approx(at_T)


def test_duration_for_velocity_limit():
    T = quintic.duration_for_velocity_limit(delta=3.0, v_max=0.3)
    assert T == pytest.approx(15.0 * 3.0 / (8.0 * 0.3))
    assert quintic.duration_for_velocity_limit(delta=0.0, v_max=0.3) == 0.0
    assert quintic.duration_for_velocity_limit(delta=3.0, v_max=0.0) == 0.0


def test_yaw_wraps_are_not_this_modules_job():
    # Sanity: quintic is purely scalar; wrap-around handling belongs to the
    # mission node's yaw channel, not to this module.
    coeffs = quintic.solve_quintic(3.0, 0.0, 0.0, 3.0 + math.pi, 0.0, 0.0, T=2.0)
    p, _, _ = quintic.evaluate(coeffs, 2.0)
    assert p == pytest.approx(3.0 + math.pi)


def test_solve_quintic_rejects_non_positive_duration():
    with pytest.raises(ValueError):
        quintic.solve_quintic(0.0, 0.0, 0.0, 1.0, 0.0, 0.0, T=0.0)
