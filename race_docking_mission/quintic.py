"""Closed-form quintic (5th-order) point-to-point trajectory segments.

A single channel's motion over one segment is

    p(t) = c0 + c1*t + c2*t^2 + c3*t^3 + c4*t^4 + c5*t^5,   t in [0, T]

fully determined by position/velocity/acceleration boundary conditions at
t=0 and t=T (6 equations, 6 unknowns). This module has no ROS dependency
so it can be unit tested directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuinticCoeffs:
    """Coefficients c0..c5 for one quintic segment, valid on t in [0, duration]."""

    c0: float
    c1: float
    c2: float
    c3: float
    c4: float
    c5: float
    duration: float


def solve_quintic(
    p0: float, v0: float, a0: float,
    pT: float, vT: float, aT: float,
    T: float,
) -> QuinticCoeffs:
    """Solve the 6x6 (block-triangular) boundary value problem in closed form.

    c0, c1, c2 follow directly from the t=0 conditions. c3, c4, c5 solve the
    remaining 3x3 linear system from the t=T conditions.

    Raises ValueError if T <= 0.
    """
    if T <= 0.0:
        raise ValueError(f"quintic duration T must be > 0, got {T}")

    c0 = p0
    c1 = v0
    c2 = a0 / 2.0

    # Remaining contribution required from the cubic/quartic/quintic terms.
    b_p = pT - (c0 + c1 * T + c2 * T ** 2)
    b_v = vT - (c1 + 2.0 * c2 * T)
    b_a = aT - (2.0 * c2)

    A = np.array([
        [T ** 3,      T ** 4,      T ** 5],
        [3 * T ** 2,  4 * T ** 3,  5 * T ** 4],
        [6 * T,       12 * T ** 2, 20 * T ** 3],
    ], dtype=float)
    b = np.array([b_p, b_v, b_a], dtype=float)

    c3, c4, c5 = np.linalg.solve(A, b)

    return QuinticCoeffs(c0=c0, c1=c1, c2=c2, c3=float(c3), c4=float(c4), c5=float(c5), duration=T)


def evaluate(coeffs: QuinticCoeffs, t: float, clamp: bool = True) -> tuple[float, float, float]:
    """Return (position, velocity, acceleration) at time t.

    By default t is clamped into [0, coeffs.duration] first -- evaluating a
    quintic past its own validity window holds at the t=T boundary
    conditions (position, velocity=0, acceleration=0), which is exactly the
    "hold target until next replan" behavior the mission needs. Pass
    clamp=False to evaluate the raw polynomial outside [0, T].
    """
    if clamp:
        t = min(max(t, 0.0), coeffs.duration)

    p = (coeffs.c0 + coeffs.c1 * t + coeffs.c2 * t ** 2
         + coeffs.c3 * t ** 3 + coeffs.c4 * t ** 4 + coeffs.c5 * t ** 5)
    v = (coeffs.c1 + 2.0 * coeffs.c2 * t + 3.0 * coeffs.c3 * t ** 2
         + 4.0 * coeffs.c4 * t ** 3 + 5.0 * coeffs.c5 * t ** 4)
    a = (2.0 * coeffs.c2 + 6.0 * coeffs.c3 * t + 12.0 * coeffs.c4 * t ** 2
         + 20.0 * coeffs.c5 * t ** 3)
    return p, v, a


def duration_for_velocity_limit(delta: float, v_max: float) -> float:
    """Rest-to-rest quintic peak-velocity duration: T = 15*|delta| / (8*v_max).

    Returns 0.0 if delta is ~0 (no motion needed on this channel) or if
    v_max <= 0 (treated as unconstrained by this channel).
    """
    if v_max <= 0.0 or abs(delta) < 1e-12:
        return 0.0
    return 15.0 * abs(delta) / (8.0 * v_max)


def peak_abs_velocity(coeffs: QuinticCoeffs, samples: int = 200) -> float:
    """Numerically estimate the peak |velocity| over [0, duration].

    Used only for the post-hoc "did we exceed the velocity limit by more
    than 10%" warning check -- the closed-form v_peak formula assumes a
    rest-to-rest segment, which does not hold exactly once nonzero
    boundary velocity/acceleration are inherited from a previous segment.
    """
    if samples < 2:
        samples = 2
    ts = np.linspace(0.0, coeffs.duration, samples)
    vs = [evaluate(coeffs, float(t), clamp=False)[1] for t in ts]
    return float(max(abs(v) for v in vs))
