"""
A trajectory is a path sampled at a time-scaling. This module provides
convenience functions for commonly combined paths and time-scalings.

f(t) ≡ p(s(t))
f(t): [0, T] → ℝⁿ or SO(3)
"""

from .path import *
from .time_scaling import *


def slerp_traj(q_start, q_end, tf, scaling=Order.FIRST):
    f, f_dot = slerp_path(q_start, q_end)
    s, s_dot = scaling(tf)
    return lambda t: f(s(t)), lambda t: f_dot(s(t), s_dot(t))


def linear_traj(p_start, p_end, tf, scaling=Order.FIFTH):
    f, f_dot = linear_path(p_start, p_end)
    s, s_dot = scaling(tf)
    return lambda t: f(s(t)), lambda t: f_dot(None, s_dot(t))
