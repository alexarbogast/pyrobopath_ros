"""
A polynomial time-scaling converts a time "t" to a parameter in the range of 0 to 1.
A path sampled at a time-scaling s(t) gives a trajectory.

s(t): [0, T] → [0, 1]
"""

import numpy as np
from enum import Enum


def first_order_scaling(tf):
    s = np.polynomial.Polynomial([0.0, 1 / tf])
    s_dot = s.deriv()
    return s, s_dot


def third_order_scaling(tf):
    s = np.polynomial.Polynomial([0.0, 0.0, 3 / tf**2, -2 / tf**3])
    s_dot = s.deriv()
    return s, s_dot


def fifth_order_scaling(tf):
    s = np.polynomial.Polynomial([0.0, 0.0, 0.0, 10 / tf**3, -15 / tf**4, 6 / tf**5])
    s_dot = s.deriv()
    return s, s_dot


class Order(Enum):
    FIRST = first_order_scaling
    THIRD = third_order_scaling
    FIFTH = fifth_order_scaling
