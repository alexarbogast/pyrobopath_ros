"""
A path maps a sampling parameter "s" to an output in ℝⁿ or SO(3).
A path sampled along a time-scaling s(t) defines a trajectory.

p(s): [0, 1] → ℝⁿ or SO(3)
p(s(t)): [0, T] → ℝⁿ or SO(3)
"""

import quaternion


def slerp_path(q_start, q_end):
    def f(s):
        return quaternion.slerp_evaluate(q_start, q_end, s)

    def f_dot(s, s_dot):
        aa = quaternion.as_rotation_vector(q_start.inverse() * q_end)
        aa_quat = quaternion.from_vector_part(aa * s_dot)
        q_interp = f(s)
        return quaternion.as_vector_part(q_interp * aa_quat * q_interp.conjugate())

    return f, f_dot


def linear_path(p_start, p_end):
    def f(s):
        fs = p_start + (p_end - p_start) * s
        return fs

    def f_dot(s, s_dot):
        fdot_sdot = (p_end - p_start) * s_dot
        return fdot_sdot

    return f, f_dot
