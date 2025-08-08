from .trajectory import slerp_traj, linear_traj, Order

# =================== Trajectory Interface ===================
from typing import List
import numpy as np
import quaternion

from geometry_msgs.msg import Pose
from pyrobopath_ros.msg import ScheduleTrajectoryPoint


def create_trajectory(points: List[ScheduleTrajectoryPoint]):
    segments = []
    for i in range(1, len(points)):
        ps = points[i - 1]
        pe = points[i]
        t_start = ps.time_from_start.to_sec()
        t_end = pe.time_from_start.to_sec()
        segment = TrajectorySegment(t_start, ps.pose, t_end, pe.pose)
        segments.append(segment)
    return segments


def pose_to_np(pose: Pose):
    p = pose.position
    q = pose.orientation
    position = np.array([p.x, p.y, p.z])
    quat = quaternion.quaternion(q.w, q.x, q.y, q.z)
    return position, quat


class TrajectorySegment:
    def __init__(
        self, start_time: float, start_pose: Pose, end_time: float, end_pose: Pose
    ):
        ps, qs = pose_to_np(start_pose)
        pe, qe = pose_to_np(end_pose)

        self.time_from_start = start_time
        self.duration = end_time - start_time
        self.f_pos = linear_traj(ps, pe, self.duration, scaling=Order.FIRST)
        self.f_orient = slerp_traj(qs, qe, self.duration, scaling=Order.FIRST)

    def sample(self, t):
        t = np.clip(t, 0.0, self.duration)
        p = self.f_pos[0](t)
        q = self.f_orient[0](t)
        v = self.f_pos[1](t)
        ω = self.f_orient[1](t)
        return p, q, v, ω
