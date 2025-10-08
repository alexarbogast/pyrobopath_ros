from typing import List
from gcodeparser import GcodeParser
from copy import deepcopy
import numpy as np

# ros
from rclpy.duration import Duration
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import Pose

# pyrobopath
from pyrobopath.toolpath import Toolpath
from pyrobopath.toolpath import Rotation, Transform
from pyrobopath.toolpath_scheduling import MultiAgentToolpathSchedule, ToolpathSchedule

# pyrobopath_ros
from pyrobopath_msgs.msg import ScheduleTrajectoryPoint, ScheduleTrajectory

MAX_BACKWARDS_TIME = 1e-8
TIME_DIFF_THRESHOLD = 1e-8


def toolpath_from_gcode(filepath) -> Toolpath:
    """
    Parse a G-code file into an internal toolpath representation.

    Parameters
    ----------
    filepath : str
        Absolute path to a G-code file.

    Returns
    -------
    Toolpath
        Toolpath created from the input file.
    """

    with open(filepath, "r") as f:
        gcode = f.read()
    parsed_gcode = GcodeParser(gcode)

    toolpath = Toolpath.from_gcode(parsed_gcode.lines)
    return toolpath


def schedule_info_string(schedule: MultiAgentToolpathSchedule):
    """
    Returns a string containing the schedule duration, total number of events,
    and events for each agent

    Parameters
    ----------
    schedule : MultiAgentToolpathSchedule
        The schedule to inspect.

    Returns
    -------
    str
        A string containing the schedule info
    """
    info = ""
    info += f"Schedule duration: {schedule.duration()}\n"
    info += f"Total Events: {schedule.n_events()}\n"
    agents_info = "Agent Events: "
    for agent, sched in schedule.schedules.items():
        agents_info += f"{agent}: {len(sched._events)}, "
    info += agents_info + "\n"
    return info


## Trajectory and schedule modification
def offset_trajectory_times(traj: List[JointTrajectoryPoint], offset: float):
    """
    Apply a time offset to each point in a joint trajectory.

    Parameters
    ----------
    traj : list of JointTrajectoryPoint
        List of trajectory points to modify.
    offset : float
        Time offset in seconds.

    Notes
    -----
    Modifies `traj` in-place.
    """
    for point in traj:
        time_from_start = Duration.from_msg(point.time_from_start).nanoseconds
        point.time_from_start = Duration(
            nanoseconds=time_from_start + int(offset * 1e9)
        ).to_msg()


def compile_schedule_plans(
    plans: List[FollowJointTrajectory.Goal],
) -> FollowJointTrajectory.Goal:
    """
    Merge multiple trajectory goals into a single trajectory.

    Parameters
    ----------
    plans : list of FollowJointTrajectory.Goal
        List of goals to merge. All plans must share tolerance settings.

    Returns
    -------
    FollowJointTrajectory.Goal
        Combined trajectory goal.
    """
    goal = FollowJointTrajectory.Goal()
    goal.path_tolerance = plans[0].path_tolerance
    goal.goal_tolerance = plans[0].goal_tolerance
    goal.goal_time_tolerance = plans[0].goal_time_tolerance
    goal.trajectory = deepcopy(plans[0].trajectory)

    for p in plans[1:]:
        # occasionally the accumulated error in trajectory times is nano-seconds
        # in the past. Replace the time with the value from the schedule if it's
        # below the threshold
        t1_end = goal.trajectory.points[-1].time_from_start
        t2_start = p.trajectory.points[0].time_from_start

        diff = (t1_end.sec + t1_end.nanosec * 1e-9) - (
            t2_start.sec + t2_start.nanosec * 1e-9
        )

        if abs(diff) < MAX_BACKWARDS_TIME:
            goal.trajectory.points.extend(p.trajectory.points[1:])
        else:
            goal.trajectory.points.extend(p.trajectory.points[:])

    return goal


def create_pose(point: np.ndarray, rot_offset: Rotation):
    """
    Create a Pose from a 3D point with an orientation defined by rotation offset.

    Parameters
    ----------
    point : ndarray, shape (3,)
        Cartesian position [x, y, z].
    rot_offset : Rotation
        Additional rotation to apply.

    Returns
    -------
    Pose
        The resulting pose.
    """
    pose = Pose()
    pose.position.x = point[0]
    pose.position.y = point[1]
    pose.position.z = point[2]

    theta = np.arctan2(point[1], point[0])
    rot = Rotation([np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2)])
    q = (rot @ rot_offset).quat

    pose.orientation.w = q.w
    pose.orientation.x = q.x
    pose.orientation.y = q.y
    pose.orientation.z = q.z
    return pose


def create_schedule_trajectory(
    sched: ToolpathSchedule, rot_offset: Rotation, transform: Transform
) -> ScheduleTrajectory:
    """
    Convert a toolpath schedule into a ROS 2 trajectory message.

    Parameters
    ----------
    sched : ToolpathSchedule
        Toolpath schedule containing events and trajectories.
    rot_offset : Rotation
        Rotation offset to apply to each pose.
    transform : Transform
        Transform to apply to points before pose creation.

    Returns
    -------
    ScheduleTrajectory
        Compiled trajectory message with time-stamped poses.
    """
    traj_points = []
    initial_point = sched._events[0].traj[0]
    initial_point_base = transform * initial_point.data
    traj_points.append(
        (initial_point.time, create_pose(initial_point_base, rot_offset))
    )
    for event in sched._events:
        for p in event.traj:
            if abs(p.time - traj_points[-1][0]) < TIME_DIFF_THRESHOLD:
                continue
            p_base = transform * p.data
            traj_points.append((p.time, create_pose(p_base, rot_offset)))

    traj = ScheduleTrajectory()
    for t, p in traj_points:
        point = ScheduleTrajectoryPoint()
        point.pose = p
        point.time_from_start = Duration(seconds=t).to_msg()
        traj.points.append(point)
    return traj
