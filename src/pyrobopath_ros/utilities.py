from typing import List
from gcodeparser import GcodeParser
from copy import deepcopy
import numpy as np

# ros
import rospy
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from geometry_msgs.msg import Pose

# pyrobopath
from pyrobopath.toolpath import Toolpath
from pyrobopath.toolpath import Rotation, Transform
from pyrobopath.toolpath_scheduling import MultiAgentToolpathSchedule, ToolpathSchedule

# pyrobopath_ros
from pyrobopath_ros.msg import ScheduleTrajectoryPoint, ScheduleTrajectory

MAX_BACKWARDS_TIME = 1e-8


def toolpath_from_gcode(filepath) -> Toolpath:
    """Parse gcode file to internal toolpath representation.

    :param filepath: The absolute path to a Gcode file
    :type filepath: str
    :return: A toolpath created from the input filepath
    :rtype: Toolpath
    """

    with open(filepath, "r") as f:
        gcode = f.read()
    parsed_gcode = GcodeParser(gcode)

    toolpath = Toolpath.from_gcode(parsed_gcode.lines)
    return toolpath


def print_schedule_info(schedule: MultiAgentToolpathSchedule):
    """Print the schedule duration, total number of events,
    and events for each agent

    :param schedule: The schedule to print info
    :type schedule: MultiAgentToolpathSchedule
    """
    print(f"Schedule duration: {schedule.duration()}")
    print(f"Total Events: {schedule.n_events()}")
    agents_info = "Agent Events: "
    for agent, sched in schedule.schedules.items():
        agents_info += f"{agent}: {len(sched._events)}, "
    print(agents_info)


## Trajectory and schedule modification
def offset_trajectory_times(traj: List[JointTrajectoryPoint], offset: float):
    """
    Applies a time offset to each point in a joint trajectory.

    This function adjusts the `time_from_start` of each `JointTrajectoryPoint`
    in the given trajectory by adding the specified offset.

    :param traj: A list of `JointTrajectoryPoint` instances representing the trajectory.
    :type traj: List[JointTrajectoryPoint]
    :param offset: The time offset to apply, in seconds.
    :type offset: float
    """
    for point in traj:
        point.time_from_start += rospy.Duration(offset)


def compile_schedule_plans(
    plans: List[FollowJointTrajectoryGoal],
) -> FollowJointTrajectoryGoal:
    """
    Merges multiple `FollowJointTrajectoryGoal` plans into a single trajectory.

    This function takes a list of `FollowJointTrajectoryGoal` objects and
    combines their trajectories into a single goal. It ensures that consecutive
    plans are smoothly connected by avoiding duplicate waypoints at the
    transition between plans.

    :param plans: A list of `FollowJointTrajectoryGoal` instances to be merged.
                  Assumes that all plans share the same tolerance settings.
    :type plans: List[FollowJointTrajectoryGoal]
    :return: A single `FollowJointTrajectoryGoal` containing the merged trajectory.
    :rtype: FollowJointTrajectoryGoal
    """
    goal = FollowJointTrajectoryGoal()
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
        diff = t1_end.to_sec() - t2_start.to_sec()

        if abs(diff) < MAX_BACKWARDS_TIME:
            goal.trajectory.points.extend(p.trajectory.points[1:])
        else:
            goal.trajectory.points.extend(p.trajectory.points[:])

    return goal


def create_pose(point: np.ndarray, rot_offset: Rotation):
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
    # compile trajectory points
    traj_points = []
    initial_point = sched._events[0].traj[0]
    initial_point_base = transform * initial_point.data
    traj_points.append(
        (initial_point.time, create_pose(initial_point_base, rot_offset))
    )
    for event in sched._events:
        for p in event.traj:
            if abs(p.time - traj_points[-1][0]) < MAX_BACKWARDS_TIME:
                continue
            p_base = transform * p.data
            traj_points.append((p.time, create_pose(p_base, rot_offset)))

    traj = ScheduleTrajectory()
    for t, p in traj_points:
        point = ScheduleTrajectoryPoint()
        point.pose = p
        point.time_from_start = rospy.Duration.from_sec(t)
        traj.points.append(point)
    return traj
