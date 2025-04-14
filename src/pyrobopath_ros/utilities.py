from typing import List
from gcodeparser import GcodeParser

# ros
import rospy
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

# pyrobopath
from pyrobopath.toolpath import Toolpath
from pyrobopath.toolpath_scheduling import MultiAgentToolpathSchedule

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
    goal.trajectory = plans[0].trajectory

    for p in plans[1:]:
        # occasionally the accumlated error in trajectory times is nano-seconds
        # in the past. Replace the time with the value from the schedule if it's
        # below the threshold
        t1_end = goal.trajectory.points[-1].time_from_start
        t2_start = p.trajectory.points[0].time_from_start
        diff = t1_end.to_sec() - t2_start.to_sec()

        if abs(diff) < MAX_BACKWARDS_TIME:
            goal.trajectory.points.extend(p.trajectory.points[1:])
        else:
            goal.trajectory.points.extend(p.trajectory.points)

    return goal
