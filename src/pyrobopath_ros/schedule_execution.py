"""Pyrobopath interfaces for schedule execution in ROS"""

from __future__ import annotations
from typing import Dict, List

# ros
import rospy
import tf2_ros

# pyrobopath
from pyrobopath.process import DependencyGraph, create_dependency_graph_by_z
from pyrobopath.toolpath import Toolpath
from pyrobopath.toolpath_scheduling import (
    MultiAgentToolpathSchedule,
    MultiAgentToolpathPlanner,
    PlanningOptions,
)

# pyrobopath_ros
from pyrobopath_ros.agent_execution_context import AgentExecutionContext
from pyrobopath_ros.utilities import (
    print_schedule_info,
    create_schedule_trajectory,
)

JOINT_STATE_TIMEOUT = 5  # seconds


class ScheduleExecution(object):
    """
    The ScheduleExecution class connects the necessary ROS interfaces to execute
    probopath `ToolpathSchedules` in ROS.

    This object creates and manages an :class:`AgentExecutionContext` for each
    namespace in the list provided by the ros parameter namespace. This class
    acts as the interface between pyrobopath scheduling and cartesian
    trajectory planning and execution.

    ROS Parameters:
        |  `namespaces`: A list of unique namespaces for each robot.
        |  `retract_height`: The distance a robot should move upward between contours
        |  `collision_gap_threshold`: The linear interpolation distance for collision checking
    """

    def __init__(self) -> None:
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        try:
            self._namespaces: List[str] = rospy.get_param("namespaces")  # type: ignore
        except KeyError as e:
            rospy.logerr(f"Missing required parameter: {e}")
            raise e

        self._contexts: Dict[str, AgentExecutionContext] = dict()
        for ns in self._namespaces:  # type: ignore
            self._build_agent_contexts(ns)

        self._schedule: MultiAgentToolpathSchedule | None = None

        self._initialize_pyrobopath()
        rospy.loginfo("Pyrobopath: ready to plan!")

        rospy.on_shutdown(self._shutdown)

    @property
    def agent_models(self):
        return {id: context.agent_model for id, context in self._contexts.items()}

    def _initialize_pyrobopath(self):
        """Initialize the pyrobopath toolpath planner and planning options
        from ROS parameters
        """
        retract_height = rospy.get_param("retract_height", 0.0)
        collision_offset = rospy.get_param("collision_offset", 1.0)
        collision_gap_threshold = rospy.get_param("collision_gap_treshold", 0.003)

        self._planner = MultiAgentToolpathPlanner(self.agent_models)
        self._options = PlanningOptions(
            retract_height=retract_height,
            collision_offset=collision_offset,
            collision_gap_threshold=collision_gap_threshold,
        )

    def _build_agent_contexts(self, id: str):
        """Build an AgentExecutionContext with a unique id and initialize
        the agent's parameters

        :param id: Unique id for agent.
        :type id: str
        """
        self._contexts[id] = AgentExecutionContext(id)
        self._contexts[id].initialize(self.tf_buffer)

    def move_home(self, tf=2.0):
        """Moves all agents to the joint positions in the `/{ns}/home_position`
        parameter.
        """
        for context in self._contexts.values():
            context.move_home(tf)

        for context in self._contexts.values():
            context.joint_execution_client.wait_for_result()

    def schedule_toolpath(
        self, toolpath: Toolpath, dependency_graph: DependencyGraph | None = None
    ):
        """Finds the schedule for the provided toolpath and performs
        Cartesian motion planning on the resulting schedule.

        If no dependency graph is provided, a default all-to-all dependency
        graph is created between the layers in the toolpath. The resulting plan
        is stored internally.

        :param toolpath: The pyrobopath toolpath
        :type toolpath: Toolpath
        :param dependency_graph: an optional dependency graph, defaults to None
        :type dependency_graph: DependencyGraph, optional
        """

        if dependency_graph is None:
            dependency_graph = create_dependency_graph_by_z(toolpath)

        for context in self._contexts.values():
            context.update_tf(self.tf_buffer)

        # Schedule multi-agent toolpath
        rospy.loginfo(f"\n{(50 * '#')}\nScheduling Toolpath:\n{(50 * '#')}\n")
        self._schedule = self._planner.plan(toolpath, dependency_graph, self._options)
        rospy.loginfo(f"\n{(50 * '#')}\nFound Toolpath Plan!\n{(50 * '#')}\n")
        print_schedule_info(self._schedule)
        print()

    def execute_schedule(self):
        """
        Executes the multi-agent toolpath schedule.

        This function processes a precomputed schedule by planning Cartesian
        motions for scheduled events and sending the motion plans to the
        respective execution clients for each agent. The function ensures that
        all agents complete execution before reporting the final execution
        time.

        :raises Warning: If the schedule is empty, execution is aborted with a
        warning.
        """
        if self._schedule is None:
            rospy.logwarn("Cannot execute schedule. Schedule is empty.")
            return

        rospy.loginfo(f"\n{(50 * '#')}\nExecuting Schedule\n{(50 * '#')}\n")

        # convert schedules to trajectories
        trajectory_buffer = dict()
        for id, context in self._contexts.items():
            trajectory_buffer[id] = create_schedule_trajectory(
                self._schedule.schedules[id], context.eef_rotation, context.task_to_base
            )

        start_time = rospy.get_time()
        for id in self._contexts.keys():
            self._contexts[id].execute_trajectory(trajectory_buffer[id])

        for context in self._contexts.values():
            context.schedule_execution_client.wait_for_result()

        end_time = rospy.get_time()
        rospy.loginfo(f"\n{(50 * '#')}\nSchedule Execution Succeeded\n{(50 * '#')}\n")
        rospy.loginfo(f"Start time: {start_time} sec")
        rospy.loginfo(f"End time: {end_time} sec")
        rospy.loginfo(f"Elapsed: {end_time - start_time} sec\n")

    def _shutdown(self):
        rospy.logwarn("Received shutdown request. Cancelling all active goals")
        for context in self._contexts.values():
            context.shutdown()
