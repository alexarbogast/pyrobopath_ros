"""Pyrobopath interfaces for schedule execution in ROS"""

from __future__ import annotations
from collections import defaultdict
from typing import Dict, List, Tuple

# ros
import rospy
import tf2_ros
from sensor_msgs.msg import JointState
from control_msgs.msg import FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint
from actionlib_msgs.msg import GoalStatus
from cartesian_planning_msgs.msg import ErrorCodes
from cartesian_planning_msgs.srv import (
    PlanCartesianTrajectoryResponse,
    PlanCartesianTrajectoryRequest,
)

# pyrobopath
from pyrobopath.process import DependencyGraph, create_dependency_graph_by_z
from pyrobopath.toolpath import Toolpath
from pyrobopath.toolpath_scheduling import (
    MultiAgentToolpathSchedule,
    MultiAgentToolpathPlanner,
    PlanningOptions,
    MoveEvent,
    ContourEvent,
)

# pyrobopath_ros
from pyrobopath_ros.agent_execution_context import AgentExecutionContext
from pyrobopath_ros.utilities import (
    print_schedule_info,
    offset_trajectory_times,
    compile_schedule_plans,
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

    MotionPlan = List[Tuple[float, FollowJointTrajectoryGoal]]
    MotionPlanBuffer = Dict[str, MotionPlan]

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

        rospy.loginfo("Waiting for plan_cartesian_trajectory servers...")
        for context in self._contexts.values():
            context.planning_client.wait_for_service()

        rospy.loginfo("Waiting for follow_trajectory_action servers...")
        for context in self._contexts.values():
            context.execution_client.wait_for_server()

        self._schedule: MultiAgentToolpathSchedule | None = None
        self._schedule_plan_buffer: ScheduleExecution.MotionPlanBuffer = defaultdict(
            list
        )

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
        # send trajectories to home
        for id in self._contexts.keys():
            start_state = rospy.wait_for_message(f"/{id}/joint_states", JointState)

            point = JointTrajectoryPoint()
            point.positions = self._contexts[id].joint_home
            point.velocities = [0] * len(point.positions)
            point.accelerations = [0] * len(point.positions)
            point.time_from_start = rospy.Duration.from_sec(tf)

            goal = FollowJointTrajectoryGoal()
            goal.trajectory.joint_names = start_state.name
            goal.trajectory.points = [point]
            self._contexts[id].execution_client.send_goal(goal)

        # wait for completion
        self._wait_for_execution()

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

        # Schedule multi agent toolpath
        rospy.loginfo(f"\n{(50 * '#')}\nScheduling Toolpath:\n{(50 * '#')}\n")
        self._schedule = self._planner.plan(toolpath, dependency_graph, self._options)
        rospy.loginfo(f"\n{(50 * '#')}\nFound Toolpath Plan!\n{(50 * '#')}\n")
        print_schedule_info(self._schedule)
        print()

    def execute_schedule(self):
        """
        Executes the scheduled multi-agent motion plan.

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

        # Cartesian motion planning for scheduled events
        self._schedule_plan_buffer.clear()
        if not self._plan_multi_agent_schedule(self._schedule):
            return

        # Compile plans into single trajectory goal
        start_time = rospy.get_time()
        compiled_plans = dict()
        for agent, plans in self._schedule_plan_buffer.items():
            for start_t, plan in plans:
                offset_trajectory_times(plan.trajectory.points, start_t)
            compiled_plans[agent] = compile_schedule_plans([p for _, p in plans])

        # Send goals to action server
        traj_preprocess_offset = 1.0
        t_start = rospy.get_time() + traj_preprocess_offset
        for agent, plan in compiled_plans.items():
            plan.trajectory.header.stamp = rospy.Time.from_sec(t_start)
            rospy.loginfo(f"[{agent}] queuing motion plan for event")
            self._contexts[agent].execution_client.send_goal(
                plan, done_cb=self._execution_client_done_cb
            )

        self._wait_for_execution()

        end_time = rospy.get_time()
        print()
        rospy.loginfo(f"\n{(50 * '#')}\nSchedule Execution Succeeded\n{(50 * '#')}\n")
        rospy.loginfo(f"Start time: {start_time} sec")
        rospy.loginfo(f"End time: {end_time} sec")
        rospy.loginfo(f"Elapsed: {end_time - start_time} sec\n")

    def _plan_multi_agent_schedule(self, schedule: MultiAgentToolpathSchedule):
        """Populates the schedule plan buffer with motion plans from each
        event in `schedule`.

        :param schedule: The schedule
        :type schedule: MultiAgentToolpathSchedule
        """
        rospy.loginfo("Planning events in multi-agent schedule")
        for agent, sched in schedule.schedules.items():
            joint_state_topic = f"/{agent}/joint_states"
            start_state = JointState()
            try:
                start_state = rospy.wait_for_message(
                    joint_state_topic, JointState, JOINT_STATE_TIMEOUT
                )
            except rospy.ROSException as e:
                rospy.logerr(
                    "Timed out waiting for JointState on topic: " + joint_state_topic
                )
                return False

            for event in sched._events:
                resp = self._plan_event(event, agent, start_state)  # type: ignore
                if resp.error_code.val == ErrorCodes.SUCCESS:
                    if not resp.trajectory.points:
                        rospy.logwarn("Planning service returned with empty trajectory")
                        continue

                    # create trajectory action server goal
                    goal = FollowJointTrajectoryGoal()
                    goal.trajectory = resp.trajectory
                    self._schedule_plan_buffer[agent].append((event.start, goal))

                    start_state.position = resp.trajectory.points[-1].positions
                    start_state.velocity = resp.trajectory.points[-1].velocities
                else:
                    rospy.logerr(
                        "Failed to plan Cartesian trajectory. "
                        + "Planning service returned with ERROR_CODE: "
                        + str(resp.error_code.val)
                    )
                    return False
        rospy.loginfo("Motion planning succeeded for all events in schedule")
        return True

    def _plan_event(self, event: MoveEvent, agent, start_state: JointState):
        """Peforms Cartesian motion planning for a pyrobopath MoveEvent

        :param event: event with cartesian path
        :type event: MoveEvent
        :param agent: the agent to plan for
        :type agent: Hashable
        :param start_state: the starting joint configuration
        :type start_state: JointState

        :return: the response from the cartesian planning server
        :rtype: PlanCartesianTrajectoryResponse
        """
        context = self._contexts[agent]
        path_base = [context.task_to_base * p for p in event.data]

        req = PlanCartesianTrajectoryRequest()
        req.start_state = start_state
        req.path = [context.create_pose(p) for p in path_base]

        if isinstance(event, ContourEvent):
            req.max_linear_velocity = context.agent_model.velocity
            req.max_angular_velocity = 1.0
            req.scaling = PlanCartesianTrajectoryRequest.SCALING_FIRST
            req.max_angular_velocity
        else:
            req.max_linear_velocity = context.agent_model.travel_velocity
            req.max_angular_velocity = 1.0
            req.scaling = PlanCartesianTrajectoryRequest.SCALING_FIRST

        resp = PlanCartesianTrajectoryResponse()
        try:
            resp = context.planning_client(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"Cartesian planning service failed with exception: {e}")
        return resp

    def _wait_for_execution(self):
        for context in self._contexts.values():
            context.execution_client.wait_for_result()

    def _execution_client_done_cb(self, state, result):
        if state == GoalStatus.SUCCEEDED:
            rospy.loginfo("Goal Succeeded!")
        elif state == GoalStatus.ABORTED:
            rospy.loginfo(f"Goal was aborted!. result: {result}")
        elif state == GoalStatus.REJECTED:
            rospy.logerr(f"Goal was rejected. result: {result}")
            self._abort_execution()
        else:
            rospy.loginfo(f"Goal finished with state: {state}, result: {result}")

    def _abort_execution(self):
        rospy.loginfo("Aborting execution")
        for context in self._contexts.values():
            context.execution_client.cancel_all_goals()

    def _shutdown(self):
        rospy.logwarn("Received shutdown request. Cancelling all active goals")
        for context in self._contexts.values():
            context.shutdown()
