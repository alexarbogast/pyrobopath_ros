from collections import defaultdict
from typing import Dict, List, Tuple, Hashable

# ros
import rospy
import actionlib
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from cartesian_planning_msgs.msg import ErrorCodes
from cartesian_planning_msgs.srv import (
    PlanCartesianTrajectory,
    PlanCartesianTrajectoryResponse,
    PlanCartesianTrajectoryRequest,
)

# pyrobopath
from pyrobopath.toolpath_scheduling import (
    MultiAgentToolpathSchedule,
    MoveEvent,
    ContourEvent,
)

# pyrobopath_ros
from pyrobopath_ros.agent_context import AgentContext
from pyrobopath_ros.utilities import (
    offset_trajectory_times,
    compile_schedule_plans,
    create_pose,
)
from .executor_base import Executor

JOINT_STATE_TIMEOUT = 5  # seconds

# TODO: Read values through parameters
JOINT_SPACE_CONTROLLER = "position_trajectory_controller"


class CartesianPlanningExecutor(Executor):
    MotionPlan = List[Tuple[float, FollowJointTrajectoryGoal]]
    MotionPlanBuffer = Dict[str, MotionPlan]

    def __init__(self, contexts: Dict[str, AgentContext]):
        super(CartesianPlanningExecutor, self).__init__(contexts)
        self._clients = {
            id: AgentExecutionClient(id, context)
            for id, context in self._contexts.items()
        }

        self._schedule_plan_buffer: CartesianPlanningExecutor.MotionPlanBuffer = (
            defaultdict(list)
        )

    def execute(self, sched: MultiAgentToolpathSchedule):
        self._schedule_plan_buffer.clear()
        if not self._plan_multi_agent_schedule(sched):
            return

        # Compile plans into single trajectory goal
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
            self._clients[agent].execute_trajectory(plan)

        for client in self._clients.values():
            client.joint_execution_client.wait_for_result()

    def move_home(self, tf=2.0):
        for client in self._clients.values():
            client.move_home(tf)

        for client in self._clients.values():
            client.joint_execution_client.wait_for_result()

    def shutdown(self):
        for client in self._clients.values():
            client.shutdown()

    def _plan_multi_agent_schedule(self, schedule: MultiAgentToolpathSchedule):
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
        context = self._contexts[agent]
        path_base = [context.task_to_base * p for p in event.data]

        req = PlanCartesianTrajectoryRequest()
        req.start_state = start_state
        req.path = [create_pose(p, context.eef_rotation) for p in path_base]

        if isinstance(event, ContourEvent):
            req.max_linear_velocity = context.agent_model.velocity
            req.max_angular_velocity = 1.0
            req.scaling = PlanCartesianTrajectoryRequest.SCALING_FIRST
        else:
            req.max_linear_velocity = context.agent_model.travel_velocity
            req.max_angular_velocity = 1.0
            req.scaling = PlanCartesianTrajectoryRequest.SCALING_FIRST

        resp = PlanCartesianTrajectoryResponse()
        try:
            resp = self._clients[agent].planning_client(req)
        except rospy.ServiceException as e:
            rospy.logerr(f"Cartesian planning service failed with exception: {e}")
        return resp


class AgentExecutionClient:
    def __init__(self, id: Hashable, context: AgentContext):
        self.id = id
        self.context = context

        self.joint_execution_client = actionlib.SimpleActionClient(
            f"{self.id}/{JOINT_SPACE_CONTROLLER}/follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )

        self.planning_client = rospy.ServiceProxy(
            f"{self.id}/cartesian_planning_server/plan_cartesian_trajectory",
            PlanCartesianTrajectory,
        )

        self.joint_execution_client.wait_for_server()

    def execute_trajectory(self, traj: FollowJointTrajectoryGoal):
        rospy.loginfo(f"Robot {self.id}: Sending trajectory goal")
        self.joint_execution_client.send_goal(traj)

    def move_home(self, tf=2.0):
        """Moves agents to the joint positions in the `/{ns}/home_position`
        parameter.
        """
        start_state = rospy.wait_for_message(f"/{self.id}/joint_states", JointState)

        point = JointTrajectoryPoint()
        point.positions = self.context.joint_home
        point.velocities = [0] * len(point.positions)
        point.accelerations = [0] * len(point.positions)
        point.time_from_start = rospy.Duration.from_sec(tf)

        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = start_state.name
        goal.trajectory.points = [point]
        self.joint_execution_client.send_goal(goal)

    def shutdown(self):
        self.joint_execution_client.cancel_all_goals()
