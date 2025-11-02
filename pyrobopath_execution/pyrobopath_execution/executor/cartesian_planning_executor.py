from collections import defaultdict

# ros
import rclpy
from rclpy.time import Time
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.task import Future
from rclpy.wait_for_message import wait_for_message

from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory
from cartesian_planning_msgs.msg import ErrorCodes
from cartesian_planning_msgs.srv import PlanCartesianTrajectory

# pyrobopath
from pyrobopath.toolpath_scheduling import (
    MultiAgentToolpathSchedule,
    MoveEvent,
    ContourEvent,
)

# pyrobopath_ros
from pyrobopath_execution.agent_context import AgentContext
from pyrobopath_execution.utilities import (
    offset_trajectory_times,
    compile_schedule_plans,
    create_pose,
)
from .executor_base import Executor

JOINT_STATE_TIMEOUT = 5  # seconds


class CartesianPlanningExecutor(Executor):
    MotionPlan = list[tuple[float, FollowJointTrajectory.Goal]]
    MotionPlanBuffer = dict[str, MotionPlan]

    def __init__(self, node: Node, contexts: dict[str, AgentContext]):
        super().__init__(node, contexts)

        self._clients = {
            id: AgentExecutionClient(node, id, context)
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
        t_start = Time(
            seconds=self._node.get_clock().now().nanoseconds * 1e-9
            + traj_preprocess_offset
        )
        for agent, plan in compiled_plans.items():
            plan.trajectory.header.stamp = t_start.to_msg()
            self._clients[agent].execute_trajectory(plan)

        for client in self._clients.values():
            client.wait_for_all_results()

    def move_home(self, tf=2.0):
        for client in self._clients.values():
            client.move_home(tf)

        for client in self._clients.values():
            client.wait_for_all_results()

    def shutdown(self):
        for client in self._clients.values():
            client.shutdown()

    def _plan_multi_agent_schedule(self, schedule: MultiAgentToolpathSchedule):
        self._node.get_logger().info("Planning events in multi-agent schedule")
        for agent, sched in schedule.schedules.items():
            start_state = self._clients[agent].get_joint_state()

            for event in sched._events:
                resp = self._plan_event(event, agent, start_state)  # type: ignore
                if resp.error_code.val == ErrorCodes.SUCCESS:
                    if not resp.trajectory.points:
                        self._node.get_logger().warn(
                            "Planning service returned with empty trajectory"
                        )
                        continue

                    # create trajectory action server goal
                    goal = FollowJointTrajectory.Goal()
                    goal.trajectory = resp.trajectory
                    self._schedule_plan_buffer[agent].append((event.start, goal))

                    start_state.position = resp.trajectory.points[-1].positions
                    start_state.velocity = resp.trajectory.points[-1].velocities
                else:
                    self._node.get_logger().error(
                        "Failed to plan Cartesian trajectory. "
                        + "Planning service returned with ERROR_CODE: "
                        + str(resp.error_code.val)
                    )
                    return False
        self._node.get_logger().info(
            "Motion planning succeeded for all events in schedule"
        )
        return True

    def _plan_event(self, event: MoveEvent, agent, start_state: JointState):
        context = self._contexts[agent]
        client = self._clients[agent].planning_client

        # Transform task-space path to base frame
        path_base = [context.task_to_base * p for p in event.data]

        req = PlanCartesianTrajectory.Request()
        req.start_state = start_state
        req.path = [create_pose(p, context.eef_rotation) for p in path_base]
        req.max_angular_velocity = 1.0
        req.scaling = PlanCartesianTrajectory.Request.SCALING_FIRST

        if isinstance(event, ContourEvent):
            req.max_linear_velocity = context.agent_model.velocity
        else:
            req.max_linear_velocity = context.agent_model.travel_velocity

        future = client.call_async(req)
        rclpy.spin_until_future_complete(self._node, future)
        if not future.done() or future.result() is None:
            msg = "Planning service call failed"
            self._node.get_logger().error(msg)
            raise RuntimeError(msg)

        response = future.result()
        if response.error_code.val != ErrorCodes.SUCCESS:
            msg = (
                "Failed to plan Cartesian trajectory. "
                f"Error code: {response.error_code.val}"
            )
            self._node.get_logger().error(msg)
            raise RuntimeError(msg)

        return response


class AgentExecutionClient:
    def __init__(self, node: Node, id: str, context: AgentContext):
        self.node = node
        self.id = id
        self.context = context

        controller_param_name = f"{self.id}.controller"
        joint_param_name = f"{self.id}.joints"
        joint_states_topic_param_name = f"{self.id}.joint_states_topic"

        # read parameters
        self.node.declare_parameter(controller_param_name, "")
        controller = self.node.get_parameter(controller_param_name).value
        if controller == "":
            raise RuntimeError(f"Missing required parameter: {controller_param_name}")

        self.node.declare_parameter(joint_param_name, [""])
        self.joints: list[str] = self.node.get_parameter(joint_param_name).value
        if self.joints == [""]:
            raise RuntimeError(f"Missing required parameter: {joint_param_name}")

        self.node.declare_parameter(joint_states_topic_param_name, "")
        self.joint_states_topic: str = self.node.get_parameter(
            joint_states_topic_param_name
        ).value
        if self.joint_states_topic == "":
            self.node.get_logger().info(
                f"Parameter '{joint_states_topic_param_name}' was not provided. "
                "Defaulting to '/joint_states'"
            )
            self.joint_states_topic = "/joint_states"

        # action client: <controller>/follow_joint_trajectory
        action_name = f"{controller}/follow_joint_trajectory"
        self.joint_execution_client = ActionClient(
            node, FollowJointTrajectory, action_name
        )

        # service client: /<id>/cartesian_planning_server/plan_cartesian_trajectory
        srv_name = f"{self.id}/cartesian_planning_server/plan_cartesian_trajectory"
        self.planning_client = node.create_client(PlanCartesianTrajectory, srv_name)

        self._pending_goal_futures: list[Future] = []

        while not self.joint_execution_client.wait_for_server(timeout_sec=0.5):
            if not rclpy.ok():
                self.node.get_logger().error(
                    "ROS Shutdown before action became available"
                )
                return
            self.node.get_logger().info(f"Waiting for action server: {action_name}...")
        self.node.get_logger().info(f"Connected action client {action_name}")

        while not self.planning_client.wait_for_service(timeout_sec=0.5):
            if not rclpy.ok():
                self.node.get_logger().error(
                    "ROS Shutdown before service became available"
                )
                return
            self.node.get_logger().info(f"Waiting for service server: {srv_name}...")
        self.node.get_logger().info(f"Connected server client {srv_name}")

        self.node.get_logger().info(f"Setup agent execution client for {self.id}")

    def execute_trajectory(self, traj: FollowJointTrajectory.Goal):
        self.node.get_logger().info(f"Robot {self.id}: Sending trajectory goal")
        send_goal_future = self.joint_execution_client.send_goal_async(traj)

        # Wait for goal handle
        rclpy.spin_until_future_complete(self.node, send_goal_future, timeout_sec=5.0)
        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.node.get_logger().error("FollowJointTrajectory: goal rejected")
            return

        result_future = goal_handle.get_result_async()
        self._pending_goal_futures.append(result_future)

    def wait_for_all_results(self):
        # wait for all stored result futures
        for fut in list(self._pending_goal_futures):
            rclpy.spin_until_future_complete(self.node, fut)

        self._pending_goal_futures.clear()

    def move_home(self, tf=2.0):
        """Moves agents to the joint positions in the `/{ns}/home_position`
        parameter.
        """
        start_state = self.get_joint_state()

        point = JointTrajectoryPoint()
        point.positions = list(self.context.joint_home)
        point.velocities = [0.0] * len(point.positions)
        point.accelerations = [0.0] * len(point.positions)
        point.time_from_start = Duration(seconds=tf).to_msg()

        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = start_state.name
        goal.trajectory.points = [point]

        self.execute_trajectory(goal)

    def get_joint_state(self) -> JointState:
        _, state = wait_for_message(
            JointState,
            self.node,
            self.joint_states_topic,
            time_to_wait=JOINT_STATE_TIMEOUT,
        )

        if state is None:
            msg = (
                "Timed out waiting for JointState on topic: " + self.joint_states_topic
            )
            self.node.get_logger().error(msg)
            raise RuntimeError(msg)

        # filter joint state
        index_map = [state.name.index(j) for j in self.joints]

        filtered = JointState()
        filtered.header = state.header
        filtered.name = self.joints
        filtered.position = [state.position[j] for j in index_map]
        filtered.velocity = [state.velocity[j] for j in index_map]
        filtered.effort = [state.effort[j] for j in index_map]
        return filtered

    def shutdown(self):
        # cancel goals if action server provides cancellation
        for fut in self._pending_goal_futures:
            fut.cancel()
        self._pending_goal_futures.clear()
