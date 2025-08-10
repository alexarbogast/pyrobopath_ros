from typing import Dict

# ros
import rospy
import actionlib
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest

# pyrobopath
from pyrobopath.toolpath_scheduling import MultiAgentToolpathSchedule

# pyrobopath_ros
from pyrobopath_ros.msg import (
    ScheduleTrajectory,
    FollowScheduleTrajectoryAction,
    FollowScheduleTrajectoryGoal,
)
from pyrobopath_ros.agent_context import AgentContext
from pyrobopath_ros.utilities import create_schedule_trajectory
from .executor_base import Executor

# TODO: Read values through parameters
JOINT_SPACE_CONTROLLER = "position_trajectory_controller"
TASK_SPACE_CONTROLLER = "pose_controller"


class TaskspaceControlExecutor(Executor):
    def __init__(self, contexts: Dict[str, AgentContext]):
        super(TaskspaceControlExecutor, self).__init__(contexts)
        self._clients = {
            id: AgentExecutionClient(id, context)
            for id, context in self._contexts.items()
        }

    def execute(self, sched: MultiAgentToolpathSchedule):
        # convert schedules top trajectories
        trajectory_buffer = dict()
        for id, context in self._contexts.items():
            trajectory_buffer[id] = create_schedule_trajectory(
                sched.schedules[id], context.eef_rotation, context.task_to_base
            )

        for id in self._clients.keys():
            self._clients[id].execute_trajectory(trajectory_buffer[id])

        for client in self._clients.values():
            client.schedule_execution_client.wait_for_result()

    def move_home(self, tf=2.0):
        for client in self._clients.values():
            client.move_home(tf)

        for client in self._clients.values():
            client.joint_execution_client.wait_for_result()

    def shutdown(self):
        for client in self._clients.values():
            client.shutdown()


class AgentExecutionClient:
    def __init__(self, id: str, context: AgentContext):
        self.id = id
        self.context = context

        self.joint_execution_client = actionlib.SimpleActionClient(
            f"{self.id}/{JOINT_SPACE_CONTROLLER}/follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )
        self.schedule_execution_client = actionlib.SimpleActionClient(
            f"{self.id}/follow_schedule_trajectory", FollowScheduleTrajectoryAction
        )
        self.controller_manager_client = ControllerManagerClient(self.id)

        self.joint_execution_client.wait_for_server()
        self.schedule_execution_client.wait_for_server()

    def execute_trajectory(self, traj: ScheduleTrajectory):
        self._start_taskspace_control()
        goal = FollowScheduleTrajectoryGoal(trajectory=traj)
        rospy.loginfo(f"Robot {self.id}: Sending trajectory goal")
        self.schedule_execution_client.send_goal(goal)

    def move_home(self, tf=2.0):
        """Moves agents to the joint positions in the `/{ns}/home_position`
        parameter.
        """
        self._start_joint_control()
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
        self.schedule_execution_client.cancel_all_goals()

    def _start_joint_control(self):
        self.controller_manager_client.switch_controller(
            [JOINT_SPACE_CONTROLLER], [TASK_SPACE_CONTROLLER]
        )

    def _start_taskspace_control(self):
        self.controller_manager_client.switch_controller(
            [TASK_SPACE_CONTROLLER], [JOINT_SPACE_CONTROLLER]
        )


class ControllerManagerClient:
    def __init__(self, id):
        self.id = id
        rospy.wait_for_service(f"{self.id}/controller_manager/switch_controller")
        self.switch_controller_client = rospy.ServiceProxy(
            f"{self.id}/controller_manager/switch_controller", SwitchController
        )

    def switch_controller(self, start_controllers, stop_controllers):
        try:
            req = SwitchControllerRequest()
            req.start_controllers = start_controllers
            req.stop_controllers = stop_controllers
            req.strictness = 1
            self.switch_controller_client.call(req)
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
