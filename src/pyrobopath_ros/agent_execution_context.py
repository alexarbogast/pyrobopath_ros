from __future__ import annotations
from typing import List, Hashable
import sys
import numpy as np

# ros
import rospy
import actionlib
import tf2_ros
import geometry_msgs.msg as gm
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from controller_manager_msgs.srv import SwitchController, SwitchControllerRequest

# pyrobopath
from pyrobopath.process import AgentModel
from pyrobopath.toolpath.path import Transform, Rotation
from pyrobopath.collision_detection import FCLRobotBBCollisionModel

# pyrobopath_ros
from pyrobopath_ros.msg import (
    ScheduleTrajectory,
    FollowScheduleTrajectoryAction,
    FollowScheduleTrajectoryGoal,
)

TF_TIMEOUT = rospy.Duration(5)  # seconds

JOINT_SPACE_CONTROLLER = "position_trajectory_controller"
TASK_SPACE_CONTROLLER = "pose_controller"


def tf_to_transform(transform_tf: gm.Transform) -> Transform:
    p = transform_tf.translation
    q = transform_tf.rotation
    pose = Transform([p.x, p.y, p.z], [q.w, q.x, q.y, q.z])
    return pose


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


class AgentExecutionContext:
    """All items related to a single agents execution. The context is composed
    of the following components:

    PlanCartesianTrajectory client:
        This service client finds joint trajectories from Cartesian schedules
        found with pyrobopath.

    FollowJointTrajectory client:
        This action client executes joint trajectories found from the Cartesian
        planning server.

    This class also stores a copy of the AgentModel that is created from
    parameters on the ROS parameter server.

    :param id: A unique ID for the agent
    :type id: Hashable
    :param tf_buffer: A reference to the tf2 buffer for locating agent frames
    :type tf_buffer: tf2_ros.Buffer
    """

    def __init__(self, id: Hashable):
        self.id = id
        self._agent_model: AgentModel | None = None
        self.base_frame: str = ""
        self.eef_frame: str = ""
        self.task_frame: str = ""
        self.joint_home: List[float] = []
        self.eef_to_task: Transform = Transform()
        self.task_to_base: Transform = Transform()
        self.base_to_task: Transform = Transform()
        self.eef_rotation: Rotation = Rotation()

        self.joint_execution_client = actionlib.SimpleActionClient(
            f"{self.id}/position_trajectory_controller/follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )

        self.schedule_execution_client = actionlib.SimpleActionClient(
            f"{self.id}/follow_schedule_trajectory", FollowScheduleTrajectoryAction
        )
        self.controller_manager_client = ControllerManagerClient(self.id)

        self.joint_execution_client.wait_for_server()
        self.schedule_execution_client.wait_for_server()

    def initialize(self, tf_buffer: tf2_ros.Buffer):
        """Initialize the context with values from the ROS parameter server

        Create an :class:`pyrobopath.toolpath_scheduling.AgentModel` from
        from ROS parameter server values
        """
        try:
            capabilities = rospy.get_param(f"{self.id}/capabilities", [0])

            self.base_frame: str = str(rospy.get_param(f"{self.id}/base_frame"))
            self.eef_frame: str = str(rospy.get_param(f"{self.id}/eef_frame"))
            self.task_frame: str = str(rospy.get_param(f"{self.id}/task_frame"))
            self.joint_home = rospy.get_param(f"{self.id}/home_position")

            col_dim = (
                rospy.get_param(f"{self.id}/collision/length"),
                rospy.get_param(f"{self.id}/collision/width"),
                rospy.get_param(f"{self.id}/collision/height"),
            )
            col_offset = rospy.get_param(f"{self.id}/collision/offset", np.zeros(3))

            self.eef_rotation = Rotation(
                rospy.get_param(f"{self.id}/eef_rotation", [1.0, 0.0, 0.0, 0.0])
            )

            velocity = rospy.get_param(f"{self.id}/velocity")
            travel_velocity = rospy.get_param(f"{self.id}/travel_velocity", velocity)

        except KeyError as e:
            rospy.logerr(f"Missing required parameter: {e} in namespace {self.id}")
            sys.exit()

        # only update parameters if the model has been previously created
        self.update_tf(tf_buffer, self._agent_model is not None)

        # build collision model
        collision_model = FCLRobotBBCollisionModel(
            col_dim, anchor=self.eef_to_task.t, offset=col_offset
        )

        # build agent model
        self._agent_model = AgentModel(
            capabilities=capabilities,
            collision_model=collision_model,
            base_frame_position=self.eef_to_task.t,
            home_position=self.base_to_task.t,
            velocity=velocity,
            travel_velocity=travel_velocity,
        )

    @property
    def agent_model(self):
        return self._agent_model

    def update_tf(self, tf_buffer: tf2_ros.Buffer, sync_agent_model=True):
        """Updates the values of the task frame, base frame, and end effector
        frame from the tf2 buffer

        :param tf_buffer: the buffer from which to update the frame data
        :type tf_buffer: :class:`tf2_ros.Buffer`
        """
        try:
            base_to_task = tf_buffer.lookup_transform(
                self.task_frame, self.base_frame, rospy.Time(), timeout=TF_TIMEOUT
            )
            task_to_base = tf_buffer.lookup_transform(
                self.base_frame, self.task_frame, rospy.Time(), timeout=TF_TIMEOUT
            )
            eef_to_task = tf_buffer.lookup_transform(
                self.task_frame, self.eef_frame, rospy.Time(), timeout=TF_TIMEOUT
            )
            self.eef_to_task = tf_to_transform(eef_to_task.transform)
            self.task_to_base = tf_to_transform(task_to_base.transform)
            self.base_to_task = tf_to_transform(base_to_task.transform)
        except:
            rospy.logfatal(f"Failed to find transforms for agent {self.id}")

        if sync_agent_model:
            self.sync_agent_model()

    def sync_agent_model(self):
        """Syncronize the home and base frame positions between the
        agent model and the last update tf frames
        """
        self._agent_model.home_position = self.eef_to_task.t
        self._agent_model.base_frame_position = self.base_to_task.t
        if isinstance(self._agent_model.collision_model, FCLRobotBBCollisionModel):
            self._agent_model.collision_model.anchor = (
                self._agent_model.base_frame_position
            )

    def move_home(self, tf=2.0):
        """Moves agents to the joint positions in the `/{ns}/home_position`
        parameter.
        """
        self.start_joint_control()
        start_state = rospy.wait_for_message(f"/{self.id}/joint_states", JointState)

        point = JointTrajectoryPoint()
        point.positions = self.joint_home
        point.velocities = [0] * len(point.positions)
        point.accelerations = [0] * len(point.positions)
        point.time_from_start = rospy.Duration.from_sec(tf)

        goal = FollowJointTrajectoryGoal()
        goal.trajectory.joint_names = start_state.name
        goal.trajectory.points = [point]
        self.joint_execution_client.send_goal(goal)

    def execute_trajectory(self, traj: ScheduleTrajectory):
        self.start_taskspace_control()
        goal = FollowScheduleTrajectoryGoal(trajectory=traj)
        rospy.loginfo(f"Robot {self.id}: Sending trajectory goal")
        self.schedule_execution_client.send_goal(goal)

    def shutdown(self):
        self.joint_execution_client.cancel_all_goals()

    def start_joint_control(self):
        self.controller_manager_client.switch_controller(
            [JOINT_SPACE_CONTROLLER], [TASK_SPACE_CONTROLLER]
        )

    def start_taskspace_control(self):
        self.controller_manager_client.switch_controller(
            [TASK_SPACE_CONTROLLER], [JOINT_SPACE_CONTROLLER]
        )
