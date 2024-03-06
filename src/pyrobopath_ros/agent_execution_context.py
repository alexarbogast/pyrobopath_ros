from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
import quaternion as quat
from typing import Hashable

# ros
import rospy
import actionlib
import tf2_ros
from geometry_msgs.msg import Pose, Transform
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from cartesian_planning_msgs.srv import *

# pyrobopath
from pyrobopath.collision_detection.fcl_collision_models import FCLCollisionModel
from pyrobopath.tools.linalg import SE3
from pyrobopath.tools.types import NDArray
from pyrobopath.collision_detection import FCLRobotBBCollisionModel
from pyrobopath.toolpath_scheduling import AgentModel


def transform_tf_to_np(transform):
    """Returns a 4x4 numpy homogeneous transformation from a provided tf frame

    :param transform: the tf transform to convert
    :type transform: :class:`geometry_msgs.msg.TransformStamped`
    :return: a 4x4 homogeneous transformation matrix
    :rtype: np.ndarray
    """
    p = transform.translation
    q = transform.rotation

    p = np.array([p.x, p.y, p.z])
    q = quat.quaternion(q.w, q.x, q.y, q.z)

    np_tf = np.identity(4, dtype=np.float64)
    np_tf[:3, :3] = quat.as_rotation_matrix(q)
    np_tf[:3, 3] = p
    return np_tf


def transform_to_pose(transform_tf: Transform) -> SE3:
    p = transform_tf.translation
    q = transform_tf.rotation
    se3 = SE3.Quaternion(q.w, q.x, q.y, q.z)
    se3.t = np.array([p.x, p.y, p.z])
    return se3


def _read_ros_param_values(params: Dict[str, Any]) -> Dict[str, Any]:
    for key, default in params.items():
        if default:
            params[key] = rospy.get_param(key, default)
        else:
            params[key] = rospy.get_param(key)
    return params


class AgentExecutionContext(object):
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
        self.eef_to_task: SE3 = SE3()
        self.task_to_base: SE3 = SE3()
        self.base_to_task: SE3 = SE3()
        self.eef_rotation = quat.quaternion()

        self.planning_client = rospy.ServiceProxy(
            f"{self.id}/cartesian_planning_server/plan_cartesian_trajectory",
            PlanCartesianTrajectory,
        )
        self.execution_client = actionlib.SimpleActionClient(
            f"{self.id}/position_trajectory_controller/follow_joint_trajectory",
            FollowJointTrajectoryAction,
        )

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

            eef_rotation = np.array(
                rospy.get_param(f"{self.id}/eef_rotation", [1.0, 0.0, 0.0, 0.0])
            )
            self.eef_rotation = quat.quaternion(
                eef_rotation[0], eef_rotation[1], eef_rotation[2], eef_rotation[3]
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
            x=col_dim[0], y=col_dim[1], z=col_dim[2], anchor=self.eef_to_task.t
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
                self.task_frame, self.base_frame, rospy.Time()
            )
            task_to_base = tf_buffer.lookup_transform(
                self.base_frame, self.task_frame, rospy.Time()
            )
            eef_to_task = tf_buffer.lookup_transform(
                self.task_frame, self.eef_frame, rospy.Time()
            )
            self.eef_to_task = transform_to_pose(eef_to_task.transform)
            self.task_to_base = transform_to_pose(task_to_base.transform)
            self.base_to_task = transform_to_pose(base_to_task.transform)
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

    def shutdown(self):
        self.execution_client.cancel_all_goals()

    def create_pose(self, point: np.ndarray):
        """Create a pose from a given point that aligns the robot configuration
        with that required from an FCLRobotBBCollisionModel

        The default pose is a frame that is initially coincident with the robot's
        base frame. This frame is rotated about the vertical z-axis until the
        x-axis aims towards the end effector. Then the frame is translated to
        `point`.

        :param point: A 3D point from which to make the pose
        :type point: np.ndarray
        :return: A pose from the provided point
        :rtype: geometry_msgs.msg.Pose
        """
        pose = Pose()
        pose.position.x = point[0]
        pose.position.y = point[1]
        pose.position.z = point[2]

        theta = np.arctan2(point[1], point[0])
        q = quat.quaternion(np.cos(theta / 2), 0.0, 0.0, np.sin(theta / 2))
        rot = q * self.eef_rotation

        pose.orientation.w = rot.w
        pose.orientation.x = rot.x
        pose.orientation.y = rot.y
        pose.orientation.z = rot.z
        return pose
