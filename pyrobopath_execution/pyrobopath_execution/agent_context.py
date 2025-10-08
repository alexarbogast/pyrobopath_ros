from __future__ import annotations
from typing import List, Hashable

# ros
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros
import geometry_msgs.msg as gm

# pyrobopath
from pyrobopath.process import AgentModel
from pyrobopath.toolpath.path import Transform, Rotation
from pyrobopath.collision_detection import FCLRobotBBCollisionModel

TF_TIMEOUT = Duration(seconds=0.5)


def tf_to_transform(transform_tf: gm.Transform) -> Transform:
    p = transform_tf.translation
    q = transform_tf.rotation
    pose = Transform([p.x, p.y, p.z], [q.w, q.x, q.y, q.z])
    return pose


class AgentContext:
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

    schema = {
        "capabilities": {"default": [0], "required": False},
        "base_frame": {"default": "", "required": True},
        "eef_frame": {"default": "", "required": True},
        "task_frame": {"default": "", "required": True},
        "home_position": {"default": [0.0], "required": True},
        "collision.length": {"default": 0.0, "required": True},
        "collision.width": {"default": 0.0, "required": True},
        "collision.height": {"default": 0.0, "required": True},
        "collision.offset": {"default": [0.0, 0.0, 0.0], "required": False},
        "eef_rotation": {"default": [1.0, 0.0, 0.0, 0.0], "required": False},
        "velocity": {"default": 0.0, "required": True},
        "travel_velocity": {"default": -1.0, "required": False},
    }

    def __init__(self, id: Hashable, node: Node):
        self.node = node
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

    def initialize(self, tf_buffer: tf2_ros.Buffer):
        """Initialize the context with values from the ROS parameter server

        Create an :class:`pyrobopath.toolpath_scheduling.AgentModel` from
        from ROS parameter server values
        """
        params = {}
        for name, spec in AgentContext.schema.items():
            full_name = f"{self.id}.{name}"
            self.node.declare_parameter(full_name, spec["default"])
            value = self.node.get_parameter(full_name).value
            if spec["required"] and (value is None or value == "" or value == []):
                raise RuntimeError(f"Missing required parameter: {full_name}")
            params[name] = value

        capabilities = params["capabilities"]
        self.base_frame = params["base_frame"]
        self.eef_frame = params["eef_frame"]
        self.task_frame = params["task_frame"]
        self.joint_home = params["home_position"]

        col_dim = (
            params["collision.length"],
            params["collision.width"],
            params["collision.height"],
        )
        col_offset = params["collision.offset"]

        self.eef_rotation = Rotation(params[f"eef_rotation"])

        velocity = params["velocity"]
        travel_velocity = params["travel_velocity"]
        if travel_velocity == -1.0:
            travel_velocity = velocity

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
        if self._agent_model is None:
            raise RuntimeError("AgentContext has not been initialized")
        return self._agent_model

    def update_tf(self, tf_buffer: tf2_ros.Buffer, sync_agent_model=True):
        """Updates the values of the task frame, base frame, and end effector
        frame from the tf2 buffer

        :param tf_buffer: the buffer from which to update the frame data
        :type tf_buffer: :class:`tf2_ros.Buffer`
        """
        self.base_to_task = self.read_tf(tf_buffer, self.task_frame, self.base_frame)
        self.task_to_base = self.read_tf(tf_buffer, self.base_frame, self.task_frame)
        self.eef_to_task = self.read_tf(tf_buffer, self.task_frame, self.eef_frame)

        if sync_agent_model:
            self.sync_agent_model()

    def read_tf(self, tf_buffer: tf2_ros.Buffer, target_frame: str, source_frame: str):
        now = Time()
        while not tf_buffer.can_transform(target_frame, source_frame, now):
            rclpy.spin_once(self.node, timeout_sec=0.1)
            self.node.get_logger().info(
                f"waiting for trasformation: '{target_frame}' to '{source_frame}'"
            )

        try:
            tf_msg = tf_buffer.lookup_transform(
                target_frame, source_frame, now, timeout=TF_TIMEOUT
            )
            transform = tf_to_transform(tf_msg.transform)
            return transform
        except Exception as e:
            self.node.get_logger().fatal(
                f"Failed to find transforms for agent {self.id}: {e}"
            )

    def sync_agent_model(self):
        """Syncronize the home and base frame positions between the
        agent model and the last update tf frames
        """
        self.agent_model.home_position = self.eef_to_task.t
        self.agent_model.base_frame_position = self.base_to_task.t
        if isinstance(self.agent_model.collision_model, FCLRobotBBCollisionModel):
            self.agent_model.collision_model.anchor = (
                self.agent_model.base_frame_position
            )
