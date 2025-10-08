"""Pyrobopath interfaces for schedule execution in ROS"""

from __future__ import annotations
from typing import Dict, Optional
import time

# ros
import rclpy
from rclpy.node import Node
import tf2_ros

# pyrobopath
from pyrobopath.process import AgentModel, DependencyGraph, create_dependency_graph_by_z
from pyrobopath.toolpath import Toolpath
from pyrobopath.toolpath_scheduling import (
    MultiAgentToolpathSchedule,
    MultiAgentToolpathPlanner,
    PlanningOptions,
)

# pyrobopath_ros
from .agent_context import AgentContext
from .executor import ExecutorFactory
from .utilities import schedule_info_string


class ScheduleExecution(Node):
    """
    The ScheduleExecution class connects the necessary ROS interfaces to execute
    pyrobopath `ToolpathSchedules` in ROS.

    This object creates and manages an :class:`AgentContext` for each
    namespace in the list provided by the ros parameter namespace. This class
    acts as the interface between pyrobopath scheduling and Cartesian
    trajectory planning and execution.

    Parameters (ROS params):
        namespaces (list of str): A list of unique namespaces for each robot
        exec_method (str): "taskspace_control" or "cartesian_planning"
        retract_height (float, default=0.0)
        collision_offset (float, default=1.0)
        collision_gap_threshold (float, default=0.003)
    """

    schema = {
        "namespaces": {"default": [""], "required": True},
        "exec_method": {"default": "cartesian_planning", "required": False},
        "retract_height": {"default": 0.0, "required": False},
        "collision_offset": {"default": 1.0, "required": False},
        "collision_gap_threshold": {"default": 0.003, "required": False},
    }

    def __init__(self, node_name: str = "schedule_execution") -> None:
        super().__init__(node_name)

        params = {}
        for name, spec in ScheduleExecution.schema.items():
            self.declare_parameter(name, spec["default"])
            value = self.get_parameter(name).value
            if spec["required"] and (value is None or value == "" or value == []):
                raise RuntimeError(f"Missing required parameter: {name}")
            params[name] = value

        self.ros_params = params

        # TF buffer + listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Warm up TF buffer
        start = time.time()
        while time.time() - start < 0.5:
            rclpy.spin_once(self, timeout_sec=0.1)

        self._contexts: Dict[str, AgentContext] = {}
        for ns in params["namespaces"]:
            self._build_agent_contexts(ns)

        self._schedule: Optional[MultiAgentToolpathSchedule] = None

        self._initialize_pyrobopath()
        self.get_logger().info("Initialized pyrobopath: ready to plan!")

        # Create executor (cartesian_planning or taskspace_control)
        method = params["exec_method"]
        self._executor = ExecutorFactory.create(method, self, self._contexts)
        self.get_logger().info(f"Initialized executor with method: {method}")

        # Setup shutdown hook
        rclpy.get_default_context().on_shutdown(self._shutdown)

    @property
    def agent_models(self) -> Dict[str, AgentModel]:
        return {id: context.agent_model for id, context in self._contexts.items()}

    @property
    def schedule(self) -> MultiAgentToolpathSchedule:
        if self._schedule is None:
            raise RuntimeError("Schedule has not been initialized yet.")
        return self._schedule

    def _initialize_pyrobopath(self):
        """Initialize the pyrobopath toolpath planner and planning options
        from ROS parameters
        """
        self._planner = MultiAgentToolpathPlanner(self.agent_models)
        self._options = PlanningOptions(
            retract_height=self.ros_params["retract_height"],
            collision_offset=self.ros_params["collision_offset"],
            collision_gap_threshold=self.ros_params["collision_gap_threshold"],
        )

    def _build_agent_contexts(self, id: str):
        """Build an AgentContext with a unique id and initialize
        the agent's parameters

        :param id: Unique id for agent.
        :type id: str
        """
        self._contexts[id] = AgentContext(id, self)
        self._contexts[id].initialize(self.tf_buffer)

    def move_home(self, tf=2.0):
        """Moves all agents to the joint positions in the `/{ns}/home_position`
        parameter.
        """
        self._executor.move_home(tf)

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
        self.get_logger().info(f"\n{(50 * '#')}\nScheduling Toolpath:\n{(50 * '#')}\n")
        self._schedule = self._planner.plan(toolpath, dependency_graph, self._options)
        self.get_logger().info(f"\n{(50 * '#')}\nFound Toolpath Plan!\n{(50 * '#')}\n")
        self.get_logger().info(schedule_info_string(self._schedule))

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
            self.get_logger().warn("Cannot execute schedule. Schedule is empty.")
            return

        self.get_logger().info(f"\n{(50 * '#')}\nExecuting Schedule\n{(50 * '#')}\n")

        start_time = self.get_clock().now()
        self._executor.execute(self._schedule)
        end_time = self.get_clock().now()
        elapsed = end_time - start_time

        self.get_logger().info(
            f"\n{(50 * '#')}\nSchedule Execution Succeeded\n{(50 * '#')}\n"
        )
        self.get_logger().info(f"Start time: {start_time.nanoseconds / 1e9} sec")
        self.get_logger().info(f"End time: {end_time.nanoseconds / 1e9} sec")
        self.get_logger().info(f"Elapsed: {elapsed.nanoseconds / 1e9} sec\n")

    def _shutdown(self):
        self.get_logger().warn("Received shutdown request. Cancelling all active goals")
        self._executor.shutdown()
