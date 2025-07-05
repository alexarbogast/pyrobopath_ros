#!/usr/bin/env python
import os
from enum import Enum
import numpy as np
from pyrobopath import scheduling
from pyrobopath.toolpath import Toolpath

import rospy
import rospkg

from pyrobopath.process import AgentModel, create_dependency_graph_by_z
from pyrobopath.collision_detection import FCLRobotBBCollisionModel
from pyrobopath.toolpath_scheduling import MultiAgentToolpathPlanner, PlanningOptions

from pyrobopath_ros import toolpath_from_gcode, print_schedule_info


class Materials(Enum):
    MATERIAL_A = 1
    MATERIAL_B = 2


class RoboPath(object):
    def __init__(self):
        bf1 = np.array([-500.0, 0.0, 0.0])
        bf2 = np.array([500.0, 0.0, 0.0])

        # create agent_models
        agent1 = AgentModel(
            base_frame_position=bf1,
            home_position=np.array([-300, 0.0, 0.0]),
            capabilities=[0],
            velocity=50.0,
            travel_velocity=50.0,
            collision_model=FCLRobotBBCollisionModel((200.0, 50.0, 300), bf1),
        )
        agent2 = AgentModel(
            base_frame_position=bf2,
            home_position=np.array([300.0, 0.0, 0.0]),
            capabilities=[1],
            velocity=50.0,
            travel_velocity=50.0,
            collision_model=FCLRobotBBCollisionModel((200.0, 50.0, 300), bf2),
        )
        agent_models = {"robot1": agent1, "robot2": agent2}

        # initialize toolpath planner
        self._planner = MultiAgentToolpathPlanner(agent_models)
        self._options = PlanningOptions(
            retract_height=0.1,
            collision_offset=3.0,
            collision_gap_threshold=5.0,
        )

    def execute(self, filepath: str):
        toolpath = toolpath_from_gcode(filepath)
        toolpath = self.filter_toolpath(toolpath)

        dependency_graph = create_dependency_graph_by_layers(toolpath)
        schedule = self._planner.plan(toolpath, dependency_graph, self._options)
        print_schedule_info(schedule)

    def filter_toolpath(self, toolpath: Toolpath) -> Toolpath:
        toolpath.contours = toolpath.contours[:20]
        return toolpath


if __name__ == "__main__":
    try:
        rospy.init_node("robopath")

        package_path = rospkg.RosPack().get_path("pyrobopath_ros")
        filepath = os.path.join(package_path, "resources", "multi_tool_square.gcode")

        app = RoboPath()
        app.execute(filepath)
    except rospy.ROSInterruptException:
        pass
