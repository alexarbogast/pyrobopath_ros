from launch import LaunchDescription
from launch.substitutions import (
    PathJoinSubstitution,
    FindExecutable,
    Command,
)

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("pyrobopath_examples")
    cartesian_planners_config = PathJoinSubstitution(
        [pkg_share, "config", "cartesian_planners.yaml"]
    )
    robot_description_file = PathJoinSubstitution(
        [pkg_share, "urdf", "two_robot6R.xacro"]
    )

    robot_description = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            robot_description_file,
        ]
    )

    rob1_server_node = Node(
        package="cartesian_planning_server",
        executable="cartesian_planning_server",
        name="cartesian_planning_server",
        namespace="rob1",
        output="screen",
        parameters=[
            cartesian_planners_config,
            {"robot_description": robot_description},
        ],
    )
    rob2_server_node = Node(
        package="cartesian_planning_server",
        executable="cartesian_planning_server",
        name="cartesian_planning_server",
        namespace="rob2",
        output="screen",
        parameters=[
            cartesian_planners_config,
            {"robot_description": robot_description},
        ],
    )

    nodes_to_start = [
        rob1_server_node,
        rob2_server_node,
    ]
    return LaunchDescription(nodes_to_start)
