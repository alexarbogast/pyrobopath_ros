from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = PathJoinSubstitution(
        [FindPackageShare("pyrobopath_examples"), "config", "collision_geometry.yaml"]
    )

    node1 = Node(
        package="pyrobopath_examples",
        executable="visualize_collision_geometry",
        namespace="rob1",
        name="collision_visualization",
        parameters=[config_file],
        output="screen",
    )

    node2 = Node(
        package="pyrobopath_examples",
        executable="visualize_collision_geometry",
        namespace="rob2",
        name="collision_visualization",
        parameters=[config_file],
        output="screen",
    )

    return LaunchDescription([node1, node2])
