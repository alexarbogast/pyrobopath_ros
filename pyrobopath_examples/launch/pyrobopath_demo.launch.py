from launch import LaunchDescription
from launch.substitutions import PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = PathJoinSubstitution(
        [FindPackageShare("pyrobopath_examples"), "config", "pyrobopath.yaml"]
    )

    pyrobopath_node = Node(
        package="pyrobopath_examples",
        executable="pyrobopath_demo",
        name="pyrobopath_demo",
        output="screen",
        parameters=[config_file],
    )

    return LaunchDescription([pyrobopath_node])
