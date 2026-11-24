from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg_share = FindPackageShare("pyrobopath_examples")
    rviz_config_file = PathJoinSubstitution(
        [pkg_share, "config", "pyrobopath_examples.rviz"]
    )
    controller_config = PathJoinSubstitution([pkg_share, "config", "controllers.yaml"])
    description_file = PathJoinSubstitution([pkg_share, "urdf", "two_robot6R.xacro"])

    robot_description = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            description_file,
        ]
    )

    control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            {"robot_description": robot_description},
            controller_config,
        ],
        output="both",
    )

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name="robot_state_publisher",
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    robot_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=[
            "joint_state_broadcaster",
            "rob1_joint_trajectory_controller",
            "rob2_joint_trajectory_controller",
            "--controller-manager",
            "controller_manager",
        ],
    )

    planning_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("pyrobopath_examples"),
                    "launch",
                    "cartesian_planning_servers.launch.py",
                ]
            )
        ),
    )

    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", rviz_config_file],
    )

    nodes_to_start = [
        control_node,
        robot_state_publisher_node,
        robot_controller_spawner,
        planning_launch,
        rviz_node,
    ]

    return LaunchDescription(nodes_to_start)
