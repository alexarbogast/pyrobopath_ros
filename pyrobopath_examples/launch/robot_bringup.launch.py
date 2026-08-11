from launch import LaunchDescription
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution

from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    rviz_config_file = (
        PathJoinSubstitution(
            [
                FindPackageShare("pyrobopath_examples"),
                "config",
                "pyrobopath_examples.rviz",
            ]
        ),
    )

    pkg_share = FindPackageShare("pyrobopath_examples")
    controller_config = PathJoinSubstitution(
        [pkg_share, "config", "robot6R_controllers.yaml"]
    )
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
            "/rob1/joint_trajectory_controller",
            "/rob2/joint_trajectory_controller",
            "--controller-manager",
            "controller_manager",
        ],
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
        rviz_node,
    ]

    return LaunchDescription(nodes_to_start)
