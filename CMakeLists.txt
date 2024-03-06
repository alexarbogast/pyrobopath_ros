cmake_minimum_required(VERSION 3.0.2)
project(pyrobopath_ros)

find_package(catkin REQUIRED COMPONENTS
rospy
cartesian_planning_server
sensor_msgs
geometry_msgs
control_msgs
trajectory_msgs
cartesian_planning_server
)

catkin_python_setup()

catkin_package(CATKIN_DEPENDS
rospy
cartesian_planning_server
)

catkin_install_python(PROGRAMS
scripts/pyrobopath_ros_demo.py
DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
