cmake_minimum_required(VERSION 3.0.2)
project(pyrobopath_ros)

find_package(catkin REQUIRED COMPONENTS
  rospy
  cartesian_planning_server
  std_msgs
  sensor_msgs
  geometry_msgs
  control_msgs
  trajectory_msgs
  taskspace_control_msgs
  actionlib_msgs
  cartesian_planning_server
)

catkin_python_setup()

add_action_files(DIRECTORY action FILES
  FollowScheduleTrajectory.action
)

add_message_files(DIRECTORY msg FILES
  ScheduleTrajectoryPoint.msg
  ScheduleTrajectory.msg
)

generate_messages(DEPENDENCIES actionlib_msgs std_msgs taskspace_control_msgs)

catkin_package(CATKIN_DEPENDS
  rospy
  std_msgs
  taskspace_control_msgs
  actionlib_msgs
  cartesian_planning_server
)

catkin_install_python(PROGRAMS
  scripts/schedule_execution_server.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
