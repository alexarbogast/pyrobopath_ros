cmake_minimum_required(VERSION 3.5)
project(pyrobopath_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

set(msg_files
  "msg/ScheduleTrajectoryPoint.msg"
  "msg/ScheduleTrajectory.msg"
)

set(action_files
  "action/FollowScheduleTrajectory.action"
)

rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${action_files}
  DEPENDENCIES std_msgs geometry_msgs action_msgs
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()
