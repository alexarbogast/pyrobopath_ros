#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionServer, CancelResponse, GoalResponse

from geometry_msgs.msg import Point, Vector3, Quaternion
from taskspace_control_msgs.msg import PoseTwistSetpoint
from pyrobopath_ros.action import FollowScheduleTrajectory

from pyrobopath_ros.trajectory import create_trajectory


class ScheduleExecutionServer(Node):
    def __init__(self):
        super().__init__("schedule_execution_server")

        # Publisher for controller setpoints
        self.setpoint_pub = self.create_publisher(
            PoseTwistSetpoint, "pose_controller/setpoint", 10
        )

        # Action server
        self._action_server = ActionServer(
            self,
            FollowScheduleTrajectory,
            "follow_schedule_trajectory",
            execute_callback=self.execute_cb,
            goal_callback=self.goal_cb,
            cancel_callback=self.cancel_cb,
        )

        self.get_logger().info("Schedule Execution Action Server started.")

        self.action_server = actionlib.SimpleActionServer(
            "follow_schedule_trajectory",
            FollowScheduleTrajectoryAction,
            execute_cb=self.execute_cb,
            auto_start=False,
        )

        self.setpoint_pub = rospy.Publisher(
            "pose_controller/setpoint", PoseTwistSetpoint, queue_size=1
        )

        self.rate = rospy.Rate(500)  # 100 Hz control loop
        self.action_server.start()
        rospy.loginfo("Schedule Execution Action Server started.")

    def execute_cb(self, goal: FollowScheduleTrajectoryGoal):
        traj = create_trajectory(goal.trajectory.points)
        tf = goal.trajectory.points[-1].time_from_start.to_sec()

        start_time = rospy.Time.now()
        success = True
        index = 0

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            t = (now - start_time).to_sec()

            # TODO: handle preemption

            if t > tf:
                break

            while index < len(traj) - 1 and t > traj[index + 1].time_from_start:
                index += 1

            t_sample = t - traj[index].time_from_start
            p, q, v, ω = traj[index].sample(t_sample)
            setpoint_msg = PoseTwistSetpoint()
            setpoint_msg.pose.position = Point(*p)
            setpoint_msg.pose.orientation = Quaternion(q.x, q.y, q.z, q.w)
            setpoint_msg.twist.linear = Vector3(*v)
            setpoint_msg.twist.angular = Vector3(*ω)

            self.setpoint_pub.publish(setpoint_msg)
            feedback = FollowScheduleTrajectoryFeedback(current_index=index)
            self.action_server.publish_feedback(feedback)

            self.rate.sleep()

        result = FollowScheduleTrajectoryResult(success=success)
        self.action_server.set_succeeded(result)


if __name__ == "__main__":
    rclpy.init()
    node = ScheduleExecutionServer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
