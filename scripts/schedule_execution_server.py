#!/usr/bin/env python

import rospy
import actionlib

from geometry_msgs.msg import Point, Vector3, Quaternion
from taskspace_control_msgs.msg import PoseTwistSetpoint
from pyrobopath_ros.msg import (
    FollowScheduleTrajectoryAction,
    FollowScheduleTrajectoryGoal,
    FollowScheduleTrajectoryFeedback,
    FollowScheduleTrajectoryResult,
)
from pyrobopath_ros.trajectory import create_trajectory


class ScheduleExecutionServer:
    def __init__(self):
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
    rospy.init_node("schedule_execution_server")
    server = ScheduleExecutionServer()
    rospy.spin()
