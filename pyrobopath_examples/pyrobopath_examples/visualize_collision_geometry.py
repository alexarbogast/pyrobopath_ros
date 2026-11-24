#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
from visualization_msgs.msg import Marker


class CollisionVisualization(Node):
    schema = {
        "eef_frame": {"default": "", "required": True},
        "collision.length": {"default": 0.0, "required": True},
        "collision.width": {"default": 0.0, "required": True},
        "collision.height": {"default": 0.0, "required": True},
        "collision.offset": {"default": [0.0, 0.0, 0.0], "required": False},
    }

    def __init__(self):
        super().__init__("collision_visualization")

        params = {}
        for name, spec in CollisionVisualization.schema.items():
            self.declare_parameter(name, spec["default"])
            value = self.get_parameter(name).value
            if spec["required"] and (value is None or value == "" or value == []):
                raise RuntimeError(f"Missing required parameter: {name}")
            params[name] = value

        length = params["collision.length"]
        width = params["collision.width"]
        height = params["collision.height"]
        offset = params["collision.offset"]

        # Publisher with transient_local QoS (like latch=True in ROS 1)
        qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(Marker, "visualization_marker", qos)

        marker = Marker()
        marker.header.frame_id = params["eef_frame"]
        marker.header.stamp = self.get_clock().now().to_msg()

        marker.ns = "attached_box"
        marker.id = 0
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.frame_locked = True

        marker.pose.position.x = -length / 2 - offset[0]
        marker.pose.position.y = offset[1]
        marker.pose.position.z = offset[2]
        marker.pose.orientation.x = 0.0
        marker.pose.orientation.y = 0.0
        marker.pose.orientation.z = 0.0
        marker.pose.orientation.w = 1.0

        marker.scale.x = length
        marker.scale.y = width
        marker.scale.z = height

        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.5

        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = CollisionVisualization()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
