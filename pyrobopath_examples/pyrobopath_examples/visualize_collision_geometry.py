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

        self.marker = Marker()
        self.marker.header.frame_id = params["eef_frame"]
        self.marker.header.stamp = self.get_clock().now().to_msg()

        self.marker.ns = "attached_box"
        self.marker.id = 0
        self.marker.type = Marker.CUBE
        self.marker.action = Marker.ADD
        self.marker.frame_locked = True

        self.marker.pose.position.x = -length / 2 - offset[0]
        self.marker.pose.position.y = offset[1]
        self.marker.pose.position.z = offset[2]
        self.marker.pose.orientation.x = 0.0
        self.marker.pose.orientation.y = 0.0
        self.marker.pose.orientation.z = 0.0
        self.marker.pose.orientation.w = 1.0

        self.marker.scale.x = length
        self.marker.scale.y = width
        self.marker.scale.z = height

        self.marker.color.r = 1.0
        self.marker.color.g = 0.0
        self.marker.color.b = 0.0
        self.marker.color.a = 0.5

        self.marker_timer = self.create_timer(1.0, self.publish_marker)

    def publish_marker(self):
        self.marker.header.stamp = self.get_clock().now().to_msg()
        self.marker_pub.publish(self.marker)


def main(args=None):
    rclpy.init(args=args)
    node = CollisionVisualization()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
