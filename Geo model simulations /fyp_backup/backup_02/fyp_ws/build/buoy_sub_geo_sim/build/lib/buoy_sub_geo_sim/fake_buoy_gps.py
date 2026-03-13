import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix


class FakeBuoyGPS(Node):
    def __init__(self):
        super().__init__('fake_buoy_gps')

        self.declare_parameter('origin_lat', 7.2906)
        self.declare_parameter('origin_lon', 80.6337)
        self.declare_parameter('origin_alt', 0.0)

        self.origin_lat = float(self.get_parameter('origin_lat').value)
        self.origin_lon = float(self.get_parameter('origin_lon').value)
        self.origin_alt = float(self.get_parameter('origin_alt').value)

        self.sub = self.create_subscription(
            Odometry,
            '/buoy/odom',
            self.odom_callback,
            10
        )

        self.pub = self.create_publisher(NavSatFix, '/buoy/rtk_fix', 10)

        self.get_logger().info('fake_buoy_gps started')

    def odom_callback(self, msg: Odometry):
        x = msg.pose.pose.position.x   # East
        y = msg.pose.pose.position.y   # North
        z = msg.pose.pose.position.z

        lat_rad = math.radians(self.origin_lat)
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(lat_rad)

        dlat = y / meters_per_deg_lat
        dlon = x / meters_per_deg_lon

        nav = NavSatFix()
        nav.header.stamp = self.get_clock().now().to_msg()
        nav.header.frame_id = 'world'
        nav.latitude = self.origin_lat + dlat
        nav.longitude = self.origin_lon + dlon
        nav.altitude = self.origin_alt + z

        # Small covariance for "RTK-like" simulation
        nav.position_covariance = [
            0.01, 0.0, 0.0,
            0.0, 0.01, 0.0,
            0.0, 0.0, 0.04
        ]
        nav.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN

        self.pub.publish(nav)


def main():
    rclpy.init()
    node = FakeBuoyGPS()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()