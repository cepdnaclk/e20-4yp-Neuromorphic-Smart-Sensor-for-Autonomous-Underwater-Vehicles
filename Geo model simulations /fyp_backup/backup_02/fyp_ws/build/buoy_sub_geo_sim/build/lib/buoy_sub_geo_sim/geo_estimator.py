import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

from buoy_sub_geo_sim.geo_model import calculate_coordinates


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class GeoEstimator(Node):
    def __init__(self):
        super().__init__('geo_estimator')

        self.buoy_odom = None
        self.sub_odom = None

        self.create_subscription(Odometry, '/buoy/odom', self.buoy_cb, 10)
        self.create_subscription(Odometry, '/submarine/odom', self.sub_cb, 10)

        self.pub = self.create_publisher(PoseStamped, '/submarine/estimated_pose', 10)

        self.timer = self.create_timer(0.5, self.estimate)

        self.get_logger().info('GeoEstimator started')

    def buoy_cb(self, msg):
        self.buoy_odom = msg

    def sub_cb(self, msg):
        self.sub_odom = msg

    def estimate(self):
        if self.buoy_odom is None or self.sub_odom is None:
            return

        bx = self.buoy_odom.pose.pose.position.x
        by = self.buoy_odom.pose.pose.position.y
        bz = self.buoy_odom.pose.pose.position.z

        sx = self.sub_odom.pose.pose.position.x
        sy = self.sub_odom.pose.pose.position.y
        sz = self.sub_odom.pose.pose.position.z

        dx = sx - bx
        dy = sy - by
        dz = sz - bz

        # Simulated measurements from truth
        L = math.sqrt(dx * dx + dy * dy + dz * dz)

        # Example synthetic angles
        alpha = math.atan2(dx, abs(dz) + 1e-6)
        beta = alpha * 0.9
        mu = math.atan2(dy, abs(dz) + 1e-6)
        eta = mu * 0.9

        z_input = abs(dz)

        result = calculate_coordinates(alpha, beta, mu, eta, L, z_input)

        est = PoseStamped()
        est.header.stamp = self.get_clock().now().to_msg()
        est.header.frame_id = 'world'
        est.pose.position.x = bx + result['x']
        est.pose.position.y = by + result['y']
        est.pose.position.z = sz
        est.pose.orientation.w = 1.0

        self.pub.publish(est)


def main():
    rclpy.init()
    node = GeoEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()