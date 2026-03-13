import json
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from buoy_sub_geo_sim.geo_model import calculate_coordinates


class GeoEstimator(Node):
    def __init__(self):
        super().__init__('geo_estimator')

        self.create_subscription(String, '/simulated_measurements', self.meas_cb, 10)
        self.pub = self.create_publisher(PoseStamped, '/submarine/estimated_pose', 10)

        self.get_logger().info('geo_estimator started')

    def meas_cb(self, msg):
        data = json.loads(msg.data)

        bx = float(data['buoy_x'])
        by = float(data['buoy_y'])
        bz = float(data['buoy_z'])

        alpha = float(data['alpha'])
        beta = float(data['beta'])
        mu = float(data['mu'])
        eta = float(data['eta'])
        L = float(data['tether_length'])
        depth = float(data['depth'])

        sign_x = float(data['sign_x'])
        sign_y = float(data['sign_y'])

        result = calculate_coordinates(alpha, beta, mu, eta, L, depth)

        est = PoseStamped()
        est.header.stamp = self.get_clock().now().to_msg()
        est.header.frame_id = 'world'
        est.pose.position.x = bx + sign_x * result['x']
        est.pose.position.y = by + sign_y * result['y']
        est.pose.position.z = bz + result['z']
        est.pose.orientation.w = 1.0

        self.pub.publish(est)


def main():
    rclpy.init()
    node = GeoEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()