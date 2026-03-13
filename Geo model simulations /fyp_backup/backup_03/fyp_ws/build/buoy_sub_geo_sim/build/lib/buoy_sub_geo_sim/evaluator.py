import math

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry


class Evaluator(Node):
    def __init__(self):
        super().__init__('evaluator')

        self.truth = None
        self.estimate = None

        self.create_subscription(Odometry, '/submarine/odom', self.truth_cb, 10)
        self.create_subscription(Odometry, '/submarine/estimated_odom', self.est_cb, 10)

        self.timer = self.create_timer(0.5, self.report)

        self.get_logger().info('evaluator started')

    def truth_cb(self, msg):
        self.truth = msg

    def est_cb(self, msg):
        self.estimate = msg

    def report(self):
        if self.truth is None or self.estimate is None:
            return

        tx = self.truth.pose.pose.position.x
        ty = self.truth.pose.pose.position.y
        tz = self.truth.pose.pose.position.z

        ex = self.estimate.pose.pose.position.x
        ey = self.estimate.pose.pose.position.y
        ez = self.estimate.pose.pose.position.z

        err = math.sqrt((tx - ex) ** 2 + (ty - ey) ** 2 + (tz - ez) ** 2)

        self.get_logger().info(f'Position error = {err:.3f} m')


def main():
    rclpy.init()
    node = Evaluator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()