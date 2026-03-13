import math
import os
from datetime import datetime

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped


class LoggerNode(Node):
    def __init__(self):
        super().__init__('logger_node')

        self.truth = None
        self.raw_est = None
        self.filtered_est = None

        self.create_subscription(Odometry, '/submarine/odom', self.truth_cb, 10)
        self.create_subscription(PoseStamped, '/submarine/raw_pose', self.raw_cb, 10)
        self.create_subscription(PoseStamped, '/submarine/estimated_pose', self.filtered_cb, 10)

        self.log_file = os.path.expanduser('~/simulation_log.txt')

        with open(self.log_file, 'w') as f:
            f.write(
                'time,'
                'true_x,true_y,true_z,'
                'raw_x,raw_y,raw_z,raw_error,'
                'kf_x,kf_y,kf_z,kf_error\n'
            )

        self.timer = self.create_timer(0.5, self.write_log)
        self.get_logger().info(f'Logging to {self.log_file}')

    def truth_cb(self, msg):
        self.truth = msg

    def raw_cb(self, msg):
        self.raw_est = msg

    def filtered_cb(self, msg):
        self.filtered_est = msg

    def write_log(self):
        if self.truth is None or self.raw_est is None or self.filtered_est is None:
            return

        tx = self.truth.pose.pose.position.x
        ty = self.truth.pose.pose.position.y
        tz = self.truth.pose.pose.position.z

        rx = self.raw_est.pose.position.x
        ry = self.raw_est.pose.position.y
        rz = self.raw_est.pose.position.z

        kx = self.filtered_est.pose.position.x
        ky = self.filtered_est.pose.position.y
        kz = self.filtered_est.pose.position.z

        raw_err = math.sqrt((tx - rx) ** 2 + (ty - ry) ** 2 + (tz - rz) ** 2)
        kf_err = math.sqrt((tx - kx) ** 2 + (ty - ky) ** 2 + (tz - kz) ** 2)

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        line = (
            f'{now},'
            f'{tx:.3f},{ty:.3f},{tz:.3f},'
            f'{rx:.3f},{ry:.3f},{rz:.3f},{raw_err:.3f},'
            f'{kx:.3f},{ky:.3f},{kz:.3f},{kf_err:.3f}\n'
        )

        with open(self.log_file, 'a') as f:
            f.write(line)

        self.get_logger().info(line.strip())


def main():
    rclpy.init()
    node = LoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()