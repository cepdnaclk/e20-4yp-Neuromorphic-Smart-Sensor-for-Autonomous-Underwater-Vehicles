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

        self.sub_truth = None
        self.sub_est = None

        self.create_subscription(Odometry, '/submarine/odom', self.truth_cb, 10)
        self.create_subscription(PoseStamped, '/submarine/estimated_pose', self.est_cb, 10)

        self.log_file = os.path.expanduser('~/simulation_log.txt')

        with open(self.log_file, 'w') as f:
            f.write('time,true_x,true_y,true_z,est_x,est_y,est_z,error\n')

        self.timer = self.create_timer(1.0, self.write_log)

        self.get_logger().info(f'Logging to {self.log_file}')

    def truth_cb(self, msg):
        self.sub_truth = msg

    def est_cb(self, msg):
        self.sub_est = msg

    def write_log(self):
        if self.sub_truth is None or self.sub_est is None:
            return

        tx = self.sub_truth.pose.pose.position.x
        ty = self.sub_truth.pose.pose.position.y
        tz = self.sub_truth.pose.pose.position.z

        ex = self.sub_est.pose.position.x
        ey = self.sub_est.pose.position.y
        ez = self.sub_est.pose.position.z

        error = math.sqrt((tx - ex) ** 2 + (ty - ey) ** 2 + (tz - ez) ** 2)

        now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        line = f'{now},{tx:.3f},{ty:.3f},{tz:.3f},{ex:.3f},{ey:.3f},{ez:.3f},{error:.3f}\n'

        with open(self.log_file, 'a') as f:
            f.write(line)

        self.get_logger().info(line.strip())


def main():
    rclpy.init()
    node = LoggerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()