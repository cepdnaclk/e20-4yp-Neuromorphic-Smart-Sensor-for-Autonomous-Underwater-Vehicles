import json
import math
import random

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import String


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class MeasurementGenerator(Node):
    def __init__(self):
        super().__init__('measurement_generator')

        self.buoy_odom = None
        self.sub_odom = None

        self.declare_parameter('rtk_noise_xy', 0.03)          # meters
        self.declare_parameter('heading_noise_deg', 1.0)      # degrees
        self.declare_parameter('tether_length_noise', 0.02)   # meters
        self.declare_parameter('angle_noise_deg', 2.0)        # degrees
        self.declare_parameter('depth_noise', 0.03)           # meters

        self.rtk_noise_xy = float(self.get_parameter('rtk_noise_xy').value)
        self.heading_noise_deg = float(self.get_parameter('heading_noise_deg').value)
        self.tether_length_noise = float(self.get_parameter('tether_length_noise').value)
        self.angle_noise_deg = float(self.get_parameter('angle_noise_deg').value)
        self.depth_noise = float(self.get_parameter('depth_noise').value)

        self.create_subscription(Odometry, '/buoy/odom', self.buoy_cb, 10)
        self.create_subscription(Odometry, '/submarine/odom', self.sub_cb, 10)

        self.pub = self.create_publisher(String, '/simulated_measurements', 10)

        self.timer = self.create_timer(0.2, self.publish_measurements)

        self.get_logger().info('measurement_generator started')

    def buoy_cb(self, msg):
        self.buoy_odom = msg

    def sub_cb(self, msg):
        self.sub_odom = msg

    def publish_measurements(self):
        if self.buoy_odom is None or self.sub_odom is None:
            return

        bx = self.buoy_odom.pose.pose.position.x
        by = self.buoy_odom.pose.pose.position.y
        bz = self.buoy_odom.pose.pose.position.z

        sx = self.sub_odom.pose.pose.position.x
        sy = self.sub_odom.pose.pose.position.y
        sz = self.sub_odom.pose.pose.position.z

        buoy_yaw = yaw_from_quaternion(self.buoy_odom.pose.pose.orientation)

        dx = sx - bx
        dy = sy - by
        dz = sz - bz

        true_length = math.sqrt(dx * dx + dy * dy + dz * dz)

        # keep inside tether limit
        if true_length > 5.0:
            return

        horizontal = math.sqrt(dx * dx + dy * dy)
        true_depth = abs(dz)

        # synthetic angles from truth, then noise added
        alpha = math.atan2(abs(dx), true_depth + 1e-6)
        beta = alpha * 0.9
        mu = math.atan2(abs(dy), true_depth + 1e-6)
        eta = mu * 0.9

        # sign handling
        sign_x = 1.0 if dx >= 0 else -1.0
        sign_y = 1.0 if dy >= 0 else -1.0

        # add realistic noise
        bx_meas = bx + random.gauss(0.0, self.rtk_noise_xy)
        by_meas = by + random.gauss(0.0, self.rtk_noise_xy)

        heading_meas = buoy_yaw + random.gauss(0.0, math.radians(self.heading_noise_deg))
        length_meas = true_length + random.gauss(0.0, self.tether_length_noise)
        depth_meas = true_depth + random.gauss(0.0, self.depth_noise)

        alpha_meas = alpha + random.gauss(0.0, math.radians(self.angle_noise_deg))
        beta_meas = beta + random.gauss(0.0, math.radians(self.angle_noise_deg))
        mu_meas = mu + random.gauss(0.0, math.radians(self.angle_noise_deg))
        eta_meas = eta + random.gauss(0.0, math.radians(self.angle_noise_deg))

        data = {
            "buoy_x": bx_meas,
            "buoy_y": by_meas,
            "buoy_z": bz,
            "buoy_heading": heading_meas,
            "tether_length": max(0.05, min(length_meas, 5.0)),
            "depth": max(0.01, depth_meas),
            "alpha": alpha_meas,
            "beta": beta_meas,
            "mu": mu_meas,
            "eta": eta_meas,
            "sign_x": sign_x,
            "sign_y": sign_y,
            "horizontal_truth": horizontal
        }

        msg = String()
        msg.data = json.dumps(data)
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = MeasurementGenerator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()