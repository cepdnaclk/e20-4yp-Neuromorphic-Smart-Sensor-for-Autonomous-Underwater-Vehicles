import json
import math
import random

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String

R_EARTH = 6_378_137.0


def xy_to_latlon(origin_lat, origin_lon, x_east, y_north):
    lat_rad = math.radians(origin_lat)
    lat = origin_lat + math.degrees(y_north / R_EARTH)
    lon = origin_lon + math.degrees(x_east / (R_EARTH * math.cos(lat_rad)))
    return lat, lon


class MeasurementGenerator(Node):
    def __init__(self):
        super().__init__('measurement_generator')

        self.declare_parameter('origin_lat', 7.208300)
        self.declare_parameter('origin_lon', 79.835800)
        self.declare_parameter('rtk_noise_m', 0.03)
        self.declare_parameter('angle_noise_deg', 1.5)
        self.declare_parameter('length_noise_m', 0.02)
        self.declare_parameter('depth_noise_m', 0.03)

        self.origin_lat = float(self.get_parameter('origin_lat').value)
        self.origin_lon = float(self.get_parameter('origin_lon').value)
        self.rtk_noise_m = float(self.get_parameter('rtk_noise_m').value)
        self.angle_noise_deg = float(self.get_parameter('angle_noise_deg').value)
        self.length_noise_m = float(self.get_parameter('length_noise_m').value)
        self.depth_noise_m = float(self.get_parameter('depth_noise_m').value)

        self.buoy_odom = None
        self.sub_odom = None

        self.create_subscription(Odometry, '/buoy/odom', self.buoy_cb, 10)
        self.create_subscription(Odometry, '/submarine/odom', self.sub_cb, 10)

        self.meas_pub = self.create_publisher(String, '/simulated_measurements', 10)
        self.gps_pub = self.create_publisher(NavSatFix, '/buoy/rtk_fix', 10)

        self.timer = self.create_timer(0.2, self.publish_measurements)

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

        dx = sx - bx
        dy = sy - by
        depth = abs(sz - bz)
        L_true = math.sqrt(dx * dx + dy * dy + depth * depth)

        if L_true > 5.0:
            return

        noisy_bx = bx + random.gauss(0.0, self.rtk_noise_m)
        noisy_by = by + random.gauss(0.0, self.rtk_noise_m)
        buoy_lat_meas, buoy_lon_meas = xy_to_latlon(self.origin_lat, self.origin_lon, noisy_bx, noisy_by)

        gps = NavSatFix()
        gps.header.stamp = self.get_clock().now().to_msg()
        gps.header.frame_id = 'world'
        gps.latitude = buoy_lat_meas
        gps.longitude = buoy_lon_meas
        gps.altitude = bz
        self.gps_pub.publish(gps)

        alpha_true = math.atan2(abs(dx), depth + 1e-9)
        mu_true = math.atan2(abs(dy), depth + 1e-9)
        beta_true = 0.8 * alpha_true
        eta_true = 0.8 * mu_true

        angle_noise = math.radians(self.angle_noise_deg)

        data = {
            "lat_B": buoy_lat_meas,
            "lon_B": buoy_lon_meas,
            "alpha": alpha_true + random.gauss(0.0, angle_noise),
            "mu": mu_true + random.gauss(0.0, angle_noise),
            "beta": beta_true + random.gauss(0.0, angle_noise),
            "eta": eta_true + random.gauss(0.0, angle_noise),
            "L": max(0.05, min(5.0, L_true + random.gauss(0.0, self.length_noise_m))),
            "z": max(0.01, depth + random.gauss(0.0, self.depth_noise_m))
        }

        msg = String()
        msg.data = json.dumps(data)
        self.meas_pub.publish(msg)


def main():
    rclpy.init()
    node = MeasurementGenerator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()