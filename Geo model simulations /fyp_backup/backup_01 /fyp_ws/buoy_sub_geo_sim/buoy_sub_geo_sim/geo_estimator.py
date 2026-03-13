import math

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix
from geometry_msgs.msg import Quaternion


def yaw_from_quaternion(q: Quaternion) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


class GeoEstimator(Node):
    def __init__(self):
        super().__init__('geo_estimator')

        self.declare_parameter('origin_lat', 7.2906)
        self.declare_parameter('origin_lon', 80.6337)
        self.declare_parameter('origin_alt', 0.0)

        self.origin_lat = float(self.get_parameter('origin_lat').value)
        self.origin_lon = float(self.get_parameter('origin_lon').value)
        self.origin_alt = float(self.get_parameter('origin_alt').value)

        self.buoy_fix = None
        self.buoy_odom = None
        self.sub_odom = None

        self.create_subscription(NavSatFix, '/buoy/rtk_fix', self.on_buoy_fix, 10)
        self.create_subscription(Odometry, '/buoy/odom', self.on_buoy_odom, 10)
        self.create_subscription(Odometry, '/submarine/odom', self.on_sub_odom, 10)

        self.est_fix_pub = self.create_publisher(NavSatFix, '/submarine/estimated_fix', 10)
        self.est_odom_pub = self.create_publisher(Odometry, '/submarine/estimated_odom', 10)

        self.timer = self.create_timer(0.1, self.update)

        self.get_logger().info('geo_estimator started')

    def on_buoy_fix(self, msg):
        self.buoy_fix = msg

    def on_buoy_odom(self, msg):
        self.buoy_odom = msg

    def on_sub_odom(self, msg):
        self.sub_odom = msg

    def run_geo_model(self, buoy_x, buoy_y, buoy_z, buoy_yaw, sub_x, sub_y, sub_z):
        """
        Replace this with YOUR geo model.
        Current version uses a simple straight-tether geometry example.

        Returns estimated submarine local coordinates (x, y, z)
        """

        dx = sub_x - buoy_x
        dy = sub_y - buoy_y
        dz = sub_z - buoy_z

        tether_length = math.sqrt(dx*dx + dy*dy + dz*dz)
        horizontal = math.sqrt(dx*dx + dy*dy)

        azimuth_world = math.atan2(dy, dx)
        azimuth_relative_to_buoy = azimuth_world - buoy_yaw
        depth = buoy_z - sub_z

        # --- Example: reconstruct local position from geometry ---
        est_horizontal = math.sqrt(max(tether_length**2 - depth**2, 0.0))
        est_azimuth_world = buoy_yaw + azimuth_relative_to_buoy

        est_dx = est_horizontal * math.cos(est_azimuth_world)
        est_dy = est_horizontal * math.sin(est_azimuth_world)
        est_dz = -depth

        est_x = buoy_x + est_dx
        est_y = buoy_y + est_dy
        est_z = buoy_z + est_dz

        return est_x, est_y, est_z

    def local_xy_to_latlon(self, x, y, z):
        lat_rad = math.radians(self.origin_lat)
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(lat_rad)

        lat = self.origin_lat + (y / meters_per_deg_lat)
        lon = self.origin_lon + (x / meters_per_deg_lon)
        alt = self.origin_alt + z
        return lat, lon, alt

    def update(self):
        if self.buoy_fix is None or self.buoy_odom is None or self.sub_odom is None:
            return

        buoy_pos = self.buoy_odom.pose.pose.position
        buoy_ori = self.buoy_odom.pose.pose.orientation
        sub_pos = self.sub_odom.pose.pose.position

        buoy_yaw = yaw_from_quaternion(buoy_ori)

        est_x, est_y, est_z = self.run_geo_model(
            buoy_pos.x, buoy_pos.y, buoy_pos.z,
            buoy_yaw,
            sub_pos.x, sub_pos.y, sub_pos.z
        )

        lat, lon, alt = self.local_xy_to_latlon(est_x, est_y, est_z)

        est_fix = NavSatFix()
        est_fix.header.stamp = self.get_clock().now().to_msg()
        est_fix.header.frame_id = 'world'
        est_fix.latitude = lat
        est_fix.longitude = lon
        est_fix.altitude = alt
        self.est_fix_pub.publish(est_fix)

        est_odom = Odometry()
        est_odom.header.stamp = est_fix.header.stamp
        est_odom.header.frame_id = 'world'
        est_odom.child_frame_id = 'submarine_estimated'
        est_odom.pose.pose.position.x = est_x
        est_odom.pose.pose.position.y = est_y
        est_odom.pose.pose.position.z = est_z
        est_odom.pose.pose.orientation.w = 1.0
        self.est_odom_pub.publish(est_odom)


def main():
    rclpy.init()
    node = GeoEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()