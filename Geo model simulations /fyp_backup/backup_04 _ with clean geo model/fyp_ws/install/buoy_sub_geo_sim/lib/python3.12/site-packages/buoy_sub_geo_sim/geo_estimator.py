import json
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String

from buoy_sub_geo_sim.geo_model import get_auv_position


R_EARTH = 6_378_137.0


def gps_to_local_xy(lat0, lon0, lat, lon):
    lat0_rad = math.radians(lat0)
    dy = math.radians(lat - lat0) * R_EARTH
    dx = math.radians(lon - lon0) * R_EARTH * math.cos(lat0_rad)
    return dx, dy


class GeoEstimator(Node):
    def __init__(self):
        super().__init__('geo_estimator')

        self.declare_parameter('origin_lat', 7.208300)
        self.declare_parameter('origin_lon', 79.835800)

        self.origin_lat = float(self.get_parameter('origin_lat').value)
        self.origin_lon = float(self.get_parameter('origin_lon').value)

        self.create_subscription(String, '/simulated_measurements', self.meas_cb, 10)

        self.fix_pub = self.create_publisher(NavSatFix, '/submarine/estimated_fix', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/submarine/estimated_pose', 10)

    def meas_cb(self, msg):
        data = json.loads(msg.data)

        try:
            result = get_auv_position(
                lat_B=float(data['lat_B']),
                lon_B=float(data['lon_B']),
                alpha=float(data['alpha']),
                mu=float(data['mu']),
                beta=float(data['beta']),
                eta=float(data['eta']),
                L=float(data['L']),
                z=float(data['z']),
            )
        except Exception as e:
            self.get_logger().warn(f'Geo model skipped sample: {e}')
            return

        nav = NavSatFix()
        nav.header.stamp = self.get_clock().now().to_msg()
        nav.header.frame_id = 'world'
        nav.latitude = result['lat_AUV']
        nav.longitude = result['lon_AUV']
        nav.altitude = -result['depth_AUV']
        self.fix_pub.publish(nav)

        x_local, y_local = gps_to_local_xy(
            self.origin_lat,
            self.origin_lon,
            result['lat_AUV'],
            result['lon_AUV']
        )

        pose = PoseStamped()
        pose.header = nav.header
        pose.pose.position.x = x_local
        pose.pose.position.y = y_local
        pose.pose.position.z = -result['depth_AUV']
        pose.pose.orientation.w = 1.0
        self.pose_pub.publish(pose)


def main():
    rclpy.init()
    node = GeoEstimator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()