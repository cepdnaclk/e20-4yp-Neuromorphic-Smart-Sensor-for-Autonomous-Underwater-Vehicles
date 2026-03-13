import json
import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix
from std_msgs.msg import String

from buoy_sub_geo_sim.geo_model import (
    AUVKalmanFilter,
    get_auv_position,
    gps_to_local_xy,
)


class GeoEstimator(Node):
    def __init__(self):
        super().__init__('geo_estimator')

        self.declare_parameter('origin_lat', 7.208300)
        self.declare_parameter('origin_lon', 79.835800)
        self.declare_parameter('process_noise', 0.3)
        self.declare_parameter('measurement_noise', 1.5)

        self.origin_lat = float(self.get_parameter('origin_lat').value)
        self.origin_lon = float(self.get_parameter('origin_lon').value)

        process_noise = float(self.get_parameter('process_noise').value)
        measurement_noise = float(self.get_parameter('measurement_noise').value)

        self.kf = AUVKalmanFilter(
            process_noise=process_noise,
            measurement_noise=measurement_noise
        )

        self.last_time = None

        self.create_subscription(String, '/simulated_measurements', self.meas_cb, 10)

        self.raw_fix_pub = self.create_publisher(NavSatFix, '/submarine/raw_fix', 10)
        self.raw_pose_pub = self.create_publisher(PoseStamped, '/submarine/raw_pose', 10)

        self.filtered_fix_pub = self.create_publisher(NavSatFix, '/submarine/estimated_fix', 10)
        self.filtered_pose_pub = self.create_publisher(PoseStamped, '/submarine/estimated_pose', 10)

        self.get_logger().info('geo_estimator with Kalman filter started')

    def publish_fix(self, pub, lat, lon, depth):
        msg = NavSatFix()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'world'
        msg.latitude = lat
        msg.longitude = lon
        msg.altitude = -depth
        pub.publish(msg)

    def publish_pose(self, pub, lat, lon, depth):
        x, y = gps_to_local_xy(self.origin_lat, self.origin_lon, lat, lon)

        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = 'world'
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = -depth
        pose.pose.orientation.w = 1.0
        pub.publish(pose)

    def meas_cb(self, msg):
        data = json.loads(msg.data)

        lat_B = float(data['lat_B'])
        lon_B = float(data['lon_B'])
        alpha = float(data['alpha'])
        mu = float(data['mu'])
        beta = float(data['beta'])
        eta = float(data['eta'])
        L = float(data['L'])
        z = float(data['z'])

        now = time.time()
        dt = 0.2 if self.last_time is None else max(0.01, now - self.last_time)
        self.last_time = now

        try:
            raw = get_auv_position(lat_B, lon_B, alpha, mu, beta, eta, L, z)

            self.publish_fix(
                self.raw_fix_pub,
                raw['lat_AUV'],
                raw['lon_AUV'],
                raw['depth_AUV']
            )
            self.publish_pose(
                self.raw_pose_pub,
                raw['lat_AUV'],
                raw['lon_AUV'],
                raw['depth_AUV']
            )

            self.kf.predict(dt)
            result = self.kf.update_from_geometry(lat_B, lon_B, alpha, mu, beta, eta, L, z)
            filtered = result['filtered']

            self.publish_fix(
                self.filtered_fix_pub,
                filtered['lat_AUV'],
                filtered['lon_AUV'],
                filtered['depth_AUV']
            )
            self.publish_pose(
                self.filtered_pose_pub,
                filtered['lat_AUV'],
                filtered['lon_AUV'],
                filtered['depth_AUV']
            )

        except Exception as e:
            self.get_logger().warn(f'Geo estimation skipped sample: {e}')


def main():
    rclpy.init()
    node = GeoEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()