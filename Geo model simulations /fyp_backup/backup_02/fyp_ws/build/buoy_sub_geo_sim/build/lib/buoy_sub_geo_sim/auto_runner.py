import subprocess
import time

import rclpy
from rclpy.node import Node


class AutoRunner(Node):
    def __init__(self):
        super().__init__('auto_runner')

        self.positions = [
            (5.0, 0.0, -8.0),
            (8.0, 2.0, -10.0),
            (10.0, -3.0, -12.0),
            (12.0, 4.0, -15.0),
            (15.0, -5.0, -18.0),
        ]

        self.index = 0
        self.timer = self.create_timer(5.0, self.move_next)

        self.get_logger().info('AutoRunner started')

    def move_next(self):
        if self.index >= len(self.positions):
            self.get_logger().info('All simulation steps completed')
            self.timer.cancel()
            return

        x, y, z = self.positions[self.index]

        cmd = [
            'gz', 'service',
            '-s', '/world/ocean_world/set_pose',
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000',
            '--req',
            f'name: "submarine", position: {{x: {x}, y: {y}, z: {z}}}, orientation: {{x: 0, y: 0, z: 0, w: 1}}'
        ]

        try:
            subprocess.run(cmd, check=True)
            self.get_logger().info(f'Moved submarine to x={x}, y={y}, z={z}')
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f'Failed to move submarine: {e}')

        self.index += 1


def main():
    rclpy.init()
    node = AutoRunner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()