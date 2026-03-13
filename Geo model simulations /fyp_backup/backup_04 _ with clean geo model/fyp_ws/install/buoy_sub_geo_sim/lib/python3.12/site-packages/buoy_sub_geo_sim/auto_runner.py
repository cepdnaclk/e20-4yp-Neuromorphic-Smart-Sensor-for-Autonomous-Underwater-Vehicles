import math
import subprocess

import rclpy
from rclpy.node import Node


class AutoRunner(Node):
    def __init__(self):
        super().__init__('auto_runner')

        self.positions = self.generate_positions()
        self.index = 0

        self.timer = self.create_timer(4.0, self.move_next)

        self.get_logger().info(f'auto_runner started with {len(self.positions)} positions')

    def generate_positions(self):
        positions = []
        for x in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
            for y in [-2.0, -1.0, 0.0, 1.0, 2.0]:
                for z in [-1.0, -2.0, -3.0, -4.0]:
                    d = math.sqrt(x * x + y * y + z * z)
                    if d <= 4.8:
                        positions.append((x, y, z))
        return positions

    def move_next(self):
        if self.index >= len(self.positions):
            self.get_logger().info('All positions completed')
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
            self.get_logger().info(f'Moved submarine to ({x}, {y}, {z})')
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f'Failed to move submarine: {e}')

        self.index += 1


def main():
    rclpy.init()
    node = AutoRunner()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()