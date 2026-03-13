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

        req = (
            f'name: "submarine", '
            f'position: {{x: {x}, y: {y}, z: {z}}}, '
            f'orientation: {{x: 0, y: 0, z: 0, w: 1}}'
        )

        cmd = [
            'gz', 'service',
            '-s', '/world/ocean_world/set_pose',
            '--reqtype', 'gz.msgs.Pose',
            '--reptype', 'gz.msgs.Boolean',
            '--timeout', '1000',
            '--req', req
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            self.get_logger().info(f'Moved submarine to ({x}, {y}, {z})')
            if result.stdout.strip():
                self.get_logger().info(result.stdout.strip())
        except FileNotFoundError:
            self.get_logger().error('gz command not found')
            self.timer.cancel()
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f'Failed to move submarine: {e}')
            if e.stdout:
                self.get_logger().error(e.stdout)
            if e.stderr:
                self.get_logger().error(e.stderr)

        self.index += 1


def main():
    rclpy.init()
    node = AutoRunner()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()