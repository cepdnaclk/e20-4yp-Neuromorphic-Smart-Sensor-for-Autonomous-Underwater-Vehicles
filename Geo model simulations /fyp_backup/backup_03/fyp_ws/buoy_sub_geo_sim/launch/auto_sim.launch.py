import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('buoy_sub_geo_sim')
    world = os.path.join(pkg_share, 'worlds', 'ocean_world.sdf')

    gazebo = ExecuteProcess(
        cmd=['gz', 'sim', '-r', world],
        output='screen'
    )

    bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=[
            '/model/buoy/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
            '/model/submarine/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry',
        ],
        remappings=[
            ('/model/buoy/odometry', '/buoy/odom'),
            ('/model/submarine/odometry', '/submarine/odom'),
        ],
        output='screen'
    )

    measurement_generator = Node(
        package='buoy_sub_geo_sim',
        executable='measurement_generator',
        name='measurement_generator',
        output='screen',
        parameters=[{
            'rtk_noise_xy': 0.03,
            'heading_noise_deg': 1.0,
            'tether_length_noise': 0.02,
            'angle_noise_deg': 2.0,
            'depth_noise': 0.03,
        }]
    )

    estimator = Node(
        package='buoy_sub_geo_sim',
        executable='geo_estimator',
        name='geo_estimator',
        output='screen'
    )

    logger = Node(
        package='buoy_sub_geo_sim',
        executable='logger_node',
        name='logger_node',
        output='screen'
    )

    runner = Node(
        package='buoy_sub_geo_sim',
        executable='auto_runner',
        name='auto_runner',
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        bridge,
        measurement_generator,
        estimator,
        logger,
        runner
    ])