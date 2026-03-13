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
        output='screen',
        remappings=[
            ('/model/buoy/odometry', '/buoy/odom'),
            ('/model/submarine/odometry', '/submarine/odom'),
        ]
    )

    fake_gps = Node(
        package='buoy_sub_geo_sim',
        executable='fake_buoy_gps',
        name='fake_buoy_gps',
        parameters=[{
            'origin_lat': 7.2906,
            'origin_lon': 80.6337,
            'origin_alt': 0.0,
        }],
        output='screen'
    )

    estimator = Node(
        package='buoy_sub_geo_sim',
        executable='geo_estimator',
        name='geo_estimator',
        parameters=[{
            'origin_lat': 7.2906,
            'origin_lon': 80.6337,
            'origin_alt': 0.0,
        }],
        output='screen'
    )

    evaluator = Node(
        package='buoy_sub_geo_sim',
        executable='evaluator',
        name='evaluator',
        output='screen'
    )

    return LaunchDescription([
        gazebo,
        bridge,
        fake_gps,
        estimator,
        evaluator,
    ])