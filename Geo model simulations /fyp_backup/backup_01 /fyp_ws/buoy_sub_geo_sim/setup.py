from setuptools import setup
from glob import glob
import os

package_name = 'buoy_sub_geo_sim'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name] if os.path.exists('resource/' + package_name) else []),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/worlds', glob('worlds/*.sdf')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='you',
    maintainer_email='you@example.com',
    description='Buoy + submarine + geo model simulation in Gazebo and ROS 2',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'fake_buoy_gps = buoy_sub_geo_sim.fake_buoy_gps:main',
            'geo_estimator = buoy_sub_geo_sim.geo_estimator:main',
            'evaluator = buoy_sub_geo_sim.evaluator:main',
        ],
    },
)