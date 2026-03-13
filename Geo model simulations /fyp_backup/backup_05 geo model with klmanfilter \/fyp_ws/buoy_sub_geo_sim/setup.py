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
        'measurement_generator = buoy_sub_geo_sim.measurement_generator:main',
        'geo_estimator = buoy_sub_geo_sim.geo_estimator:main',
        'logger_node = buoy_sub_geo_sim.logger_node:main',
        'auto_runner = buoy_sub_geo_sim.auto_runner:main',
    ],
},

)