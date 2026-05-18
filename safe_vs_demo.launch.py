from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

import os


def generate_launch_description():
    pkg_share = get_package_share_directory('puzzlebot_mc2')
    params = os.path.join(pkg_share, 'config', 'safe_vs_params.yaml')

    return LaunchDescription([
        Node(
            package='puzzlebot_mc2',
            executable='color_features_detector.py',
            name='color_features_detector',
            output='screen',
            parameters=[params],
        ),

        Node(
            package='puzzlebot_mc2',
            executable='safe_vs_controller.py',
            name='safe_vs_controller',
            output='screen',
            parameters=[params],
        ),

        Node(
            package='puzzlebot_mc2',
            executable='cmd_vel_to_wheels.py',
            name='cmd_vel_to_wheels',
            output='screen',
            parameters=[params],
        ),
    ])
