import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('race_docking_mission'),
        'config', 'search_mission.yaml',
    )

    params_file_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='Path to the search mission parameters YAML.',
    )

    node = Node(
        package='race_docking_mission',
        executable='search_mission_node',
        name='search_mission_node',
        namespace='race_auv',
        output='screen',
        parameters=[LaunchConfiguration('params_file')],
    )

    return LaunchDescription([params_file_arg, node])
