from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, IncludeLaunchDescription
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    # Path to Gazebo launch file
    gazebo_launch_file = os.path.join(
        get_package_share_directory('gazebo_ros'),
        'launch',
        'gazebo.launch.py'
    )
    start_gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(gazebo_launch_file)
    )

    sdf_model_path = os.path.join(
        get_package_share_directory('my_env'),
        'models',
        'turtlebot3_waffle',
        'model.sdf'
    )

    # Spawn the robot
    spawn_robot_cmd = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-entity', 'waffle_unique',
            '-file', sdf_model_path,
            '-x', '0.0',
            '-y', '0.0',
            '-z', '0.01'
        ],
        output='screen'
    )

    ld = LaunchDescription()

    # Launch Gazebo and spawn the robot
    ld.add_action(start_gazebo)
    ld.add_action(LogInfo(msg=f"Spawning TurtleBot3 from {sdf_model_path}"))
    ld.add_action(spawn_robot_cmd)

    return ld
