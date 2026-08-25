# Demo: simulated 3D landmark field + pf_node localization
# Run: ros2 launch cuda_pf_localization demo.launch.py
# Then: rviz2 (add PointCloud2 /detections, /particles, Marker /landmarks_true)

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory("cuda_pf_localization")
    params = os.path.join(pkg, "config", "defaults.yaml")
    map_file = os.path.join(pkg, "config", "demo_map.yaml")

    sim = Node(
        package="cuda_pf_localization",
        executable="landmark_sim_demo",
        name="landmark_sim_demo",
        output="screen",
        parameters=[
            {"rate": 20.0, "speed": 0.6, "n_landmarks": 30,
             "max_range": 6.0, "range_noise": 0.05, "odom_noise": 0.03, "seed": 7},
        ],
    )

    pf = Node(
        package="cuda_pf_localization",
        executable="pf_node",
        name="pf_node",
        output="screen",
        parameters=[params, {"map_file": map_file}],
        remappings=[
            ("~/landmarks", "/landmark_sim_demo/detections"),
            ("~/odom_pose", "/landmark_sim_demo/odom_pose"),
        ],
    )

    return LaunchDescription([sim, pf])
