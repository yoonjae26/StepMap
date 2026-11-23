# Sensors needed for terrain data collection (no SLAM):
#   Hesai XT-16 driver (/lidar_points), static TFs base_link -> hesai_lidar / utlidar_lidar (/ camera),
#   Go2 odometry -> TF odom -> base_stabilized -> base_link (+ /odom).
#   ros2 launch ~/terrain_dataset/sensors.launch.py
# static_transform_publisher argument order (Foxy): x y z yaw pitch roll parent child   [m, rad]
import glob
import os

from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node

# base_link -> hesai_lidar: estimated from data 2026-10-04; roll corrected to 0 on 2026-10-14
# (the old -0.095 was the lying robot's body pitch).
LIDAR_TF = ["0.22", "0.0", "0.07", "1.5708", "0.0", "0.0", "base_link", "hesai_lidar"]

# base_link -> utlidar_lidar (Go2 L1 in the head): exact transform the robot uses internally, found 2026-10-15
# by ICP between /utlidar/cloud (sensor frame) and /utlidar/cloud_base (base_link): 100 % of points within
# 5 mm. Go2 does not publish this TF itself. (day2_check.py re-measures it.)
L1_TF = ["0.2822", "0.0", "0.0", "-1.0105", "-0.1413", "-2.9207", "base_link", "utlidar_lidar"]

# Depth camera on the head, facing forward — NOT CONNECTED YET. When mounted, measure its position from the
# body centre with a ruler (x forward, y left, z up, metres) and its tilt (pitch > 0 = looking down), then set
# e.g. CAMERA_TF = ["0.30", "0.0", "0.08", "0.0", "0.26", "0.0", "base_link", "camera_link"]   # 15 deg down
# If the camera driver already publishes camera_link -> *_optical_frame (RealSense does), leave
# CAMERA_OPTICAL_TF = None; otherwise it adds the fixed REP-103 optical rotation (z forward, x right, y down).
# 2026-10-18: calibrate_camera.py on the flat floor (floor plane seen by the camera vs body IMU gravity; height from
# the L1 floor): camera 10.8 deg DOWN relative to the body, z 0.189 m above base_link. The 19.5 deg derived on
# 2026-10-17 from the camera accelerometer was wrong (sign of the body pitch). x, y are still guesses (ruler).
CAMERA_TF = ["0.30", "0.0", "0.067", "-0.0099", "0.4541", "-0.0157", "base_link", "camera_link"]  # calibrate_camera.py 2026-10-22 16:17
CAMERA_OPTICAL_TF = ["0", "0", "0", "-1.5708", "0", "-1.5708", "camera_link", "camera_optical_frame"]


# Intel RealSense D435i (realsense-ros 4.0.4, ~/realsense_ws). Started automatically when the camera is on USB
# (TERRAIN_CAMERA=0 disables it). The driver publishes camera_link -> camera_*_optical_frame itself.
# Point cloud without colour texture (stream_filter 0): it then runs at the depth rate; on a USB 2 link the colour
# stream is far slower than depth. IMU: gyro + accel united into /camera/imu.
CAMERA_PARAMS = {"pointcloud.enable": True, "pointcloud.stream_filter": 0, "enable_gyro": True,
                 "enable_accel": True, "unite_imu_method": 2, "enable_infra1": False, "enable_infra2": False,
                 # decimation x2: depth image + cloud 640x480 -> 320x240 (4x less data). Without it the camera
                 # cloud alone is ~57 MB/s and the bag drops /lowstate messages.
                 "decimation_filter.enable": True,
                 # colour image + depth aligned to it: surface_seg_node.py colours the camera image per pixel
                 "enable_color": True, "align_depth.enable": True}


def realsense_connected():
    return any("RealSense" in open(f).read() for f in glob.glob("/sys/bus/usb/devices/*/product"))


def static(name, args):
    return Node(package="tf2_ros", executable="static_transform_publisher", name=name, arguments=args)


def generate_launch_description():
    actions = [
        Node(package="hesai_ros_driver", executable="hesai_ros_driver_node", name="hesai_ros_driver_node",
             output="log"),
        static("base_to_hesai_lidar", LIDAR_TF),
        static("base_to_utlidar_lidar", L1_TF),
        ExecuteProcess(cmd=["/usr/bin/python3", os.path.expanduser("~/gas_mapping/ros/go2_odom_tf.py")],
                       output="log"),
    ]
    camera = os.environ.get("TERRAIN_CAMERA", "1") != "0" and realsense_connected()
    if camera:
        actions.append(Node(package="realsense2_camera", executable="realsense2_camera_node", namespace="camera",
                            name="camera", parameters=[CAMERA_PARAMS], output="log"))
    if CAMERA_TF:
        actions.append(static("base_to_camera", CAMERA_TF))
        if CAMERA_OPTICAL_TF and not camera:
            actions.append(static("camera_to_optical", CAMERA_OPTICAL_TF))
    return LaunchDescription(actions)
