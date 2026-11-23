"""보도 위험 지점 측정 시스템: the whole measurement pipeline on the Jetson with one command.

  source ~/terrain_dataset/env.sh        # ROS 2 Foxy, CycloneDDS (cyclonedds_jetson.xml), Go2 messages
  ros2 launch ~/terrain_dataset/sidewalk_measure.launch.py session:=20261027_yeondong_A location:="제주시 연동 ..."

What runs (ROS 2 topics between them, results to the StepMap live server, the laptop pulls over WiFi / LAN):
  sensors.launch.py          Hesai XT-16, RealSense D435i (PointCloud2, colour, depth), static TFs, Go2 odom -> TF
  terrain_geometry_node.py   PointCloud2 + TF + odometry -> slope, step, obstacles     /terrain_geometry/slope
  surface_seg_node.py        depth + colour -> 함몰/돌출/구멍, 요철, damage AI 파손/균열  seg_regions, ai_regions,
                             live camera picture (/dev/shm/terrain_live.jpg)
  inspect_node.py            regions -> confirmed events + snapshots                    logs/inspect/<time>/state.json
  foot_node.py               /lowstate (joints, foot force, IMU) -> loose blocks        OFF by default (foot:=true)
                             2026-11-12: StepMap focuses on 파손 / 결손 / 이탈·분리; the loose-block part is not used
  stepmap live (:8060/:8443) + stepmap_bridge.py   events, trajectory, battery, photos -> dashboard
Arguments: session (StepMap session name, default date_time), location (place name for the report),
camera:=false (no RealSense / camera nodes), dashboard:=false (measurement only, no live server / bridge / foot),
sensors:=false (sensors.launch.py is already running, e.g. started at boot: do not open the drivers twice).
The heavy data (point clouds, images) never leaves the Jetson; only results go over the network.
"""
import os
import time

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

HOME = os.path.expanduser("~")
TD = os.path.join(HOME, "terrain_dataset")


def py(script, *args, condition=None, name=None):
    return ExecuteProcess(cmd=["python3", os.path.join(TD, script), "--ros-args", *args], output="log",
                          name=name or script.replace(".py", ""), condition=condition)


def generate_launch_description():
    session, location = LaunchConfiguration("session"), LaunchConfiguration("location")
    camera, dashboard = LaunchConfiguration("camera"), LaunchConfiguration("dashboard")
    sensors = LaunchConfiguration("sensors")
    sess_dir = [os.path.join(HOME, "stepmap_sessions") + "/", session]
    return LaunchDescription([
        DeclareLaunchArgument("session", default_value=time.strftime("%Y%m%d_%H%M%S")),
        DeclareLaunchArgument("location", default_value=""),
        DeclareLaunchArgument("camera", default_value="true"),
        DeclareLaunchArgument("dashboard", default_value="true"),
        DeclareLaunchArgument("sensors", default_value="true"),
        DeclareLaunchArgument("foot", default_value="false"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(os.path.join(TD, "sensors.launch.py")),
                                 condition=IfCondition(sensors)),
        py("terrain_geometry_node.py"),
        py("surface_seg_node.py", condition=IfCondition(camera)),
        py("inspect_node.py", "-p", ["location:=", location], condition=IfCondition(camera)),
        # StepMap live server (dashboard :8060, phone GPS :8443) in its own folder, then bridge + foot node
        ExecuteProcess(cmd=["bash", "-c", ["mkdir -p ", *sess_dir, " && cd ", os.path.join(HOME, "stepmap_live", "stepmap"),
                                          " && [ -f ~/stepmap_live/vworld_key ] && export STEPMAP_VWORLD_KEY=$(cat ~/stepmap_live/vworld_key);"
                                          " exec python3 -m stepmap live -d ", *sess_dir,
                                          " --port 8060 --https-port 8443 -c config.yaml"]],
                       output="log", name="stepmap_live", condition=IfCondition(dashboard)),
        py("stepmap_bridge.py", "-p", ["session:=", session], condition=IfCondition(dashboard)),
        py("foot_node.py", "-p", ["session:=", session], condition=IfCondition(LaunchConfiguration("foot"))),
    ])
