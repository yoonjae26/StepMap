#!/usr/bin/env python3
"""Terrain geometry node — day 4: receive a PointCloud2 and check that the input is stable;
stage 3: live plane / slope / step estimate with RANSAC (slope_core.py, same code as the offline evaluation).

Per message, in order:
  1. receive PointCloud2 (any XYZ cloud: RealSense /camera/depth/color/points, Hesai /lidar_points, ...)
  2. read the x, y, z fields straight from the binary buffer (numpy, no sensor_msgs_py in Foxy)
  3. drop NaN / Inf points (organized depth clouds are full of NaN where there is no depth)
  4. keep only a box in front of the robot, expressed in base_link (x forward, y left, z up):
     the cloud is transformed with TF when target_frame -> sensor frame exists; without TF an
     *_optical_frame cloud is rotated to body axes (z fwd, x right, y down -> x fwd, y left, z up)
     and anything else is used as it is (status says which one)
  4b. check timestamp and frame: stamp not zero, not going backwards, not older than max_age s (ROS clock,
     so it also works with use_sim_time); frame_id not empty and not changing between messages
  5. output the number of valid / ROI points, the distance to the ROI points (horizontal range from base_link:
     min / median / max) and a data status, on /terrain_geometry/status (std_msgs/String, JSON) + CSV
  6. log: a WARN when TF is missing, the data is too sparse (< min_points), the stamp/frame is wrong, or no
     cloud arrives for timeout s; status changes are logged at once, plus one summary line every log_period s
     (logs/terrain_geometry_<time>.csv has one row per cloud)

  7. (estimate_slope) at most slope_rate times per second: slope_core.estimate() on the cloud in base_link,
     levelled with the body IMU (/lowstate roll, pitch), body height from /odom; optional multi-scan window
     (slope_window s: clouds merged in the odom frame via TF). Publishes /terrain_geometry/slope (JSON: status,
     a b c d, slope_deg, slope_pct, along/side, step) and /terrain_geometry/slope_deg (Float32, NaN if not OK);
     logs/terrain_slope_<time>.csv. Slope status: OK / NO_IMU / NO_TF / FEW_POINTS / RANSAC_WEAK / BAD_NORMAL /
     BAD_HEIGHT (see slope_core.py). profile: camera | l1 | hesai | auto (from the topic name).

Status (worst first): ERROR (cannot parse) > NO_FRAME > BAD_STAMP > NO_VALID > NO_TF > LOW_POINTS > STALE > OK.
NO_TF: target_frame <- cloud frame missing; the ROI is still computed (optical axes or raw) so counts stay
comparable, but it is not trusted.

  source ~/terrain_dataset/env.sh
  python3 ~/terrain_dataset/terrain_geometry_node.py                                  # camera (default)
  python3 ~/terrain_dataset/terrain_geometry_node.py --ros-args -p topic:=/lidar_points
  python3 ~/terrain_dataset/terrain_geometry_node.py --ros-args -p topic:=/utlidar/cloud_deskewed -p slope_window:=1.0 -p max_age:=0.0
  # replay: ROS_DOMAIN_ID=42 ... -p use_sim_time:=true

Parameters: topic, target_frame (base_link), roi_x_min/max (0.3 / 2.0 m), roi_y_abs (0.8 m),
roi_z_min/max (-1.0 / 0.5 m, relative to base_link), min_points (300), max_age (1.0 s; 0 = no age check — the
Go2 L1 clouds carry the Go2 clock, 100-170 s off the Jetson clock), log_period (2.0 s),
timeout (2.0 s), csv (true); estimate_slope (true), profile (auto), slope_rate (5 Hz), slope_window (0 s),
imu_topic (/lowstate), odom_topic (/utlidar/robot_odom: body height, and odom <-> base_link when no TF).
"""
import csv
import json
import math
import os
import time

import numpy as np
import rclpy
from rclpy.clock import Clock, ClockType
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import String
import tf2_ros
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32

import slope_core as sc
import terrain_viz as tv
from visualization_msgs.msg import MarkerArray
from geometry_msgs.msg import TransformStamped

# PointField datatype -> numpy type
NP_TYPES = {PointField.INT8: "i1", PointField.UINT8: "u1", PointField.INT16: "i2", PointField.UINT16: "u2",
            PointField.INT32: "i4", PointField.UINT32: "u4", PointField.FLOAT32: "f4", PointField.FLOAT64: "f8"}

# optical (z forward, x right, y down) -> body axes (x forward, y left, z up)
OPTICAL_TO_BODY = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=np.float64)


def read_xyz(msg):
    """Step 2: N x 3 float32 array of x, y, z from a PointCloud2 (any point_step / extra fields)."""
    fields = {f.name: f for f in msg.fields}
    missing = [c for c in "xyz" if c not in fields]
    if missing:
        raise ValueError(f"cloud has no field(s) {missing}; fields = {list(fields)}")
    endian = ">" if msg.is_bigendian else "<"
    dtype = np.dtype({"names": list("xyz"),
                      "formats": [endian + NP_TYPES[fields[c].datatype] for c in "xyz"],
                      "offsets": [fields[c].offset for c in "xyz"],
                      "itemsize": msg.point_step})
    n = msg.width * msg.height
    buf = np.frombuffer(msg.data, dtype=dtype, count=n) if msg.row_step == msg.width * msg.point_step else \
        np.concatenate([np.frombuffer(msg.data, dtype=dtype, count=msg.width, offset=r * msg.row_step)
                        for r in range(msg.height)])
    return np.stack([buf["x"], buf["y"], buf["z"]], axis=1).astype(np.float32)


def quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


SLOPE_COLS = ["status", "reason", "profile", "n_scans", "n_road", "n_inliers", "inlier_ratio", "a", "b", "c", "d",
              "slope_deg", "slope_pct", "along_deg", "side_deg", "slope_unlevelled_deg", "step_detected", "step_dz_m",
              "step_dist_m", "step_confirmed", "step_confirmed_dz_m", "step_confirmed_dist_m", "n_obstacles",
              "obstacle_dist_m", "obstacle_height_m", "n_obstacles_confirmed", "obstacle_confirmed_dist_m",
              "obstacle_confirmed_height_m", "body_roll_deg", "body_pitch_deg", "body_height", "proc_ms"]


class TerrainGeometryNode(Node):
    def __init__(self):
        super().__init__("terrain_geometry_node")
        p = self.declare_parameter
        self.topic = p("topic", "/camera/depth/color/points").value
        self.target_frame = p("target_frame", "base_link").value
        self.roi = dict(x_min=p("roi_x_min", 0.3).value, x_max=p("roi_x_max", 2.0).value,
                        y_abs=p("roi_y_abs", 0.8).value,
                        z_min=p("roi_z_min", -1.0).value, z_max=p("roi_z_max", 0.5).value)
        self.min_points = p("min_points", 150 if "utlidar" in self.topic else 300).value
        self.log_period = p("log_period", 2.0).value
        self.timeout = p("timeout", 2.0).value
        self.max_age = p("max_age", 1.0).value

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.sub = self.create_subscription(PointCloud2, self.topic, self.on_cloud, qos_profile_sensor_data)
        self.status_pub = self.create_publisher(String, "/terrain_geometry/status", 10)
        # Go2 odometry (published by the robot itself): body height + odom <-> base_link fallback for the TF
        self.create_subscription(Odometry, p("odom_topic", "/utlidar/robot_odom").value, self.on_odom,
                                 qos_profile_sensor_data)

        # stage 3: live slope estimate
        self.do_slope = p("estimate_slope", True).value
        prof = p("profile", "auto").value
        if prof == "auto":
            prof = "hesai" if "lidar_points" in self.topic else ("l1" if "utlidar" in self.topic else "camera")
        self.profile = prof
        self.slope_rate = p("slope_rate", 5.0).value
        self.slope_window = p("slope_window", 0.0).value
        self.max_points = p("max_points", 20000).value      # slope estimate on at most this many points
        self.rpy = None                 # body roll, pitch from the IMU
        self.body_height = 0.30
        self.next_slope = 0.0
        self.scans = []                 # (stamp, points in odom) for slope_window
        self.slope_hist = []            # (status, slope_deg, step_dz) since the last summary
        self.last_slope_status = None
        if self.do_slope:
            try:
                from unitree_go.msg import LowState
                self.create_subscription(LowState, p("imu_topic", "/lowstate").value, self.on_lowstate,
                                         qos_profile_sensor_data)
            except ImportError:
                self.get_logger().error("unitree_go messages not found: source ~/terrain_dataset/env.sh")
            self.slope_pub = self.create_publisher(String, "/terrain_geometry/slope", 10)
            self.confirm = tv.Confirm(p("confirm_n", 3).value)
            self.viz = p("viz", True).value
            if self.viz:
                self.viz_cloud_pub = self.create_publisher(PointCloud2, "/terrain_geometry/cloud_viz", 2)
                self.viz_marker_pub = self.create_publisher(MarkerArray, "/terrain_geometry/markers", 2)
                self.tf_pub = tf2_ros.TransformBroadcaster(self)   # base_link -> terrain_level (levelled)
            self.slope_deg_pub = self.create_publisher(Float32, "/terrain_geometry/slope_deg", 10)

        # counters for the periodic summary
        self.total_msgs = 0
        self.window = []            # (wall time, n_total, n_valid, n_roi, ok, proc_ms) since last summary
        self.last_rx = None         # wall time of the last message
        self.errors = 0
        self.first_frame = None
        self.warned_no_data = False
        self.last_stamp = None
        self.last_status = None
        self.status_count = {}

        self.csv_file = self.csv = None
        if p("csv", True).value:
            logdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            os.makedirs(logdir, exist_ok=True)
            path = os.path.join(logdir, time.strftime("terrain_geometry_%Y%m%d_%H%M%S.csv"))
            self.csv_file = open(path, "w", newline="")
            self.csv = csv.writer(self.csv_file)
            self.csv.writerow(["wall_time", "stamp", "frame_id", "transform", "n_total", "n_valid", "n_nan_inf",
                               "n_roi", "dist_min", "dist_median", "dist_max", "age_s", "status", "detail", "proc_ms"])
            self.get_logger().info(f"CSV log: {path}")
            if self.do_slope:
                spath = os.path.join(logdir, time.strftime("terrain_slope_%Y%m%d_%H%M%S.csv"))
                self.slope_file = open(spath, "w", newline="")
                self.slope_csv = csv.writer(self.slope_file)
                self.slope_csv.writerow(["stamp"] + SLOPE_COLS)
                self.get_logger().info(f"slope CSV: {spath}")

        # steady (wall) clock: with use_sim_time the ROS clock stops when a replay stops, and so would the watchdog
        self.create_timer(self.log_period, self.summary, clock=Clock(clock_type=ClockType.STEADY_TIME))
        self.t_start = time.time()
        r = self.roi
        self.get_logger().info(
            f"listening on {self.topic}; ROI in {self.target_frame}: x {r['x_min']}..{r['x_max']} m, "
            f"|y| < {r['y_abs']} m, z {r['z_min']}..{r['z_max']} m; min_points {self.min_points}")
        if self.do_slope:
            self.get_logger().info(f"slope estimate: profile {self.profile}, <= {self.slope_rate:g} Hz, "
                                   f"window {self.slope_window:g} s, ROI {sc.SENSORS[self.profile]['roi_x']} m ahead")

    def on_lowstate(self, m):
        self.rpy = (float(m.imu_state.rpy[0]), float(m.imu_state.rpy[1]))

    def on_odom(self, m):
        po, q = m.pose.pose.position, m.pose.pose.orientation
        self.body_height = float(po.z)
        self.odom_pose = (quat_to_matrix(q), np.array([po.x, po.y, po.z]))     # base_link -> odom

    def lookup(self, target, source, stamp=None):
        """(R, t) mapping points in `source` to `target`, from TF; odom <-> base_link falls back to the Go2
        odometry message itself when nobody publishes that TF (sensors.launch.py / go2_odom_tf.py not running)."""
        try:
            tf = self.tf_buffer.lookup_transform(target, source, stamp or Time(), timeout=Duration(seconds=0.0))
            t, q = tf.transform.translation, tf.transform.rotation
            return quat_to_matrix(q), np.array([t.x, t.y, t.z])
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            pose = getattr(self, "odom_pose", None)
            if pose is None or {target, source} != {"odom", self.target_frame}:
                raise
            if not getattr(self, "warned_odom_fallback", False):
                self.get_logger().warn("no TF odom <-> base_link: using the odometry message instead "
                                       "(start sensors.launch.py to have the TF)")
                self.warned_odom_fallback = True
            R, t = pose
            return (R, t) if target == "odom" else (R.T, -R.T @ t)

    def slope_step(self, msg, xyz_sensor, stamp):
        """Step 7: plane / slope / step on this cloud (+ the clouds of the last slope_window s)."""
        t0 = time.perf_counter()
        frame = msg.header.frame_id
        res = {"profile": self.profile, "n_scans": 1, "roi_x": list(sc.SENSORS[self.profile]["roi_x"])}
        try:
            R_bs, t_bs = self.lookup(self.target_frame, frame)              # sensor -> base_link (latest)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            res.update(status="NO_TF", reason=f"no TF {self.target_frame} <- {frame}")
            return self.publish_slope(res, stamp, t0)
        if self.rpy is None:
            res.update(status="NO_IMU", reason="no /lowstate yet: the cloud cannot be levelled")
            return self.publish_slope(res, stamp, t0)
        xyz = xyz_sensor.astype(np.float64)
        if len(xyz) > self.max_points:                      # camera: ~70k points -> ~20k (RANSAC result unchanged)
            xyz = xyz[:: int(np.ceil(len(xyz) / self.max_points))]
        R, t = R_bs, t_bs
        if self.slope_window > 0:
            try:
                R_ob, t_ob = self.lookup("odom", self.target_frame)          # base_link -> odom now
                self.scans.append((stamp, sc.xform(sc.xform(xyz, R_bs, t_bs), R_ob, t_ob)))
                self.scans = [x for x in self.scans if x[0] > stamp - self.slope_window]
                xyz = np.concatenate([x[1] for x in self.scans])
                R, t = R_ob.T, -R_ob.T @ t_ob                                # odom -> base_link
                res["n_scans"] = len(self.scans)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                res["reason_window"] = "no odom TF: single scan"
        out, pts = sc.estimate(xyz, R, t, self.rpy[0], self.rpy[1], self.body_height, self.profile)
        res.update(out)
        return self.publish_slope(res, stamp, t0, pts)

    def publish_slope(self, res, stamp, t0, pts=None):
        st = res["status"]
        step_conf, obs_conf = self.confirm.update(res)                  # >= confirm_n consecutive estimates
        res.update(step_confirmed=int(step_conf is not None), n_obstacles_confirmed=len(obs_conf),
                   obstacles_confirmed=obs_conf)
        if step_conf:
            res.update(step_confirmed_dz_m=step_conf["dz_m"], step_confirmed_dist_m=step_conf["dist_m"])
        if obs_conf:
            res.update(obstacle_confirmed_dist_m=obs_conf[0]["dist_m"], obstacle_confirmed_height_m=obs_conf[0]["height_m"])
        if self.viz:
            now = self.get_clock().now().to_msg()
            cfg = sc.SENSORS[self.profile]
            pts = pts or {}
            groups = [(pts.get("roi", np.zeros((0, 3))), (150, 150, 150)), (pts.get("road", np.zeros((0, 3))), (60, 110, 255)),
                      (pts.get("inliers", np.zeros((0, 3))), (40, 220, 60))]
            groups += [(q, (240, 30, 30)) for q in pts.get("obstacles", [])]
            self.viz_cloud_pub.publish(tv.rgb_cloud(groups, now))
            self.viz_marker_pub.publish(tv.markers(res, pts, step_conf, obs_conf, cfg, self.body_height, now))
            if self.rpy is not None:            # terrain_level = base_link with the body roll/pitch removed
                R = sc.level_matrix(*self.rpy).T                  # orientation of terrain_level in base_link
                w = np.sqrt(max(1e-12, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
                tfm = TransformStamped()
                tfm.header.stamp, tfm.header.frame_id, tfm.child_frame_id = now, self.target_frame, tv.FRAME
                q = tfm.transform.rotation
                q.w, q.x, q.y, q.z = w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w), (R[1, 0] - R[0, 1]) / (4 * w)
                self.tf_pub.sendTransform(tfm)
        res["proc_ms"] = (time.perf_counter() - t0) * 1000.0
        def js(v):
            if isinstance(v, (float, np.floating)):
                return round(float(v), 5)
            if isinstance(v, np.integer):
                return int(v)
            if isinstance(v, dict):
                return {k: js(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [js(x) for x in v]
            return v
        clean = {k: js(v) for k, v in res.items()}
        self.slope_pub.publish(String(data=json.dumps(clean).replace("NaN", "null")))
        self.slope_deg_pub.publish(Float32(data=float(res.get("slope_deg", float("nan"))) if st == "OK" else float("nan")))
        self.slope_hist.append((st, res.get("slope_deg", float("nan")), res.get("step_confirmed_dz_m", float("nan")),
                                res["proc_ms"], res.get("n_obstacles_confirmed", 0),
                                res.get("obstacle_confirmed_height_m", float("nan")),
                                res.get("obstacle_confirmed_dist_m", float("nan"))))
        if st != self.last_slope_status:
            text = f"slope status {self.last_slope_status} -> {st}" + (f": {res.get('reason')}" if res.get("reason") else "")
            if st == "OK":
                self.get_logger().info(text)
            else:
                self.get_logger().warn(text)
            self.last_slope_status = st
        if getattr(self, "slope_csv", None):
            self.slope_csv.writerow([f"{stamp:.3f}"] + [("%.5g" % res[c]) if isinstance(res.get(c), (float, np.floating))
                                                         else res.get(c, "") for c in SLOPE_COLS])

    # ---------------------------------------------------------------- helpers
    def to_body(self, msg, xyz):
        """Step 4a: express points in target_frame. Returns (points, how)."""
        src = msg.header.frame_id
        if src == self.target_frame:
            return xyz, "none"
        try:
            # latest available transform; static sensor mounts do not depend on the stamp
            R, t = self.lookup(self.target_frame, src)
            return sc.xform(xyz, R, t), "tf"
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            if src.endswith("optical_frame"):
                return sc.xform(xyz, OPTICAL_TO_BODY), "optical_axes"
            return xyz, "raw"

    # ---------------------------------------------------------------- callback
    def on_cloud(self, msg):
        t0 = time.perf_counter()
        now = time.time()
        self.last_rx = now
        self.total_msgs += 1
        if self.warned_no_data:
            self.get_logger().info("data is back")
            self.warned_no_data = False
        if self.first_frame is None:
            self.first_frame = msg.header.frame_id
            self.get_logger().info(f"first cloud: frame {msg.header.frame_id}, {msg.width}x{msg.height}, "
                                   f"fields {[f.name for f in msg.fields]}, point_step {msg.point_step}")
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        age = self.get_clock().now().nanoseconds * 1e-9 - stamp
        frame = msg.header.frame_id
        dist = (float("nan"),) * 3
        detail = ""
        try:
            xyz = read_xyz(msg)                                     # 2
            n_total = len(xyz)
            finite = np.isfinite(xyz).all(axis=1)                   # 3
            xyz = xyz[finite]
            n_valid = len(xyz)
            pts, how = self.to_body(msg, xyz)                       # 4
            r = self.roi
            keep = ((pts[:, 0] >= r["x_min"]) & (pts[:, 0] <= r["x_max"]) & (np.abs(pts[:, 1]) <= r["y_abs"])
                    & (pts[:, 2] >= r["z_min"]) & (pts[:, 2] <= r["z_max"]))
            n_roi = int(keep.sum())
            if n_roi:                                               # 5: horizontal range from base_link
                rng = np.hypot(pts[keep, 0], pts[keep, 1])
                dist = (float(rng.min()), float(np.median(rng)), float(rng.max()))
            # 4b + 5: status, worst problem first
            if not frame:
                status, detail = "NO_FRAME", "empty header.frame_id"
            elif stamp == 0.0:
                status, detail = "BAD_STAMP", "header.stamp is zero"
            elif self.last_stamp is not None and stamp <= self.last_stamp:
                status, detail = "BAD_STAMP", f"stamp went back/repeated by {self.last_stamp - stamp:.3f} s"
            elif n_valid == 0:
                status = "NO_VALID"
            elif how not in ("tf", "none"):
                status, detail = "NO_TF", f"no TF {self.target_frame} <- {frame}, used {how}"
            elif n_roi < self.min_points:
                status, detail = "LOW_POINTS", f"{n_roi} < {self.min_points} points in front of the robot"
            elif self.max_age > 0 and abs(age) > self.max_age:     # max_age 0: off (Go2 L1 stamps use the Go2 clock)
                status, detail = "STALE", f"stamp is {age:+.2f} s from the ROS clock"
            else:
                status = "OK"
            if self.first_frame != frame:
                detail += f" frame changed {self.first_frame} -> {frame}"
                self.get_logger().warn(f"frame changed: {self.first_frame} -> {frame}")
                self.first_frame = frame
        except Exception as e:  # malformed cloud: count it, keep running
            self.errors += 1
            self.get_logger().error(f"cannot process cloud: {e}")
            n_total = n_valid = n_roi = 0
            how, status, detail = "-", "ERROR", str(e)
        self.last_stamp = stamp
        proc_ms = (time.perf_counter() - t0) * 1000.0
        self.window.append((now, n_total, n_valid, n_roi, status == "OK", proc_ms, dist[1]))
        self.status_count[status] = self.status_count.get(status, 0) + 1
        if status != self.last_status:                              # 6: log every status change at once
            text = f"status {self.last_status} -> {status}" + (f": {detail}" if detail else "")
            if status == "OK":   # Foxy: one log call site may not switch severity, so two separate calls
                self.get_logger().info(text)
            else:
                self.get_logger().warn(text)
            self.last_status = status
        out = {"stamp": round(stamp, 3), "frame_id": frame, "status": status, "detail": detail.strip(),
               "n_total": n_total, "n_valid": n_valid, "n_roi": n_roi, "transform": how,
               "dist_min": round(dist[0], 3), "dist_median": round(dist[1], 3), "dist_max": round(dist[2], 3),
               "age_s": round(age, 3)}
        self.status_pub.publish(String(data=json.dumps(out).replace("NaN", "null")))
        if self.csv:
            self.csv.writerow([f"{now:.3f}", f"{stamp:.3f}", frame, how, n_total, n_valid, n_total - n_valid,
                               n_roi, *(f"{d:.3f}" for d in dist), f"{age:.3f}", status, detail.strip(),
                               f"{proc_ms:.1f}"])
        self.last_how = how
        if self.do_slope and status not in ("ERROR", "NO_FRAME", "BAD_STAMP", "NO_VALID") and time.time() >= self.next_slope:
            self.next_slope = time.time() + 1.0 / self.slope_rate
            self.slope_step(msg, xyz, stamp)

    # ---------------------------------------------------------------- step 6: periodic status
    def summary(self):
        now = time.time()
        if self.csv_file:
            self.csv_file.flush()
        if not self.window:
            since = now - (self.last_rx or self.t_start)
            if since > self.timeout:
                self.get_logger().warn(f"NO DATA on {self.topic} for {since:.1f} s "
                                       f"(received {self.total_msgs} clouds so far)")
                self.warned_no_data = True
            return
        w = np.array([x[1:] for x in self.window], dtype=np.float64)
        d = w[:, 5][np.isfinite(w[:, 5])]
        n = len(w)
        span = now - self.window[0][0] if n > 1 else self.log_period
        rate = n / max(self.log_period, span)
        ok = int(w[:, 3].sum())
        msg = (f"{rate:4.1f} Hz | pts {w[:, 0].mean():.0f} | valid {100 * w[:, 1].sum() / max(w[:, 0].sum(), 1):.0f}% "
               f"| ROI {w[:, 2].mean():.0f} (min {w[:, 2].min():.0f}) "
               f"| dist {np.median(d) if len(d) else float('nan'):.2f} m | OK {ok}/{n} | {w[:, 4].mean():.1f} ms "
               f"| tf {getattr(self, 'last_how', '-')} | errors {self.errors}")
        if ok == n:
            self.get_logger().info(msg)
        else:
            self.get_logger().warn(msg + f"  <- {n - ok} not OK, last status {self.last_status}")
        self.window = []
        if self.slope_hist:
            h = self.slope_hist
            ok = [x for x in h if x[0] == "OK"]
            sl = np.array([x[1] for x in ok], float)
            steps = [x[2] for x in ok if np.isfinite(x[2])]
            obs = [x for x in ok if x[4]]
            txt = (f"slope [{self.profile}] OK {len(ok)}/{len(h)} | "
                   + (f"{np.median(sl):.2f}° ({100 * np.tan(np.radians(np.median(sl))):.1f} %) ± {sl.std():.2f}" if len(ok) else "–")
                   + (f" | STEP {100 * np.median(steps):+.1f} cm (confirmed)" if steps else "")
                   + (f" | OBSTACLE {100 * np.median([x[5] for x in obs]):.0f} cm @ {np.median([x[6] for x in obs]):.2f} m "
                      f"(confirmed, {max(x[4] for x in obs)} object(s))" if obs else "")
                   + f" | {np.mean([x[3] for x in h]):.0f} ms")
            if len(ok) == len(h):
                self.get_logger().info(txt)
            else:
                self.get_logger().warn(txt + f"  <- last status {self.last_slope_status}")
            self.slope_hist = []

    def destroy_node(self):
        if self.csv_file:
            self.csv_file.close()
        if getattr(self, "slope_file", None):
            self.slope_file.close()
        super().destroy_node()


def main():
    rclpy.init()
    node = TerrainGeometryNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.get_logger().info(f"stopped after {node.total_msgs} clouds, status counts {node.status_count}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
