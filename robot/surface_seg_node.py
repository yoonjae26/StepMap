#!/usr/bin/env python3
"""Live (camera-rate) geometric segmentation of the camera image: sunken / raised tiles, holes, obstacles, steps.

  python3 surface_seg_node.py                           # RealSense (sensors.launch.py running, camera looking down)
  python3 surface_seg_node.py --ros-args -p sim:=true   # Gazebo rig (sim/run_sim.sh, ROS_DOMAIN_ID=43)

Real time (2026-10-18; was ~2 fps):
  * every depth frame updates a per-pixel running average (EMA, alpha 0.25 ~ last 7 frames; a pixel whose depth jumps
    by > 3 cm restarts from the new value, so moving objects are not smeared) -> depth noise down ~2.6x
  * every depth frame is classified (surface_seg.segment), projected into the colour image and drawn: ~35 ms on the
    Jetson, i.e. the full camera rate (15 Hz). The Korean text panel is refreshed 5x per second.
  * the floor plane (slope_core.estimate: RANSAC, step, levelled by the body IMU) changes slowly and runs in a
    background thread ~4x per second on the averaged depth; every frame uses the latest plane.
Publishes /terrain_geometry/seg_image (rgb8, every frame, only while someone subscribes: seg_viewer.py) and
/terrain_geometry/seg_regions (JSON, 5 Hz: threshold, regions with kind, height_mm, area_m2, distance, fps).

Live picture for the dashboards (2026-10-26): every drawn frame is JPEG-encoded in a writer thread and written to
live_path (/dev/shm/terrain_live.jpg, atomic replace) + live_path .json (capture time, fps, latency). The StepMap live
server streams it (MJPEG). Before, the 1.2 MB image went through ROS to inspect_node, which encoded it: two Python
copies per frame and ~6 pictures/s. Camera topics are subscribed with queue depth 1: when the Jetson is busy the
newest frame is processed, never a backlog (lower latency). Regions are boxed in the StepMap level colours
(위험 >= danger_mm, 주의 >= caution_mm, the same thresholds as inspect_node and StepMap step_warn/danger_cm).

Damage AI (2026-10-26, damage_ai.py): a SegFormer trained on the AI Hub 서피스마스킹 pictures finds broken / cracked
paving in the colour picture (TensorRT on the GPU, ai_engine). It runs in its own thread on the newest frame
(ai_hz, default 5 Hz) so the 15 Hz geometry is not slowed down. Each damaged region is looked up in the depth
heights: raised / sunken >= ai_min_height_mm -> 파손 (height in cm), flat -> 균열 (score = AI probability).
Published on /terrain_geometry/ai_regions (JSON: regions with kind_ko, height_mm, score, x, y in the levelled robot
frame, odom pose of the frame) and drawn on the live picture (magenta outline + label). 요철 (unevenness): spread
(95th - 5th percentile) of the floor heights 0.4-1.2 m ahead, every frame, in seg_regions "uneven_mm" and the panel.
Without the engine file the node runs as before (geometry only).
Damage type (2026-11-06): with ai_types_engine (damage_types.engine, 8 classes) the type model runs on the same
picture, only when model A found a damaged region; each region gets type_ko 파손 / 결손 / 이탈·분리
(damage_ai.fuse_damage: depth step >= ai_step_mm at the region -> 이탈·분리, else the type model's 파손 / 결손).
Without that file, or ai_types:=false, regions carry no type_ko (as before).

Inputs: colour image + info, raw depth + info (16UC1 mm or 32FC1 m). The depth grid is projected into the colour
image with the driver's TF camera_depth_optical_frame -> camera_color_optical_frame (realsense-ros 4.0.4 here does not
publish aligned depth). camera -> base_link from TF (sensors.launch.py CAMERA_TF); body roll/pitch from /lowstate;
body height from /utlidar/robot_odom. sim:=true: Gazebo topics, the rig's fixed camera mount, level body.
"""
import array
import bisect
import collections
import json
import math
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray, String
import queue
import tf2_ros
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import slope_core as sc  # noqa: E402
import surface_seg as ss  # noqa: E402
import damage_ai as dai  # noqa: E402
from terrain_geometry_node import quat_to_matrix  # noqa: E402

KIND_KO = {ss.SUNKEN: "함몰", ss.HOLE: "구멍", ss.RAISED: "돌출", ss.OBSTACLE: "장애물"}


def to_np(m):
    if m.encoding in ("16UC1", "mono16"):
        return np.frombuffer(m.data, np.uint16).reshape(m.height, m.width).astype(np.float32) / 1000.0
    if m.encoding == "32FC1":
        return np.frombuffer(m.data, np.float32).reshape(m.height, m.width).copy()
    a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[:, :, :3]
    return a[:, :, ::-1].copy() if m.encoding.startswith("bgr") else a.copy()


class SegNode(Node):
    def __init__(self):
        super().__init__("surface_seg_node")
        p = self.declare_parameter
        self.sim = p("sim", False).value
        color = p("color_topic", "/camera/depth_camera/image_raw" if self.sim else "/camera/color/image_raw").value
        depth = p("depth_topic", "/camera/depth/image_rect_raw").value
        info = p("info_topic", "/camera/depth/camera_info").value
        cinfo = p("color_info_topic", "/camera/depth/camera_info" if self.sim else "/camera/color/camera_info").value
        self.alpha = p("ema_alpha", 0.25).value           # per-pixel running average of the depth
        self.reset_m = p("ema_reset_m", 0.03).value       # a jump larger than this restarts the pixel
        self.plane_hz = p("plane_hz", 4.0).value
        self.json_hz = p("json_hz", 5.0).value
        self.map_hz = p("map_hz", 3.0).value              # height samples for the accumulated map (inspect_node)
        self.map_cell = p("map_cell_m", 0.02).value
        self.map_max_x = p("map_max_x_m", 1.6).value      # farther pixels: depth too noisy for a cm map
        self.motion_comp = p("motion_comp", True).value   # walking: move the running average with the camera (odom)
        self.live_path = p("live_path", "/dev/shm/terrain_live.jpg").value   # dashboard picture ("" = off)
        self.live_quality = p("live_quality", 80).value
        # live picture: "damage" (StepMap: only the AI damage regions, type + height, level colours, 2026-11-13) or
        # "geometry" (all depth classes coloured: floor / sunken / raised / hole / obstacle, the earlier view)
        self.live_view = p("live_view", "damage").value
        self.levels = (p("caution_mm", 10.0).value / 1000.0, p("danger_mm", 20.0).value / 1000.0)
        self.frame_t = None                                # capture time of the depth frame being processed
        self.live_q = queue.Queue(maxsize=1)               # newest picture only: the writer never falls behind
        self.ai_path = os.path.expanduser(p("ai_engine", "~/terrain_dataset/models/damage_seg.engine").value)
        self.ai_hz = p("ai_hz", 5.0).value
        self.ai_thr = p("ai_thr", 0.5).value
        self.ai_min_h = p("ai_min_height_mm", 10.0).value
        self.types_path = os.path.expanduser(p("ai_types_engine", "~/terrain_dataset/models/damage_types.engine").value)
        self.use_types = p("ai_types", True).value
        self.ai_step_mm = p("ai_step_mm", 6.0).value
        self.types = None
        self.types_ms = 0.0
        self.ai_q = queue.Queue(maxsize=1)
        self.ai_draw = None                                # (outline mask, label bitmaps) of the last AI result
        self.ai_ms = 0.0
        self.target = "base_link"
        self.ema = None
        self.n_frames = 0
        self.rgb, self.K, self.Kc, self.optical, self.color_frame = None, None, None, None, None
        self.rpy, self.height = None, 0.30
        self.odom_pose = None                              # (R, t) odom <- base_link, latest /utlidar/robot_odom
        self.odom_hist = collections.deque(maxlen=300)     # (local time, R, t): pose at the depth frame's time
        self.clock_off = {"odom": collections.deque(maxlen=200), "depth": collections.deque(maxlen=200)}
        self.mount = None                                  # (R, t) base_link <- depth optical (static, cached)
        self.ema_pose = None                               # camera pose (odom <- optical) the running average is in
        self.warp_ms = 0.0
        self.frame_odom = None                             # (R, t) odom <- base_link when the depth frame was taken
        self.plane = None                                  # (result dict, planes) from the background thread
        self.lock = threading.Lock()
        self.rng = np.random.default_rng()                # sim: D435-like noise on the (perfect) Gazebo depth
        self.fps_t, self.fps, self.last_json = [], 0.0, 0.0
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        latest = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.create_subscription(Image, color, self.on_color, latest)
        self.create_subscription(Image, depth, self.on_depth, latest)
        self.create_subscription(CameraInfo, info, self.on_info, qos_profile_sensor_data)
        self.create_subscription(CameraInfo, cinfo, self.on_cinfo, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/utlidar/robot_odom", self.on_odom, qos_profile_sensor_data)
        try:
            from unitree_go.msg import LowState
            self.create_subscription(LowState, "/lowstate", self.on_low, qos_profile_sensor_data)
        except ImportError:
            pass
        self.img_pub = self.create_publisher(Image, "/terrain_geometry/seg_image", 2)
        self.reg_pub = self.create_publisher(String, "/terrain_geometry/seg_regions", 10)
        self.ai_pub = self.create_publisher(String, "/terrain_geometry/ai_regions", 10)
        self.map_pub = self.create_publisher(Float32MultiArray, "/terrain_geometry/height_samples", 5)
        self.last_map = 0.0
        self.t_start, self.warned = time.time(), set()
        self.running = True
        threading.Thread(target=self.plane_loop, daemon=True).start()
        if self.live_path:
            threading.Thread(target=self.live_loop, daemon=True).start()
        self.ai = None
        if os.path.isfile(self.ai_path):
            threading.Thread(target=self.ai_loop, daemon=True).start()
        else:
            self.get_logger().warn(f"no damage AI engine {self.ai_path}: geometry only (see damage_ai.py)")
        self.create_timer(5.0, self.report)
        self.get_logger().info(f"colour {color}, depth {depth}; per-pixel EMA alpha {self.alpha}, plane "
                               f"{self.plane_hz:g} Hz in the background, every depth frame segmented"
                               + (" [SIM]" if self.sim else ""))

    # ------------------------------------------------------------------ inputs
    def on_color(self, m):
        self.rgb = m
        self.color_frame = m.header.frame_id

    def on_cinfo(self, m):
        self.Kc = (m.k[0], m.k[4], m.k[2], m.k[5])

    def on_info(self, m):
        self.K = (m.k[0], m.k[4], m.k[2], m.k[5])

    def local_time(self, key, stamp):
        """Message stamp -> this computer's clock. Offset per topic = smallest (arrival - stamp) of the last messages.
        Stamps already on this clock (RealSense) are used as they are; the Go2 odometry has the robot's clock
        (minutes off); in Gazebo camera and odometry share the simulation clock -> the camera uses the odometry's
        offset (its own minimum would include the image transport delay and place the frame too late)."""
        st = stamp.sec + stamp.nanosec * 1e-9
        d = self.clock_off[key]
        d.append(time.time() - st)
        off = min(d)
        if abs(off) < 0.5:
            return st
        od = self.clock_off["odom"]
        if key == "depth" and od and abs(off - min(od)) < 1.0:
            off = min(od)
        return st + off

    def on_odom(self, m):
        o = m.pose.pose
        self.height = float(o.position.z)
        self.odom_pose = (quat_to_matrix(o.orientation), np.array([o.position.x, o.position.y, o.position.z]))
        self.odom_hist.append((self.local_time("odom", m.header.stamp), *self.odom_pose))

    def pose_at(self, t):
        """Odometry pose at local time t (position interpolated), latest if t is newer than the history."""
        h = self.odom_hist
        if not h:
            return self.odom_pose
        ts = [e[0] for e in h]
        i = bisect.bisect_left(ts, t)
        if i >= len(h):
            return h[-1][1], h[-1][2]
        if i == 0:
            return h[0][1], h[0][2]
        (ta, Ra, pa), (tb, Rb, pb) = h[i - 1], h[i]
        a = (t - ta) / max(1e-6, tb - ta)
        return (Ra if a < 0.5 else Rb), pa + a * (pb - pa)

    def on_low(self, m):
        self.rpy = (float(m.imu_state.rpy[0]), float(m.imu_state.rpy[1]))

    def warn_once(self, key, text):
        if key not in self.warned:
            self.get_logger().warn(text)
            self.warned.add(key)

    def on_depth(self, m):
        self.optical = m.header.frame_id
        z = to_np(m)
        if self.sim:
            z = z + (self.rng.standard_normal(z.shape).astype(np.float32) * (0.002 + 0.004 * np.nan_to_num(z) ** 2))
        z[~(z > 0)] = np.nan
        self.frame_t = self.local_time("depth", m.header.stamp)
        self.frame_odom = self.pose_at(self.frame_t) if self.odom_hist else self.odom_pose
        pose = self.cam_pose() if self.motion_comp else None
        with self.lock:
            if self.ema is None or self.ema.shape != z.shape:
                self.ema = z.copy()
            else:
                if pose is not None and self.ema_pose is not None:
                    self.ema = self.warp(self.ema, self.ema_pose, pose)
                e = self.ema
                upd = np.isfinite(z)
                restart = upd & (~np.isfinite(e) | (np.abs(z - e) > self.reset_m))   # new pixel / scene changed
                blend = upd & ~restart
                e[blend] += self.alpha * (z[blend] - e[blend])
                e[restart] = z[restart]
            self.ema_pose = pose
            self.n_frames += 1
            depth = self.ema.copy()
        self.frame_pose = pose
        self.process(depth)

    def cam_pose(self):
        """odom <- depth optical frame (R, t) from the latest odometry and the camera mount, or None."""
        if self.frame_odom is None or self.K is None or not self.optical:
            return None
        if self.mount is None:
            try:
                self.mount = self.camera_tf()
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                return None
        (R_ob, t_ob), (R_bo, t_bo) = self.frame_odom, self.mount
        return R_ob @ R_bo, R_ob @ t_bo + t_ob

    def warp(self, e, prev, cur):
        """Running average (depth image seen from camera pose `prev`) re-projected into the view from `cur`: while
        the robot walks the same floor point moves across the image; without this the average smears the tiles
        (~15 cm lag at 0.3 m/s). Pixels that receive nothing restart from the new frame."""
        (R0, t0), (R1, t1) = prev, cur
        R, t = R1.T @ R0, R1.T @ (t0 - t1)                 # prev optical -> cur optical
        if np.linalg.norm(t) < 0.002 and np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)) < 0.002:
            return e                                       # standing still: nothing to move
        t_w = time.perf_counter()
        P = ss.backproject(e, self.K).reshape(-1, 3)
        ok = np.isfinite(P[:, 2])
        Q = sc.xform(P[ok], R, t)
        fx, fy, cx, cy = self.K
        z = Q[:, 2]
        zs = np.where(z > 0.1, z, 1.0)
        u = (fx * Q[:, 0] / zs + cx + 0.5).astype(np.int32)
        v = (fy * Q[:, 1] / zs + cy + 0.5).astype(np.int32)
        H, W = e.shape
        g = (z > 0.1) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        out = np.full(e.shape, np.nan, np.float32)
        out[v[g], u[g]] = z[g]
        self.warp_ms = 0.9 * self.warp_ms + 0.1 * (time.perf_counter() - t_w) * 1000
        return out

    # ------------------------------------------------------------------ geometry
    def camera_tf(self):
        if self.sim:                                     # Gazebo rig mount (models/sensor_rig/model.sdf)
            optical = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
            return sc.rotation(0.0, math.radians(20.0), 0.0) @ optical, np.array([0.30, 0.0, 0.05])
        tf = self.tf_buffer.lookup_transform(self.target, self.optical, Time(), timeout=Duration(seconds=0.0))
        t, q = tf.transform.translation, tf.transform.rotation
        return quat_to_matrix(q), np.array([t.x, t.y, t.z])

    def posture(self):
        if self.rpy is None:
            if not self.sim:
                self.warn_once("imu", "no /lowstate: body assumed level (posture not corrected)")
            return 0.0, 0.0, (0.31 if self.sim else self.height)
        return self.rpy[0], self.rpy[1], self.height

    def plane_loop(self):
        """Background: floor plane / step on the averaged depth, plane_hz times per second."""
        while self.running:
            t0 = time.time()
            with self.lock:
                depth = None if self.ema is None or self.n_frames < 3 else self.ema.copy()
            if depth is not None and self.K is not None and self.optical:
                try:
                    R_bo, t_bo = self.camera_tf()
                    roll, pitch, height = self.posture()
                    pts = ss.backproject(depth, self.K).reshape(-1, 3)
                    pts = pts[np.isfinite(pts).all(1) & (pts[:, 2] > 0.2)][::8].astype(np.float64)
                    r, P = sc.estimate(pts, R_bo, t_bo, roll, pitch, height, "camera")
                    planes = self.smooth_planes(P.get("planes", []))
                    self.plane = (r, planes, R_bo, t_bo, roll, pitch)
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    self.warn_once("tf", f"no TF {self.target} <- {self.optical}: start sensors.launch.py (CAMERA_TF)")
                except Exception as e:  # noqa: BLE001  keep the thread alive
                    self.warn_once("plane", f"plane estimate failed: {e}")
            time.sleep(max(0.0, 1.0 / self.plane_hz - (time.time() - t0)))

    def smooth_planes(self, planes, a=0.35):
        """Running average of the floor plane(s) between RANSAC runs: RANSAC picks slightly different inliers each
        time, the plane wobbles by a few mm and pixels near the threshold change colour -> flicker."""
        prev = getattr(self, "_planes_s", None)
        if not planes or prev is None or len(prev) != len(planes):
            self._planes_s = [(np.array(m, float), xa, xb) for m, xa, xb in planes]
            return self._planes_s
        out = []
        for (m, xa, xb), (mp, xpa, xpb) in zip(planes, prev):
            m = np.array(m, float)
            if m[:3] @ mp[:3] < 0:
                m = -m
            ms = (1 - a) * mp + a * m
            ms /= np.linalg.norm(ms[:3])
            sm = lambda o, n: n if not np.isfinite(o) or not np.isfinite(n) else (1 - a) * o + a * n  # noqa: E731
            out.append((ms, sm(xpa, xa), sm(xpb, xb)))
        self._planes_s = out
        return out

    def to_color(self, depth, cls, regions, R_cd, t_cd, shape):
        """Project the classified depth pixels into the colour image (one pass + hole filling by dilation; the
        decimated depth grid is coarser than the colour image)."""
        P = sc.xform(ss.backproject(depth, self.K).reshape(-1, 3), R_cd, t_cd)   # depth optical -> colour optical
        fx, fy, cx, cy = self.Kc
        Z = P[:, 2]
        ok = (cls.ravel() > 0) & np.isfinite(Z) & (Z > 0.1)
        Zs = np.where(ok, Z, 1.0)
        u = (fx * P[:, 0] / Zs + cx + 0.5).astype(np.int32)
        v = (fy * P[:, 1] / Zs + cy + 0.5).astype(np.int32)
        H, W = shape
        ok &= (u >= 0) & (u < W) & (v >= 0) & (v < H)
        out = np.zeros(shape, np.uint8)
        out[v[ok], u[ok]] = cls.ravel()[ok]
        k = max(3, 2 * int(np.ceil(W / depth.shape[1])) - 1)
        dil = cv2.dilate(out, np.ones((k, k), np.uint8))
        out = np.where(out == 0, dil, out)
        u2, v2, okm = u.reshape(depth.shape), v.reshape(depth.shape), ok.reshape(depth.shape)
        self.depth_uv = (np.where(okm, u2, -1), np.where(okm, v2, -1))   # colour pixel of each depth pixel (AI fusion)
        new = []
        for g in regions:                                            # bounding boxes in colour pixels
            x0, y0, x1, y1 = g["bbox"]
            sl = (slice(y0, y1 + 1), slice(x0, x1 + 1))
            sel = okm[sl] & (cls[sl] == g["kind"])
            if not sel.any():
                continue
            us, vs = u2[sl][sel], v2[sl][sel]
            new.append(dict(g, bbox=[int(us.min()), int(vs.min()), int(us.max()), int(vs.max())]))
        return out, new

    # ------------------------------------------------------------------ per frame
    def process(self, depth):
        if self.rgb is None or self.K is None:
            if time.time() - self.t_start > 5:
                self.warn_once("wait", f"waiting for camera data: colour {'ok' if self.rgb else 'missing'}, "
                                       f"depth info {'ok' if self.K else 'missing'}")
            return
        plane = self.plane
        if plane is None:
            return
        t0 = time.perf_counter()
        r, planes, R_bo, t_bo, roll, pitch = plane
        rgb = to_np(self.rgb)
        res = {"status": r["status"], "reason": r.get("reason", ""), "slope_deg": r.get("slope_deg")}
        if r["status"] == "OK":
            R_lb = sc.level_matrix(roll, pitch)
            cls, h, thr, regions = ss.segment(depth, self.K, R_lb @ R_bo, R_lb @ t_bo, planes,
                                              sc.SENSORS["camera"]["roi_x"], thr_in=getattr(self, "thr_s", None))
            if self.frame_odom is not None and time.time() - self.last_map >= 1.0 / self.map_hz:
                self.last_map = time.time()
                self.publish_map(depth, cls, h)
            cls_d = cls                                 # classes on the depth grid (h, X, Y are on it too)
            tm = ss.segment.last_thr_measured           # noise threshold smoothed over ~2 s (no colour flicker)
            self.thr_s = tm if getattr(self, "thr_s", None) is None else 0.95 * self.thr_s + 0.05 * tm
            same_grid = self.sim or (rgb.shape[:2] == depth.shape and self.color_frame == self.optical)
            if not same_grid:
                if self.Kc is None:
                    self.warn_once("cinfo", "waiting for the colour camera_info")
                    return
                try:
                    tf = self.tf_buffer.lookup_transform(self.color_frame, self.optical, Time(),
                                                         timeout=Duration(seconds=0.0))
                except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                    self.warn_once("ctf", f"no TF {self.color_frame} <- {self.optical} (RealSense driver)")
                    return
                R_cd = quat_to_matrix(tf.transform.rotation)
                tt = tf.transform.translation
                cls, regions = self.to_color(depth, cls, regions, R_cd, np.array([tt.x, tt.y, tt.z]), rgb.shape[:2])
            else:
                vv, uu = np.indices(depth.shape)
                self.depth_uv = (np.where(cls_d > 0, uu, -1), np.where(cls_d > 0, vv, -1))
            X, Y = ss.segment.last_xy
            uneven = dai.unevenness(h, cls_d, X, (ss.FLOOR, ss.SUNKEN, ss.RAISED))
            if self.ai_hz > 0 and self.ai_path and not self.ai_q.full():
                self.ai_q.put_nowait((rgb, self.depth_uv, h, X, Y, self.frame_odom, self.frame_t))
            if self.live_view == "damage":
                d = self.ai_draw if self.ai_draw is not None and time.time() - self.ai_draw[-1] <= 1.0 else None
                img = ss.damage_overlay(rgb, d, thr, r["slope_deg"], uneven, levels=self.levels)
            else:
                img = ss.overlay(rgb, cls, regions, thr, r["slope_deg"], levels=self.levels, uneven_mm=uneven)
                self.draw_ai(img)
            res.update(thr_mm=1000 * thr, regions=[dict(g, kind_ko=KIND_KO.get(g["kind"], "")) for g in regions],
                       uneven_mm=None if uneven is None else round(uneven, 1))
        else:                                           # same image size + panel, with the reason
            why = f"측정 불가: {r['status']} — {r.get('reason', '')}"[:60]
            thr0 = getattr(self, "thr_s", None) or 0.004
            if self.live_view == "damage":
                img = ss.damage_overlay(rgb, None, thr0, note=why)
            else:
                img = ss.overlay(rgb, np.zeros(rgb.shape[:2], np.uint8), [], thr0, note=why)
        now = time.time()
        self.fps_t = [t for t in self.fps_t if t > now - 2.0] + [now]
        self.fps = (len(self.fps_t) - 1) / max(1e-3, self.fps_t[-1] - self.fps_t[0]) if len(self.fps_t) > 1 else 0.0
        res.update(proc_ms=(time.perf_counter() - t0) * 1000, fps=self.fps, stamp=now)
        if self.frame_odom is not None:                 # robot pose of this frame (odom): inspect_node places events
            R_ob, t_ob = self.frame_odom
            res.update(odom_x=float(t_ob[0]), odom_y=float(t_ob[1]), odom_yaw=float(math.atan2(R_ob[1, 0], R_ob[0, 0])))
        if self.live_path:
            try:                                        # replace a picture the writer has not taken yet
                self.live_q.get_nowait()
            except queue.Empty:
                pass
            self.live_q.put_nowait((img, self.frame_t, res["proc_ms"], self.fps))
        if self.img_pub.get_subscription_count() > 0:   # seg_viewer / rosbag; nobody else needs the raw image
            out = Image()
            out.header.stamp, out.header.frame_id = self.get_clock().now().to_msg(), self.optical or ""
            out.height, out.width, out.encoding, out.step = img.shape[0], img.shape[1], "rgb8", 3 * img.shape[1]
            # Foxy: assigning bytes to a uint8[] field checks every byte (183 ms for 1.2 MB); array.array is 5 ms
            out.data = array.array("B", np.ascontiguousarray(img, np.uint8).tobytes())
            self.img_pub.publish(out)
        if now - self.last_json >= 1.0 / self.json_hz:
            self.last_json = now
            self.reg_pub.publish(String(data=json.dumps(res, default=float).replace("NaN", "null")))
        self.last_res = res

    def live_loop(self):
        """Writer thread: newest drawn frame -> JPEG (OpenCV releases the GIL while encoding) -> live_path, and its
        timing -> live_path .json: capture (camera time on this clock), written, latency_ms (capture -> file), fps."""
        meta_path = os.path.splitext(self.live_path)[0] + ".json"
        while self.running:
            try:
                img, t_cap, proc_ms, fps = self.live_q.get(timeout=1.0)
            except queue.Empty:
                continue
            ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(img[:, :, ::-1]),
                                   [cv2.IMWRITE_JPEG_QUALITY, int(self.live_quality)])
            if not ok:
                continue
            tmp = self.live_path + ".tmp"
            with open(tmp, "wb") as fh:
                fh.write(buf.tobytes())
            os.replace(tmp, self.live_path)
            now = time.time()
            if t_cap is not None:
                self.lat_ms = 0.9 * getattr(self, "lat_ms", 1000 * (now - t_cap)) + 0.1 * 1000 * (now - t_cap)
            meta = {"capture": t_cap, "written": now, "fps": round(fps, 1), "proc_ms": round(proc_ms, 1),
                    "latency_ms": None if t_cap is None else round(1000 * (now - t_cap), 1),
                    "width": int(img.shape[1]), "height": int(img.shape[0]), "bytes": len(buf)}
            with open(meta_path + ".tmp", "w") as fh:
                json.dump(meta, fh)
            os.replace(meta_path + ".tmp", meta_path)

    def ai_loop(self):
        """Damage AI thread: newest frame -> probabilities (TensorRT) -> damaged regions + depth -> 파손 / 균열."""
        try:
            self.ai = dai.TrtModel(self.ai_path)
            self.get_logger().info(f"damage AI {self.ai_path}: input {self.ai.in_hw}, {self.ai_hz:g} Hz")
        except Exception as e:  # noqa: BLE001  geometry keeps running without it
            self.get_logger().error(f"damage AI not loaded ({e}): geometry only")
            return
        if self.use_types and os.path.isfile(self.types_path):
            try:
                self.types = dai.TrtModel(self.types_path)
                self.get_logger().info(f"damage type model {self.types_path}: 파손 / 결손 / 이탈·분리")
            except Exception as e:  # noqa: BLE001  model A keeps running without it
                self.get_logger().error(f"damage type model not loaded ({e}): no damage types")
        elif self.use_types:
            self.get_logger().warn(f"no damage type engine {self.types_path}: no damage types")
        last = 0.0
        while self.running:
            try:
                rgb, (u, v), h, X, Y, odom, t_cap = self.ai_q.get(timeout=1.0)
            except queue.Empty:
                continue
            wait = 1.0 / self.ai_hz - (time.time() - last)
            if wait > 0:
                time.sleep(wait)
            last = time.time()
            t0 = time.perf_counter()
            prob = self.ai(rgb)
            mask, pd = dai.damage_mask(prob, rgb.shape[:2], self.ai_thr)
            ptype = None
            if self.types is not None and mask.any():          # type model only when A found damage
                t1 = time.perf_counter()
                ptype = self.types(rgb)
                self.types_ms = 0.8 * self.types_ms + 0.2 * (time.perf_counter() - t1) * 1000
            found = dai.fuse_damage(mask, pd, u, v, h, X, Y, self.ai_min_h, ptype=ptype, step_mm=self.ai_step_mm)
            self.ai_ms = 0.8 * self.ai_ms + 0.2 * (time.perf_counter() - t0) * 1000
            if self.live_view == "damage":
                self.ai_draw = (dai.fuse_damage.labels, found, ss.damage_tags(found, self.levels), time.time())
            else:
                self.ai_draw = (mask, ss.ai_labels(found), time.time())
            out = {"stamp": t_cap, "ai_ms": round(self.ai_ms, 1), "types_ms": round(self.types_ms, 1),
                   "types": self.types is not None, "regions": found}
            if odom is not None:
                R_ob, t_ob = odom
                out.update(odom_x=float(t_ob[0]), odom_y=float(t_ob[1]), odom_yaw=float(math.atan2(R_ob[1, 0], R_ob[0, 0])))
            self.ai_pub.publish(String(data=json.dumps(out, default=float).replace("NaN", "null")))

    def draw_ai(self, img):
        """Outline of the last AI result (< 1 s old) on the picture part of img, with its pre-rendered labels."""
        d = self.ai_draw
        if d is None or time.time() - d[2] > 1.0:
            return
        mask, labels, _ = d
        H, W = mask.shape
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img[:H, :W], cnts, -1, ss.AI_RGB, 3)
        for (x, y), bmp in labels:
            h_, w_ = bmp.shape[:2]
            x, y = int(min(max(0, x), W - w_)), int(min(max(0, y - h_), H - h_))
            img[y:y + h_, x:x + w_] = bmp

    def publish_map(self, depth, cls, h):
        """Heights above the local road surface of this frame, averaged per map cell (map_cell_m) in the odom
        frame: [x, y, mean h, weight] per cell. Weight = 1 / depth-noise variance (D435: ~2 mm + 4 mm * range^2),
        so near views count more than far ones when inspect_node adds the frames up."""
        X, Y = ss.segment.last_xy
        m = (cls > 0) & np.isfinite(h) & (X < self.map_max_x) & (h > -0.3) & (h < 0.5)
        m[1::2, :] = False                                # every 2nd row / column is plenty for 2 cm cells
        m[:, 1::2] = False
        if m.sum() < 100:
            return
        R_ob, t_ob = self.frame_odom
        yaw = math.atan2(R_ob[1, 0], R_ob[0, 0])
        c, s_ = math.cos(yaw), math.sin(yaw)
        x, y, hh, z = X[m], Y[m], h[m], depth[m]
        wx, wy = t_ob[0] + c * x - s_ * y, t_ob[1] + s_ * x + c * y
        w = 1.0 / (0.002 + 0.004 * z * z) ** 2
        ix, iy = np.floor(wx / self.map_cell).astype(np.int64), np.floor(wy / self.map_cell).astype(np.int64)
        keys, inv = np.unique((ix << 32) + (iy & 0xFFFFFFFF), return_inverse=True)
        ws = np.bincount(inv, weights=w)
        hm = np.bincount(inv, weights=w * hh) / ws
        kx, ky = keys >> 32, (keys & 0xFFFFFFFF).astype(np.int64)
        ky[ky >= 2 ** 31] -= 2 ** 32
        out = np.stack([(kx + 0.5) * self.map_cell, (ky + 0.5) * self.map_cell, hm, ws], 1).astype(np.float32)
        self.map_pub.publish(Float32MultiArray(data=array.array("f", out.ravel().tobytes())))

    def report(self):
        res = getattr(self, "last_res", None)
        if not res:
            return
        txt = f"{res['fps']:.1f} fps, {res['proc_ms']:.0f} ms/frame" + (f", warp {self.warp_ms:.0f} ms" if self.warp_ms else "")
        if self.ai is not None:
            txt += f", damage AI {self.ai_ms:.0f} ms"
        if getattr(self, "lat_ms", None) is not None:
            txt += f", camera -> dashboard picture {self.lat_ms:.0f} ms"
        if res.get("regions"):
            g = res["regions"][0]
            txt += (f" | nearest: {g['kind_ko']} {g['height_mm'] / 10:+.1f} cm, {100 * g['front_dist_m']:.0f} cm "
                    f"from the robot front")
        self.get_logger().info(txt)


def main():
    rclpy.init()
    n = SegNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.running = False
        n.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
