#!/usr/bin/env python3
"""Loose paving blocks from the feet -> StepMap live (kind "loose" + foot_traces), StepMap plan 10.3 / 10.4.

  source ~/terrain_dataset/env.sh
  python3 foot_node.py --ros-args -p session:=<same session as stepmap_bridge> [-p loose_warn_mm:=5.0]

Inputs: /lowstate (joints, foot force, IMU; every 2nd message is used = 250 Hz), /utlidar/robot_odom (only to place a
block on the map, never for heights), /dev/shm/terrain_live.jpg + .json (surface_seg_node: camera pictures for the
photo of the block).
Per step (foot_core.FootAnalyzer): foot height relative to the diagonal partner foot, self-calibrated on normal steps,
extra sink after touch-down vs the same leg's normal steps. Per block (foot_core.BlockTracker, 0.3 m): reported when
steps of >= 2 different feet sank >= loose_warn_mm (default 5) and they are >= half of the steps on it. Sent to the StepMap live server through the
same outbox as stepmap_bridge (spool <session>/foot): event key "loose:<run>:<block>", value = extra sink [mm] of the
first over-threshold step, source "foot", leg; plus that step's and later steps' foot_traces (id empty + t_event; the
live server links them, steps of other feet on the same block within 2 s -> "확정").
Photo (10.4): the block is not visible in the picture taken when the foot is on it (the camera looks 0.3-1.5 m ahead),
so the foot position is projected into a picture from 0.5-3 s before touch-down (odom between the two times, camera
mount sensors.launch.py CAMERA_TF, colour intrinsics) -> image + image_mark "u,v,r" (dashed circle "추정").
"""
import collections
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.expanduser("~/go2_deploy"), os.path.expanduser("~/stepmap_live/stepmap")]
import foot_core as fc  # noqa: E402
from stepmap_bridge import Outbox  # noqa: E402

HOME = Path.home()
# base_link -> camera_link (sensors.launch.py CAMERA_TF: x y z yaw pitch roll), colour optical = REP-103 rotation
CAMERA_TF = (0.30, 0.0, 0.067, -0.0099, 0.4541, -0.0157)
K_COLOR = (605.405, 604.793, 323.882, 244.267)          # fx fy cx cy, 640x480 (camera_info_color.json)
IMG_W, IMG_H = 640, 480


def rot_zyx(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


R_BC = rot_zyx(CAMERA_TF[3], CAMERA_TF[4], CAMERA_TF[5])                 # base <- camera_link
T_BC = np.array(CAMERA_TF[:3])
R_CO = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)             # camera_link <- optical (z fwd, x right, y down)


def project(p_base):
    """Point in the base frame -> colour pixel (u, v, depth) or None if behind the camera."""
    p = R_CO.T @ (R_BC.T @ (p_base - T_BC))
    if p[2] < 0.1:
        return None
    fx, fy, cx, cy = K_COLOR
    return fx * p[0] / p[2] + cx, fy * p[1] / p[2] + cy, p[2]


class FootNode(Node):
    def __init__(self):
        super().__init__("foot_node")
        P = self.declare_parameter
        self.session = P("session", time.strftime("%Y%m%d_%H%M%S")).value
        server = P("server", "http://127.0.0.1:8060").value
        policy = P("policy", str(HOME / "go2_deploy" / "policy_run9.npz")).value   # only its leg kinematics
        # detection threshold 5 mm: recorded firm-floor walks gave false blocks at 3-4 mm (kinematic noise ~4-5 mm),
        # none at 5 mm while a simulated 6 mm sink was found. Lower it after the sponge-board test (StepMap 10.3);
        # the StepMap levels (주의 3 mm / 위험 6 mm) still grade the value.
        warn = P("loose_warn_mm", 5.0).value
        self.decimate = P("decimate", 2).value                                     # 500 Hz /lowstate -> 250 Hz
        self.live = P("live_path", "/dev/shm/terrain_live.jpg").value
        self.ana = fc.FootAnalyzer(policy, loose_warn_mm=warn)
        self.blocks = fc.BlockTracker(warn_mm=warn, min_legs=P("min_legs", 2).value, min_frac=P("min_frac", 0.5).value)
        self.out = Outbox(str(HOME / "stepmap_sessions" / self.session / "foot"), server, self.get_logger())
        self.run = time.strftime("%H%M%S")
        self.odom = collections.deque(maxlen=2000)            # (t, x, y, yaw)
        self.frames = collections.deque(maxlen=40)            # (t_capture, jpeg bytes, odom pose at capture)
        self.n_low = 0
        self.sent_traces = set()
        self.stats = {"steps": 0, "ready": 0, "over": 0, "events": 0}
        from unitree_go.msg import LowState
        self.create_subscription(LowState, "/lowstate", self.on_low, qos_profile_sensor_data)
        self.create_subscription(Odometry, "/utlidar/robot_odom", self.on_odom, qos_profile_sensor_data)
        self.create_timer(0.2, self.grab_frame)
        self.create_timer(10.0, self.report)
        self.get_logger().info(f"session {self.session} -> {server}; loose block: >= {warn} mm extra sink under >= 2 different feet")

    def on_odom(self, m):
        p, q = m.pose.pose.position, m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.odom.append((time.time(), p.x, p.y, yaw))

    def pose_at(self, t):
        if not self.odom:
            return None
        best = min(self.odom, key=lambda o: abs(o[0] - t))
        return best if abs(best[0] - t) < 0.5 else None

    def grab_frame(self):
        """Keep the camera pictures of the last ~8 s (5 per second) with the robot pose when each was taken."""
        try:
            meta = json.loads(Path(self.live).with_suffix(".json").read_text())
            t_cap = float(meta.get("capture") or meta["written"])
            if self.frames and self.frames[-1][0] == t_cap:
                return
            data = Path(self.live).read_bytes()
        except (OSError, ValueError, KeyError):
            return
        pose = self.pose_at(t_cap)
        if pose is not None:
            self.frames.append((t_cap, data, pose, int(meta.get("height", IMG_H))))

    def on_low(self, m):
        self.n_low += 1
        if self.n_low % self.decimate:
            return
        q = [m.motor_state[i].q for i in range(12)]
        res = self.ana.update(time.time(), q, list(m.foot_force)[:4], list(m.imu_state.quaternion))
        for r in res:
            self.step(r)

    def step(self, r):
        self.stats["steps"] += 1
        if not r.get("ready"):
            return
        self.stats["ready"] += 1
        pose = self.pose_at(r["t_event"])
        if pose is None:
            return
        _, ox, oy, yaw = pose
        fx, fy = r["foot_body"][0], r["foot_body"][1]
        x, y = ox + math.cos(yaw) * fx - math.sin(yaw) * fy, oy + math.sin(yaw) * fx + math.cos(yaw) * fy
        if r["loose"]:
            self.stats["over"] += 1
        hit = self.blocks.add(r, x, y)
        if hit is None:
            return
        key, b = hit
        first = b["first"]
        ev_key = f"loose:{self.run}:{key}"
        new = ev_key not in self.sent_traces
        if new:
            image, data, mark = self.photo(first, b["x"], b["y"])
            row = {"t": round(first["t_event"], 3), "session": self.session, "x": round(b["x"], 3), "y": round(b["y"], 3),
                   "kind": "loose", "value": round(first["extra_mm"], 2), "unit": "mm", "source": "foot",
                   "leg": first["leg"], "confidence": 0.8, "image": image, "image_mark": mark,
                   "note": f"발 접촉: 추가 침하 {first['extra_mm']:.1f} mm ({len(b['steps'])}걸음 중 "
                           f"{sum(s['extra_mm'] >= self.blocks.warn for s in b['steps'])}걸음 기준 이상)"}
            self.out.put({"type": "event", "key": ev_key, "row": row}, data)
            self.stats["events"] += 1
            self.get_logger().info(f"loose block {key}: {row['note']} at odom ({b['x']:.2f}, {b['y']:.2f})")
        for s in b["steps"]:                       # this block's over-threshold steps, each sent once
            tk = f"{s['leg']}@{s['t_event']:.3f}"
            if s["extra_mm"] >= self.blocks.warn and tk not in self.sent_traces:
                self.sent_traces.add(tk)
                self.out.put({"type": "trace", "key": f"foot:{tk}", "rows": s["trace_rows"]})
        self.sent_traces.add(ev_key)

    def photo(self, step, x, y):
        """Picture from before the foot reached the block, with the block marked: (path, jpeg, "u,v,r") or blanks."""
        t1 = step["t_event"]
        best = None
        for t0, data, (tp, ox, oy, yaw), h_img in self.frames:
            if not 0.5 <= t1 - t0 <= 3.0:
                continue
            dx, dy = x - ox, y - oy                                # block in the base frame at the picture time
            pb = np.array([math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy,
                           step["foot_body"][2]])
            uv = project(pb)
            if uv is None:
                continue
            u, v, d = uv
            if not (0.05 * IMG_W < u < 0.95 * IMG_W and 0.05 * IMG_H < v < 0.95 * IMG_H):
                continue
            score = abs(v - 0.7 * IMG_H)                           # prefer the block in the lower part (close view)
            if best is None or score < best[0]:
                r_px = fc.BlockTracker().block_m / 2 * K_COLOR[0] / d
                best = (score, data, u / IMG_W, v / h_img, min(1.0, r_px / min(IMG_W, h_img)))
        if best is None:
            return "", None, ""
        _, data, u, v, r = best
        name = f"img/{self.session}_loose_{int(t1 * 10) % 1000000}.jpg"
        return name, data, f"{u:.3f},{v:.3f},{max(0.02, r):.3f}"

    def report(self):
        s = self.stats
        self.get_logger().info(f"steps {s['steps']} (analysed {s['ready']}), over threshold {s['over']}, "
                               f"loose blocks sent {s['events']}")


def main():
    rclpy.init()
    n = FootNode()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown() if hasattr(rclpy, "try_shutdown") else rclpy.shutdown()


if __name__ == "__main__":
    main()
