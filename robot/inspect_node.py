#!/usr/bin/env python3
"""Sidewalk inspection (A1): camera measurements -> "suspect sections" (의심 구간) with a place on the robot's path.

  python3 inspect_node.py [--ros-args -p caution_mm:=10.0 -p danger_mm:=20.0 -p session:=NAME]
  (inspect.sh starts it with surface_seg_node.py and the dashboard)

Input: /terrain_geometry/seg_regions (surface_seg_node.py, 5 Hz: sunken / raised / hole regions with their position
x, y in the levelled robot frame and the robot odometry pose of that frame), /utlidar/robot_odom (path, altitude,
roll/pitch), /lowstate (foot force, if present). Snapshot of each event: the live picture surface_seg_node writes
(live_path, every frame) is copied — no 1.2 MB image over ROS (snap_source:=topic for the old way).

Events: a region is turned into a place in the odom frame (robot pose + region x, y rotated by the robot yaw) and
merged with an earlier candidate of the same direction (down = sunken/hole, up = raised) closer than merge_m.
A candidate becomes an event after confirm_n observations (one region in one frame cannot make an event: noise);
position, height and area are medians over the views that saw (nearly) the whole region (a tile leaving the bottom
of the image is only partly seen: smaller, centre shifted), the closer half of those for the height.
Severity from |height| — PROVISIONAL thresholds (no official ones yet), parameters:
  경미 (minor) |h| < caution_mm   주의 (caution) caution_mm <= |h| < danger_mm   위험 (danger) |h| >= danger_mm
Only 주의 / 위험 count as suspect sections; holes (<= -3 cm) are always 위험.

Damage AI and unevenness (2026-10-26): /terrain_geometry/ai_regions (surface_seg_node + damage_ai.py) gives 파손
(broken block, height from the depth) and 균열 (crack, flat, AI score) regions; seg_regions "uneven_mm" gives 요철
(spread of the floor heights 0.4-1.2 m ahead). They become events the same way (same kind closer than merge_m,
confirm_n views, picture of the closest view): 파손 / 요철 graded by |height| like the geometry (caution_mm /
danger_mm), 균열 by the AI score (crack_caution 0.5 / crack_danger 0.8, StepMap crack_warn / crack_danger).
state.json events: "kind" 함몰 / 돌출 / 구멍 / 파손 / 균열 / 요철, h_mm (or score for 균열).
Damage type (2026-11-06): AI regions may carry type_ko 파손 / 결손 / 이탈·분리 (damage_types.engine); each AI event
keeps a vote per type over its views and reports the most frequent one as "type_ko" (+ "type_votes").
Close-up re-measurement (2026-11-08, patrol_node.py): /inspect/closeup {"id": n, "action": "start"|"end"} while the
robot stands still in front of event n. Between start and end every observation of the same kind near the event is
collected; at "end" the event gets "close" = {h_mm or score, n, dist_m, level, verdict, photo} from their median,
its snapshot becomes the close-up picture snap/E<n>_close.jpg and its level the close-up level (the level from
walking past stays in "level_pass"). verdict: 확인 (still >= caution) / 오탐 의심 (below caution) / 재측정 실패
(fewer than 3 views, e.g. the spot is under the camera's lower edge). Confirmed events are also published on
/inspect/events (JSON list, same as state.json "events") for the patrol.

Output: logs/inspect/<session>/
  state.json     live state for the dashboard, rewritten every second (path, events, last 60 s of sensors, status)
  events.csv     one line per confirmed event (updated when an event changes)
  sensors.csv    10 Hz log: time, odom x y z, roll, pitch, foot force x4
  heightmap.npz  accumulated height map (A2, heightmap.py): 2 cm cells, height above the local road surface
  report.html    field report (A4, report.py), written when the node stops
  /dev/shm/terrain_live.jpg  latest camera picture: written by surface_seg_node.py (live_hz > 0: by this node, old way)
  snap/E<n>.jpg  camera picture (coloured segmentation) of each event, closest view that shows all of it
"""
import csv
import json
import math
import os
import signal
import sys
import time

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray, String

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import surface_seg as ss  # noqa: E402
import heightmap as hm  # noqa: E402

LEVEL_KO = ["경미", "주의", "위험"]
DIR_KO = {"down": "함몰", "up": "돌출", "hole": "구멍"}


def yaw_rp(q):
    """Quaternion -> yaw, roll, pitch [rad]."""
    w, x, y, z = q.w, q.x, q.y, q.z
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return yaw, roll, pitch


class Inspector(Node):
    def __init__(self):
        super().__init__("inspect_node")
        p = self.declare_parameter
        self.caution_mm = p("caution_mm", 10.0).value
        self.danger_mm = p("danger_mm", 20.0).value
        self.merge_m = p("merge_m", 0.30).value          # same place if closer than this
        self.confirm_n = p("confirm_n", 3).value
        self.max_dist_m = p("max_dist_m", 1.6).value      # ignore regions farther (depth too noisy for mm)
        self.min_area_m2 = p("min_area_m2", 0.003).value
        self.track_step_m = p("track_step_m", 0.05).value
        self.location = p("location", "").value          # place name for the report (no GPS yet)
        session = p("session", "").value or time.strftime("%Y%m%d_%H%M%S")
        self.dir = os.path.join(p("out_dir", os.path.join(HERE, "logs", "inspect")).value, session)
        os.makedirs(os.path.join(self.dir, "snap"), exist_ok=True)
        self.session, self.t0 = session, time.time()
        self.cands, self.next_id = [], 1
        self.track, self.dist_m = [], 0.0                 # [(t, x, y)], walked distance
        self.pose = None                                  # (t, x, y, z, yaw, roll, pitch)
        self.ff = None                                    # foot force (4), raw sensor units
        self.ff_t = 0.0
        self.sens = []                                    # last 60 s: (t, x, y, z, roll, pitch, ff0..3)
        self.seg = {"t": 0.0, "fps": 0.0, "status": "", "thr_mm": None}
        self.img, self.img_t = None, 0.0
        self.live_hz = p("live_hz", 0.0).value           # > 0: also write the dashboard picture here (old way)
        self.snap_source = p("snap_source", "live").value  # "live": copy live_path, "topic": /terrain_geometry/seg_image
        self.live_path, self.live_t = p("live_path", "/dev/shm/terrain_live.jpg").value, 0.0
        self.hmap, self.hmap_n, self.hmap_saved = hm.HeightMap(p("map_cell_m", 0.02).value), 0, 0.0
        self.create_subscription(String, "/terrain_geometry/seg_regions", self.on_regions, 10)
        self.create_subscription(String, "/terrain_geometry/ai_regions", self.on_ai, 10)
        self.crack_levels = (p("crack_caution", 0.5).value, p("crack_danger", 0.8).value)
        self.uneven_mm = p("uneven_warn_mm", 10.0).value       # 요철 candidate from this spread on
        self.acands = []                                        # AI / unevenness candidates
        self.create_subscription(Odometry, "/utlidar/robot_odom", self.on_odom, qos_profile_sensor_data)
        if self.live_hz > 0 or self.snap_source == "topic":
            self.create_subscription(Image, "/terrain_geometry/seg_image", self.on_image, qos_profile_sensor_data)
        self.create_subscription(Float32MultiArray, "/terrain_geometry/height_samples", self.on_samples, 5)
        try:
            from unitree_go.msg import LowState
            self.create_subscription(LowState, "/lowstate", self.on_low, qos_profile_sensor_data)
        except ImportError:
            pass
        self.sens_f = open(os.path.join(self.dir, "sensors.csv"), "w")
        self.sens_f.write("t,x,y,z,roll_deg,pitch_deg,ff_fr,ff_fl,ff_rr,ff_rl\n")
        self.close = None                                 # running close-up (patrol_node)
        self.ev_pub = self.create_publisher(String, "/inspect/events", 10)
        self.create_subscription(String, "/inspect/closeup", self.on_closeup, 10)
        self.create_timer(0.1, self.sample)
        self.create_timer(1.0, self.write_state)
        self.get_logger().info(f"session {session} -> {self.dir}; thresholds (provisional) caution "
                               f"{self.caution_mm:g} mm, danger {self.danger_mm:g} mm")

    # ------------------------------------------------------------------ inputs
    def on_odom(self, m):
        o = m.pose.pose
        yaw, roll, pitch = yaw_rp(o.orientation)
        self.pose = (time.time(), o.position.x, o.position.y, o.position.z, yaw, roll, pitch)
        x, y = o.position.x, o.position.y
        if not self.track:
            self.track.append((time.time() - self.t0, x, y))
        else:
            d = math.hypot(x - self.track[-1][1], y - self.track[-1][2])
            if d >= self.track_step_m:
                self.dist_m += d
                self.track.append((time.time() - self.t0, x, y))

    def on_low(self, m):
        now = time.time()
        if now - self.ff_t >= 0.05:                       # 500 Hz in, 20 Hz kept
            self.ff, self.ff_t = list(m.foot_force)[:4], now

    def on_samples(self, m):
        a = np.frombuffer(m.data, np.float32).reshape(-1, 4).astype(np.float64)
        if len(a):
            self.hmap.add(a[:, 0], a[:, 1], a[:, 2], a[:, 3])
            self.hmap_n += 1

    def on_image(self, m):
        self.img, self.img_t = m, time.time()
        if self.live_hz > 0 and self.img_t - self.live_t >= 1.0 / self.live_hz:
            self.live_t = self.img_t
            a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
            ok, buf = cv2.imencode(".jpg", a[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:                                       # dashboard.py /live.mjpg streams this file
                tmp = self.live_path + ".tmp"
                with open(tmp, "wb") as fh:
                    fh.write(buf.tobytes())
                os.replace(tmp, self.live_path)

    def on_regions(self, msg):
        r = json.loads(msg.data)
        now = time.time()
        self.seg = {"t": now, "fps": r.get("fps", 0.0), "status": r.get("status", ""), "thr_mm": r.get("thr_mm")}
        if r.get("status") != "OK" or "odom_x" not in r:
            return
        ox, oy, yaw = r["odom_x"], r["odom_y"], r["odom_yaw"]
        c, s = math.cos(yaw), math.sin(yaw)
        for g in r.get("regions", []):
            kind = g["kind"]
            if kind not in (ss.SUNKEN, ss.RAISED, ss.HOLE) or g["dist_m"] > self.max_dist_m:
                continue
            if g["area_m2"] < self.min_area_m2:
                continue
            wx, wy = ox + c * g["x"] - s * g["y"], oy + s * g["x"] + c * g["y"]
            direction = "up" if kind == ss.RAISED else "down"
            h = g["height_mm"]                            # median: the deepest pixel can be a tile joint
            self.add_obs(direction, kind, wx, wy, h, g["area_m2"], g["dist_m"], now)
        u = r.get("uneven_mm")
        if u is not None and u >= self.uneven_mm:                 # 요철: placed at the middle of the measured strip
            self.add_aobs("요철", ox + c * 0.8, oy + s * 0.8, u, 0.8, now)
        # unconfirmed candidates not seen again for 3 s: noise
        self.cands = [k for k in self.cands if k["confirmed"] or now - k["last_t"] < 3.0]
        self.acands = [k for k in self.acands if k["confirmed"] or now - k["last_t"] < 3.0]

    def on_ai(self, msg):
        """Damage AI regions -> 파손 / 균열 candidates (only regions the depth could place)."""
        r = json.loads(msg.data)
        if "odom_x" not in r:
            return
        now = time.time()
        ox, oy, yaw = r["odom_x"], r["odom_y"], r["odom_yaw"]
        c, s = math.cos(yaw), math.sin(yaw)
        for g in r.get("regions", []):
            if g.get("x") is None or math.hypot(g["x"], g["y"]) > self.max_dist_m:
                continue
            wx, wy = ox + c * g["x"] - s * g["y"], oy + s * g["x"] + c * g["y"]
            val = abs(g["height_mm"]) if g["kind_ko"] == "파손" else g["score"]
            self.add_aobs(g["kind_ko"], wx, wy, val, math.hypot(g["x"], g["y"]), now, g.get("type_ko"))

    def add_aobs(self, kind, wx, wy, val, dist, now, typ=None):
        best, bd = None, self.merge_m * (2 if kind == "요철" else 1)
        for k in self.acands:
            d = math.hypot(k["x"] - wx, k["y"] - wy)
            if k["kind"] == kind and d < bd:
                best, bd = k, d
        if best is None:
            best = {"kind": kind, "x": wx, "y": wy, "obs": [], "confirmed": False, "id": None, "snap_dist": 1e9,
                    "level": -1}
            self.acands.append(best)
        k = best
        if typ:
            k.setdefault("types", {})
            k["types"][typ] = k["types"].get(typ, 0) + 1
        self.close_obs(("a", kind), wx, wy, val, dist, now)
        k["obs"] = sorted(k["obs"] + [(dist, val, wx, wy)])[:60]
        near = k["obs"][:max(3, len(k["obs"]) // 2)]
        k["x"], k["y"] = float(np.median([o[2] for o in near])), float(np.median([o[3] for o in near]))
        k["val"] = float(np.median([o[1] for o in near]))
        k["last_t"], k["n"] = now, len(k["obs"])
        if kind == "균열":
            level = 2 if k["val"] >= self.crack_levels[1] else 1 if k["val"] >= self.crack_levels[0] else 0
        else:
            level = 2 if k["val"] >= self.danger_mm else 1 if k["val"] >= self.caution_mm else 0
        changed = level != k["level"]
        k["level"] = level
        if k.get("types"):                                # majority type changed: rewrite events.csv too
            top = max(k["types"], key=k["types"].get)
            changed |= top != k.get("type_top")
            k["type_top"] = top
        if not k["confirmed"] and k["n"] >= self.confirm_n:
            k["confirmed"], k["id"], self.next_id = True, self.next_id, self.next_id + 1
            k["t_found"] = now - self.t0
            changed = True
            self.get_logger().info(f"event E{k['id']}: {kind} {k['val']:.2f} at odom ({k['x']:.2f}, {k['y']:.2f}) "
                                   f"-> {LEVEL_KO[level]}")
        if k["confirmed"] and not k.get("close") and dist < k["snap_dist"] - 0.05 and self.picture_fresh(now):
            self.snapshot(k, dist)
        if changed and k["confirmed"]:
            self.write_events()

    # ------------------------------------------------------------------ close-up (patrol_node)
    def find(self, eid):
        for key, lst in (("g", self.cands), ("a", self.acands)):
            for k in lst:
                if k["confirmed"] and k["id"] == eid:
                    return (key, k["dir"] if key == "g" else k["kind"]), k
        return None, None

    def on_closeup(self, msg):
        try:
            r = json.loads(msg.data)
            eid, action = int(r["id"]), r["action"]
        except (ValueError, KeyError, TypeError):
            self.get_logger().warn(f"bad /inspect/closeup message: {msg.data!r}")
            return
        now = time.time()
        if action == "start":
            key, k = self.find(eid)
            if k is None:
                self.get_logger().warn(f"close-up: no confirmed event E{eid}")
                return
            self.close = {"id": eid, "key": key, "k": k, "x": k["x"], "y": k["y"], "obs": [], "t0": now,
                          "photo": None, "photo_t": 0.0}
            self.get_logger().info(f"close-up E{eid} started")
        elif action == "end" and self.close and self.close["id"] == eid:
            self.finish_close(now)

    def close_obs(self, key, wx, wy, val, dist, now):
        c = self.close
        if c is None or key != c["key"] or math.hypot(wx - c["x"], wy - c["y"]) > 1.5 * self.merge_m:
            return
        c["obs"].append((dist, val))
        if now - c["photo_t"] >= 0.5 and self.picture_fresh(now):     # robot stands still: latest view is fine
            name = f"E{c['id']}_close.jpg"
            path = os.path.join(self.dir, "snap", name)
            try:
                with open(self.live_path, "rb") as src, open(path + ".tmp", "wb") as dst:
                    dst.write(src.read())
                os.replace(path + ".tmp", path)
                c["photo"], c["photo_t"] = name, now
            except OSError:
                pass

    def finish_close(self, now):
        c, self.close = self.close, None
        k, n = c["k"], len(c["obs"])
        res = {"n": n, "secs": round(now - c["t0"], 1), "photo": c["photo"]}
        if n >= 3:
            vals = [o[1] for o in c["obs"]]
            val = float(np.median(vals))
            a = abs(val)
            if c["key"] == ("a", "균열"):
                level = 2 if a >= self.crack_levels[1] else 1 if a >= self.crack_levels[0] else 0
                res["score"] = round(val, 3)
            else:
                level = 2 if a >= self.danger_mm else 1 if a >= self.caution_mm else 0
                res["h_mm"] = round(val, 1)
                res["sd_mm"] = round(float(np.std(vals)), 1)
            res.update(level=level, dist_m=round(float(np.median([o[0] for o in c["obs"]])), 2),
                       verdict="확인" if level >= 1 else "오탐 의심")
        else:
            res["verdict"] = "재측정 실패"
        k["close"] = res
        if c["photo"]:
            k["snap"] = c["photo"]
        self.get_logger().info(f"close-up E{c['id']}: {res}")
        self.write_events()

    # ------------------------------------------------------------------ events
    def add_obs(self, direction, kind, wx, wy, h_mm, area, dist, now):
        best, bd = None, self.merge_m
        for k in self.cands:
            d = math.hypot(k["x"] - wx, k["y"] - wy)
            if k["dir"] == direction and d < bd:
                best, bd = k, d
        if best is None:
            best = {"dir": direction, "x": wx, "y": wy, "obs": [], "confirmed": False, "id": None, "first_t": now,
                    "snap_dist": 1e9, "level": -1, "hole": False}
            self.cands.append(best)
        k = best
        self.close_obs(("g", direction), wx, wy, h_mm, dist, now)
        k["obs"].append((dist, h_mm, area, wx, wy))
        k["obs"] = sorted(k["obs"])[:60]                  # keep the 60 closest observations
        n = len(k["obs"])
        # a tile leaving the bottom of the image is seen only partly: smaller area, centre shifted forward ->
        # position / height / area from the views that saw (nearly) all of it, the closest of those first
        amax = max(o[2] for o in k["obs"])
        full = [o for o in k["obs"] if o[2] >= 0.7 * amax]
        k["x"] = float(np.median([o[3] for o in full]))
        k["y"] = float(np.median([o[4] for o in full]))
        k["h_mm"] = float(np.median([o[1] for o in full[:max(3, len(full) // 2)]]))
        k["area_m2"] = float(np.median([o[2] for o in full]))
        k["last_t"], k["n"] = now, n
        a = abs(k["h_mm"])
        k["hole"] = k["dir"] == "down" and a >= 30                  # deeper than 3 cm: hole / missing block
        level = 2 if a >= self.danger_mm else 1 if a >= self.caution_mm else 0
        changed = level != k["level"]
        k["level"] = level
        if not k["confirmed"] and n >= self.confirm_n:
            k["confirmed"], k["id"], self.next_id = True, self.next_id, self.next_id + 1
            k["t_found"] = now - self.t0
            changed = True
            self.get_logger().info(f"event E{k['id']}: {DIR_KO['hole' if k['hole'] else k['dir']]} "
                                   f"{k['h_mm']:+.1f} mm, {1e4 * k['area_m2']:.0f} cm2 at odom ({k['x']:.2f}, "
                                   f"{k['y']:.2f}) -> {LEVEL_KO[level]}")
        whole = area >= 0.7 * amax                       # this view saw (nearly) the whole region
        if k["confirmed"] and whole and not k.get("close") and dist < k["snap_dist"] - 0.05 \
                and self.picture_fresh(now):
            self.snapshot(k, dist)                       # picture from the closest whole view so far
        if changed and k["confirmed"]:
            self.write_events()

    def picture_fresh(self, now):
        if self.snap_source == "topic":
            return self.img is not None and now - self.img_t < 0.5
        try:
            return now - os.stat(self.live_path).st_mtime < 0.5
        except OSError:
            return False

    def snapshot(self, k, dist):
        name = f"E{k['id']}.jpg"
        path = os.path.join(self.dir, "snap", name)
        if self.snap_source == "topic":
            m = self.img
            a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3)
            cv2.imwrite(path, a[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
        else:                                            # surface_seg_node replaces the file atomically
            try:
                with open(self.live_path, "rb") as src, open(path + ".tmp", "wb") as dst:
                    dst.write(src.read())
                os.replace(path + ".tmp", path)
            except OSError:
                return
        k["snap"], k["snap_dist"] = name, dist

    def events(self):
        out = []
        for k in self.cands:
            if not k["confirmed"]:
                continue
            out.append({"id": k["id"], "kind": DIR_KO["hole" if k["hole"] else k["dir"]], "h_mm": round(k["h_mm"], 1),
                        "area_cm2": round(1e4 * k["area_m2"], 0), "x": round(k["x"], 3), "y": round(k["y"], 3),
                        "level": k["level"], "level_ko": LEVEL_KO[k["level"]], "n_obs": k["n"],
                        "t_found": round(k["t_found"], 1), "snap": k.get("snap")})
        for k in self.acands:
            if not k["confirmed"]:
                continue
            e = {"id": k["id"], "kind": k["kind"], "x": round(k["x"], 3), "y": round(k["y"], 3), "level": k["level"],
                 "level_ko": LEVEL_KO[k["level"]], "n_obs": k["n"], "t_found": round(k["t_found"], 1),
                 "snap": k.get("snap"), "area_cm2": 0}
            if k.get("types"):
                e.update(type_ko=max(k["types"], key=k["types"].get), type_votes=dict(k["types"]))
            if k["kind"] == "균열":
                e.update(score=round(k["val"], 3), h_mm=0.0)
            else:
                e.update(h_mm=round(k["val"], 1))
            out.append(e)
        for e in out:                                     # close-up result (patrol): its level is the final one
            src = self.find(e["id"])[1]
            c = src.get("close") if src else None
            if c:
                e["close"] = c
                if "level" in c:
                    e["level_pass"], e["level"], e["level_ko"] = e["level"], c["level"], LEVEL_KO[c["level"]]
        return sorted(out, key=lambda e: e["id"])

    def write_events(self):
        ev = self.events()
        with open(os.path.join(self.dir, "events.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["id", "kind", "height_mm", "area_cm2", "odom_x", "odom_y", "level", "observations",
                        "found_s", "snapshot", "type", "close_value", "close_n", "close_dist_m", "verdict"])
            for e in ev:
                c = e.get("close") or {}
                w.writerow([e["id"], e["kind"], e["h_mm"], e["area_cm2"], e["x"], e["y"], e["level_ko"], e["n_obs"],
                            e["t_found"], e["snap"] or "", e.get("type_ko", ""), c.get("h_mm", c.get("score", "")),
                            c.get("n", ""), c.get("dist_m", ""), c.get("verdict", "")])
        self.ev_pub.publish(String(data=json.dumps(ev, ensure_ascii=False, default=float)))

    # ------------------------------------------------------------------ outputs
    def sample(self):
        if self.pose is None:
            return
        t, x, y, z, yaw, roll, pitch = self.pose
        ff = self.ff if self.ff is not None and time.time() - self.ff_t < 1.0 else [float("nan")] * 4
        row = (round(time.time() - self.t0, 2), x, y, z, math.degrees(roll), math.degrees(pitch), *ff)
        self.sens.append(row)
        self.sens = [r for r in self.sens if r[0] > row[0] - 60]
        self.sens_f.write(",".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in row) + "\n")

    def write_state(self):
        now = time.time()
        ev = self.events()
        n_c, n_d = sum(e["level"] == 1 for e in ev), sum(e["level"] == 2 for e in ev)
        if self.hmap_n and now - self.hmap_saved > 3.0:             # height map file for the dashboard
            self.hmap_saved = now
            self.hmap.save(os.path.join(self.dir, "heightmap.npz.tmp.npz"))
            if os.path.exists(os.path.join(self.dir, "heightmap.npz.tmp.npz")):
                os.replace(os.path.join(self.dir, "heightmap.npz.tmp.npz"), os.path.join(self.dir, "heightmap.npz"))
        st = {"session": self.session, "location": self.location, "start": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.t0)),
              "elapsed_s": round(now - self.t0, 1), "distance_m": round(self.dist_m, 2),
              "thresholds": {"caution_mm": self.caution_mm, "danger_mm": self.danger_mm, "provisional": True},
              "n_caution": n_c, "n_danger": n_d, "risk": "높음" if n_d else "보통" if n_c else "낮음",
              "pose": None if self.pose is None else {"x": self.pose[1], "y": self.pose[2], "z": self.pose[3],
                                                      "yaw": self.pose[4]},
              "track": [(round(t, 1), round(x, 3), round(y, 3)) for t, x, y in self.track[-20000:]],
              "events": ev, "sensors": self.sens,
              "status": {"camera_age_s": round(now - self.seg["t"], 1) if self.seg["t"] else None,
                         "camera_fps": round(self.seg["fps"] or 0.0, 1), "camera_status": self.seg["status"],
                         "thr_mm": self.seg["thr_mm"],
                         "odom_age_s": round(now - self.pose[0], 1) if self.pose else None,
                         "foot_force": self.ff is not None and now - self.ff_t < 1.0,
                         "candidates": sum(not k["confirmed"] for k in self.cands),
                         "map_cells": self.hmap.n_cells() if self.hmap_n else 0}}
        tmp = os.path.join(self.dir, "state.json.tmp")
        with open(tmp, "w") as fh:
            json.dump(st, fh, default=float, ensure_ascii=False)
        os.replace(tmp, os.path.join(self.dir, "state.json"))
        self.ev_pub.publish(String(data=json.dumps(ev, ensure_ascii=False, default=float)))
        self.sens_f.flush()

    def destroy_node(self):
        self.hmap_saved = 0.0
        self.write_state()
        self.write_events()
        self.sens_f.close()
        try:                                              # A4: field report of this session
            import report
            path = report.make(self.dir)
            self.get_logger().info(f"report: {path}")
        except Exception as e:  # noqa: BLE001  the session data is saved anyway
            self.get_logger().warn(f"report failed: {e} (make it later: python3 report.py {self.dir})")
        ev = self.events()
        self.get_logger().info(f"session {self.session}: {self.dist_m:.1f} m, {len(ev)} events "
                               f"({sum(e['level'] == 1 for e in ev)} caution, {sum(e['level'] == 2 for e in ev)} danger)"
                               f" -> {self.dir}")
        super().destroy_node()


def _stop(*_):
    raise KeyboardInterrupt


def main():
    signal.signal(signal.SIGINT, _stop)      # background jobs of a script ignore SIGINT: re-enable it
    signal.signal(signal.SIGTERM, _stop)     # and save the session on TERM too
    rclpy.init()
    n = Inspector()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        rclpy.try_shutdown() if hasattr(rclpy, "try_shutdown") else rclpy.shutdown()


if __name__ == "__main__":
    main()
