"""Stage 3 core (days 7-10): clean a point cloud, keep the road surface in front of the robot, fit a plane with
RANSAC and turn it into a slope angle / grade with the robot posture removed. No learning, pure geometry.
Used by slope_offline.py (bags) and later by the live node.

Frames
  base_link  : robot body (x forward, y left, z up) — tilts with the robot
  levelled   : base_link rotated by the body IMU roll + pitch -> z is vertical (gravity), x keeps the robot heading.
               Posture compensation = measuring the plane in this frame instead of base_link.
Plane:  a*x + b*y + c*z + d = 0, (a, b, c) unit normal pointing up (c > 0), in the levelled frame.
  slope_deg   = angle between the normal and vertical         slope_pct = 100 * tan(slope_deg)
  along_deg   > 0: ground rises in the walking direction       side_deg  > 0: ground rises to the left
  height_m    = d / c = height of base_link above the plane (floor at z = -d/c) (body height, sanity check)

Every frame gets a status; failed frames are kept (with the reason) so the valid rate can be reported:
  OK, NO_POINTS (nothing after cleaning), FEW_POINTS (too few road points in the ROI),
  RANSAC_WEAK (plane supported by too few / too small a share of points), BAD_NORMAL (plane > max_tilt,
  i.e. a wall or object won), BAD_HEIGHT (at its nearest seen point the plane is not at a height the road can
  have: expected floor -body_height, +- band, +- max_slope over that distance -> e.g. a table top).
  (Checked where the plane is seen, not under the robot: a ramp 2 m ahead extended back to x = 0 is legitimately
  far from the floor height.)
"""
import cv2
import numpy as np
import open3d as o3d

# per-sensor defaults. ROI is in the levelled frame at base_link [m]. The Hesai XT-16 (+-15 deg beams, ~0.37 m
# above the floor) only hits the floor from ~1.4 m on, so its ROI starts further out than the camera's.
SENSORS = {
# normal_radius: neighbourhood for the surface-normal test; LiDAR rings are far apart, so a small radius sees
# a single ring (a line) and its normal is meaningless.
    "hesai": {"topic": "/lidar_points", "roi_x": (1.2, 3.5), "roi_y": 1.0, "voxel": 0.03, "min_points": 150,
              "normal_radius": 0.30},
    "camera": {"topic": "/camera/depth/color/points", "roi_x": (0.3, 2.0), "roi_y": 0.6, "voxel": 0.02,
               "min_points": 300, "normal_radius": 0.08},
    "l1": {"topic": "/utlidar/cloud_deskewed", "roi_x": (0.3, 2.0), "roi_y": 0.6, "voxel": 0.03, "min_points": 100,
           "normal_radius": 0.15},
}
# base_link (centre of the body) -> front of the robot (nose), along x [m]. Distances shown to people are measured
# from here ("front_dist_m"); internal ones (dist_m, step_dist_m) stay from base_link. Calibrated 2026-10-18 with the
# user's test: box 62 cm in front of the robot, measured 1.02 m from base_link by both the camera and the L1 (so the
# sensors agree; only the reference point differed). Go2 head front is ~0.38-0.40 m ahead of the body centre.
FRONT_OFFSET_M = 0.40

PARAMS = {
    "range_min": 0.2, "range_max": 6.0,      # day 7: sensor range kept [m]
    "sor_neighbors": 16, "sor_std": 2.0,     # day 7: statistical outlier removal
    "max_slope_deg": 20.0,                   # day 8: height band follows ramps up to this angle
    # day 8: band around the expected floor z = -body_height [m]. Odometry z is only roughly the body height
    # (flat walk: L1 floor 0.31 m vs odom 0.25 m), so more room below than above (objects stand above).
    "band_below": 0.25, "band_above": 0.10,
    "normal_max_deg": 35.0,                  # day 8: drop points whose local surface is steeper (people, trees, walls)
    "ransac_dist": 0.02, "ransac_iter": 500,  # day 9
    "min_inliers": 80, "min_inlier_ratio": 0.30,
    "max_tilt_deg": 30.0,                    # validity: steeper plane = not the road
    "step_min_m": 0.03,                      # step: two parallel road planes at least this far apart in height
    "step_min_share": 0.05,                  # step: each surface needs >= this share of the road points
    "step_min_snr": 3.0,                     # step: height difference >= this many times the residual of the 2 planes
    "step_max_angle_deg": 8.0,               # step: both planes (nearly) parallel
    "step_min_width": 0.6,                   # step: both surfaces cover >= this share of the road width (a box does not)
    "step_max_overlap": 0.10,                # step: <= this share of the near surface lies beyond the step edge
    "obstacle_min_h": 0.03,                  # obstacle: points at least this high above the road plane [m]
    "obstacle_max_h": 2.0,                   # ... and at most this high (people / furniture are obstacles too)
    "obstacle_min_points": 5,                # ... clusters with fewer points are noise
    "height_range": (0.05, 0.70),            # validity: plane height under the robot, if it reaches the robot [m]
}


def xform(P, R, t=None):
    """P (N,3) -> P @ R.T + t. numpy's (N,3)@(3,3) is ~50x slower than cv2.transform on this Jetson's ARM build
    (51 ms vs 0.9 ms for 77k points, measured 2026-10-18), so every point-cloud transform goes through here."""
    P = np.ascontiguousarray(P)
    dt = np.float32 if P.dtype == np.float32 else np.float64
    M = np.hstack([np.asarray(R, dt), np.zeros((3, 1), dt) if t is None else np.asarray(t, dt).reshape(3, 1)])
    return cv2.transform(P.astype(dt, copy=False).reshape(-1, 1, 3), M).reshape(-1, 3)


def pdist(P, m):
    """Signed distance of points P (N,3) to plane m = (a, b, c, d) with a unit normal (column math: fast)."""
    return P[:, 0] * m[0] + P[:, 1] * m[1] + P[:, 2] * m[2] + m[3]


def rotation(yaw, pitch, roll):
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def level_matrix(roll, pitch):
    """base_link -> levelled frame (removes the body roll and pitch measured by the IMU)."""
    return rotation(0.0, pitch, roll)


def angles(n):
    """Unit normal (up) -> slope, along-heading, sideways angles [deg]."""
    n = n / np.linalg.norm(n)
    if n[2] < 0:
        n = -n
    return (float(np.degrees(np.arccos(np.clip(n[2], -1, 1)))),
            float(np.degrees(np.arctan2(-n[0], n[2]))), float(np.degrees(np.arctan2(-n[1], n[2]))))


# ------------------------------------------------------------------------------------------------- day 7
def clean(xyz_sensor, R_sb, t_sb, R_lb, voxel, p=PARAMS):
    """Raw sensor points -> cleaned points in the levelled frame + counts of each step.
    R_sb, t_sb: sensor -> base_link (TF). R_lb: base_link -> levelled (IMU)."""
    st = {"n_raw": len(xyz_sensor)}
    x = xyz_sensor[np.isfinite(xyz_sensor).all(1)]
    x = x[np.abs(x).sum(1) > 1e-3]                       # some drivers put (0,0,0) for "no return"
    st["n_finite"] = len(x)
    r = np.linalg.norm(x, axis=1)
    x = x[(r > p["range_min"]) & (r < p["range_max"])]
    st["n_range"] = len(x)
    lev = xform(xform(x, R_sb, t_sb), R_lb)
    return lev, st


def denoise(pts, voxel, p=PARAMS):
    if len(pts) == 0:
        return pts
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64))).voxel_down_sample(voxel)
    if len(pc.points) > p["sor_neighbors"] * 2:
        pc, _ = pc.remove_statistical_outlier(p["sor_neighbors"], p["sor_std"])
    return np.asarray(pc.points)


# ------------------------------------------------------------------------------------------------- day 8
def road_roi(lev, cfg, body_height, p=PARAMS):
    """Box in front of the robot + height band around the expected floor (z = -body_height), widened with
    distance so ramps up to max_slope_deg stay inside. Objects standing on the road (people, trees, boxes)
    stick out above the band; whatever remains of them is removed by the surface-normal test in denoise_road."""
    x0, x1 = cfg["roi_x"]
    x, y, z = lev[:, 0], lev[:, 1], lev[:, 2]
    k = np.tan(np.radians(p["max_slope_deg"]))
    z0 = -body_height
    m = ((x >= x0) & (x <= x1) & (np.abs(y) <= cfg["roi_y"]) &
         (z >= z0 - p["band_below"] - k * x) & (z <= z0 + p["band_above"] + k * x))
    return lev[m]


def keep_flat_surfaces(pts, radius=0.12, p=PARAMS):
    """Drop points whose local surface (PCA normal) is steeper than normal_max_deg."""
    if len(pts) < 10:
        return pts, 0
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    ok = np.abs(np.asarray(pc.normals)[:, 2]) > np.cos(np.radians(p["normal_max_deg"]))
    return pts[ok], int((~ok).sum())


# ------------------------------------------------------------------------------------------------- day 9
def fit_plane(pts, p=PARAMS):
    """RANSAC (Open3D) then least-squares refit on the inliers. Returns (abcd, inlier mask, rms) or None."""
    if len(pts) < 3:
        return None
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    m, inl = pc.segment_plane(distance_threshold=p["ransac_dist"], ransac_n=3, num_iterations=p["ransac_iter"])
    mask = np.zeros(len(pts), bool)
    mask[inl] = True
    q = pts[mask]
    if len(q) >= 3:                                    # refine: total least squares on the inliers
        c = q.mean(0)
        n = np.linalg.svd(q - c, full_matrices=False)[2][2]
        m = np.r_[n, -n @ c]
    m = np.asarray(m, float)
    m /= np.linalg.norm(m[:3])
    if m[2] < 0:
        m = -m
    rms = float(np.sqrt(np.mean((pdist(q, m)) ** 2))) if len(q) else float("nan")
    return m, mask, rms


def plane_z(m, x):
    """Height of plane m at (x, y=0) in the levelled frame."""
    return -(m[0] * x + m[3]) / m[2]


def lsq_plane(q):
    c = q.mean(0)
    n = np.linalg.svd(q - c, full_matrices=False)[2][2]
    m = np.r_[n, -n @ c]
    m /= np.linalg.norm(m[:3])
    return -m if m[2] < 0 else m


def find_step(road, m, mask, p=PARAMS):
    """Two road surfaces at different heights one behind the other -> step.
    A single RANSAC plane can lie diagonally across a low step (it collects points of both levels within the
    distance threshold), so the road points are split at candidate boundaries along x and a plane is fitted on
    each side; the split is a step when both sides are (nearly) parallel, their heights differ by >= step_min_m at
    the boundary and the residual drops clearly compared with one plane.
    Returns {} or a dict with step_dz_m (> 0: the far surface is higher = step up), step_dist_m (x of the near end of
    the far surface), and plane / mask: the larger of the two surfaces — the slope is measured on it."""
    if len(road) < 2 * p["min_inliers"]:
        return {}
    x = road[:, 0]
    one = lsq_plane(road)
    rms1 = np.sqrt(np.mean((pdist(road, one)) ** 2))
    best = None
    for qb in np.linspace(0.08, 0.92, 29):
        xb = float(np.quantile(x, qb))
        near, far = road[x < xb], road[x >= xb]
        if min(len(near), len(far)) < max(p["min_inliers"] // 2, p["step_min_share"] * len(road)):
            continue
        mn, mf = lsq_plane(near), lsq_plane(far)
        res = np.r_[pdist(near, mn), pdist(far, mf)]
        rms2 = np.sqrt(np.mean(res ** 2))
        if best is None or rms2 < best[0]:
            best = (rms2, xb, mn, mf, near, far)
    if best is None:
        return {}
    rms2, xb, mn, mf, near, far = best
    for _ in range(4):        # refine: give every point to the closer of the two planes, refit (2-plane k-means)
        is_near = np.abs(pdist(road, mn)) < np.abs(pdist(road, mf))
        if min(is_near.sum(), (~is_near).sum()) < p["min_inliers"] // 2:
            break
        near, far = road[is_near], road[~is_near]
        mn, mf = lsq_plane(near), lsq_plane(far)
        if np.median(near[:, 0]) > np.median(far[:, 0]):
            mn, mf, near, far = mf, mn, far, near
        xb = 0.5 * (float(np.percentile(near[:, 0], 95)) + float(np.percentile(far[:, 0], 5)))
    rms2 = np.sqrt(np.mean(np.r_[pdist(near, mn), pdist(far, mf)] ** 2))
    ang = np.degrees(np.arccos(np.clip(abs(float(mn[:3] @ mf[:3])), -1, 1)))
    dz = plane_z(mf, xb) - plane_z(mn, xb)
    # a step = two parallel surfaces whose height difference is large against their own roughness (a single
    # surface, even a ramp, split in two gives dz ~ 0 at the boundary). Not a residual-ratio test: when one side is
    # small (step just in view) one plane through both still fits most points and the ratio barely changes.
    if abs(dz) < p["step_min_m"] or ang > p["step_max_angle_deg"] or abs(dz) < p["step_min_snr"] * rms2:
        return {}
    # a step runs across the path: the far surface starts where the near one ends (the near surface does not
    # continue behind it, as the floor does around a box) and both span most of the width
    x_edge = float(np.percentile(far[:, 0], 5))
    if np.mean(near[:, 0] > x_edge + 0.10) > p["step_max_overlap"]:
        return {}
    bins = np.linspace(road[:, 1].min(), road[:, 1].max() + 1e-6, 13)
    for side in (near, far):
        if np.mean(np.histogram(side[:, 1], bins)[0] > 0) < p["step_min_width"]:
            return {}
    # slope is measured on the surface with more points: right before a step (or on a narrow platform edge) the
    # near strip is too small for a reliable angle (Gazebo: 0.25 m strip + D435 noise -> 5-7 deg errors)
    keep = np.abs(pdist(road, mn)) < np.abs(pdist(road, mf))
    keep = keep if len(near) >= len(far) else ~keep
    surf = mn if len(near) >= len(far) else mf
    both = int(((np.abs(pdist(road, mn)) < p["ransac_dist"]) | (np.abs(pdist(road, mf)) < p["ransac_dist"])).sum())
    return {"step_dz_m": float(dz), "step_dist_m": float(np.percentile(far[:, 0], 5)), "step_near_n": len(near),
            "step_far_n": len(far), "step_inliers_both": both, "plane": surf,
            "planes": [(mn, -np.inf, xb), (mf, xb, np.inf)], "mask": keep & (np.abs(pdist(road, surf)) < p["ransac_dist"])}


def heights_above(pts, planes, edge=0.15):
    """Height of each point above the road surface under it (unit normals point up). With a step, the plane on
    the point's side of the step edge; within `edge` of the edge, the closest plane below the point (the top
    corner of a step riser belongs to the upper surface, not to an obstacle on the lower one)."""
    h = np.full(len(pts), np.nan)
    for m, xa, xb in planes:
        sel = (pts[:, 0] >= xa) & (pts[:, 0] < xb)
        h[sel] = pdist(pts[sel], m)
    if len(planes) > 1:
        xb = planes[0][2]
        near = np.abs(pts[:, 0] - xb) < edge
        cand = np.stack([pdist(pts[near], m) for m, _, _ in planes], 1)
        # the riser of the step: points between the two levels at the edge belong to the step, not to an obstacle
        # (Gazebo 2026-10-18: the face of a 10 cm step read as a "6.8 cm obstacle")
        riser = (cand.max(1) > 0.0) & (cand.min(1) < 0.0)
        cand[cand < -0.02] = np.inf
        hn = cand.min(1)
        hn[riser] = 0.0
        h[near] = hn
    return h


def top_height(h):
    """Height of an object from the heights of its points: the most frequent height in its upper half (the flat top
    of a box / tile forms a peak in the height histogram), refined as the median of the points within 8 mm of that
    peak. Symmetric depth noise then cancels. The 95th percentile used before read ~1 cm (camera noise 3-6 mm:
    0.7-2 cm) too high, because it picks the upper tail of the noisy top surface (user test 2026-10-18: 6 cm box
    read 7 cm). Objects without a flat top (people, chair legs) get the peak of their upper part."""
    h = h[np.isfinite(h)]
    if len(h) < 5:
        return float(np.max(h)) if len(h) else float("nan")
    p95 = np.percentile(h, 95)
    up = h[h >= 0.5 * p95]
    bins = np.arange(up.min(), up.max() + 0.004, 0.004)
    if len(bins) < 3:
        return float(np.median(up))
    cnt, edges = np.histogram(up, bins)
    peak = 0.5 * (edges[np.argmax(cnt)] + edges[np.argmax(cnt) + 1])
    near = up[np.abs(up - peak) <= 0.008]
    return float(np.median(near))


def find_obstacles(lev, cfg, planes, p=PARAMS):
    """Objects standing on the road: points of the ROI box that are obstacle_min_h .. obstacle_max_h above the
    road surface (signed distance to the plane under them; with a step, to the surface on their side of the step),
    voxel-thinned and clustered with DBSCAN. Returns (list of dicts sorted by distance, list of point arrays).
    Each obstacle: x, y (centre, levelled frame at base_link), dist_m (nearest point, horizontal), height_m (95th
    percentile above the road), size_x_m, size_y_m (extent), n (points)."""
    x0, x1 = cfg["roi_x"]
    box = lev[(lev[:, 0] >= x0) & (lev[:, 0] <= x1) & (np.abs(lev[:, 1]) <= cfg["roi_y"])]
    if len(box) == 0:
        return [], []
    h = heights_above(box, planes)
    cand = box[(h >= p["obstacle_min_h"]) & (h <= p["obstacle_max_h"])]
    if len(cand) < p["obstacle_min_points"]:
        return [], []
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(cand)).voxel_down_sample(cfg["voxel"])
    q = np.asarray(pc.points)
    labels = np.asarray(pc.cluster_dbscan(eps=max(0.06, 2.5 * cfg["voxel"]), min_points=3))
    obs, pts = [], []
    for k in range(labels.max() + 1 if len(labels) else 0):
        c = q[labels == k]
        if len(c) < p["obstacle_min_points"]:
            continue
        # height from the raw (not voxel-averaged) candidate points under the cluster's footprint: voxel centres at the
        # box edge mix top and side points and pull the height ~3-4 mm off
        pad = cfg["voxel"]
        inb = ((cand[:, 0] >= c[:, 0].min() - pad) & (cand[:, 0] <= c[:, 0].max() + pad) &
               (cand[:, 1] >= c[:, 1].min() - pad) & (cand[:, 1] <= c[:, 1].max() + pad))
        hc = heights_above(cand[inb] if inb.sum() >= len(c) else c, planes)
        obs.append({"x": float(c[:, 0].mean()), "y": float(c[:, 1].mean()),
                    "dist_m": float(np.hypot(c[:, 0], c[:, 1]).min()),
                    "front_dist_m": float(max(0.0, c[:, 0].min() - FRONT_OFFSET_M)), "height_m": top_height(hc), "height_p95_m": float(np.nanpercentile(hc, 95)),
                    "size_x_m": float(np.ptp(c[:, 0])), "size_y_m": float(np.ptp(c[:, 1])), "n": int(len(c))})
        pts.append(c)
    order = np.argsort([o["dist_m"] for o in obs])
    return [obs[i] for i in order], [pts[i] for i in order]


# ------------------------------------------------------------------------------------------------- day 10
def estimate(xyz_sensor, R_sb, t_sb, roll, pitch, body_height, sensor, p=PARAMS):
    """Full chain for one frame. Returns a flat dict (one CSV row) and the points for plotting."""
    cfg = SENSORS[sensor]
    R_lb = level_matrix(roll, pitch)
    lev, st = clean(xyz_sensor, R_sb, t_sb, R_lb, cfg["voxel"], p)            # 7
    roi = road_roi(lev, cfg, body_height, p)                                  # 8
    st["n_roi_box"] = len(roi)
    roi = denoise(roi, cfg["voxel"], p)                                       # 7: voxel + outliers on the ROI
    st["n_denoised"] = len(roi)
    road, st["n_steep_removed"] = keep_flat_surfaces(roi, cfg["normal_radius"], p)                  # 8
    st["n_road"] = len(road)
    out = dict(st, status="OK", reason="", body_roll_deg=float(np.degrees(roll)),
               body_pitch_deg=float(np.degrees(pitch)), body_height=float(body_height))
    pts = {"roi": roi, "road": road, "inliers": np.zeros((0, 3)), "obstacles": [], "planes": []}
    if st["n_finite"] == 0:
        out.update(status="NO_POINTS", reason="cloud empty after removing NaN/Inf/zeros")
        return out, pts
    if len(road) < cfg["min_points"]:
        out.update(status="FEW_POINTS", reason=f"{len(road)} road points < {cfg['min_points']}")
        return out, pts
    fit = fit_plane(road, p)                                                  # 9
    m, mask, rms = fit
    step = find_step(road, m, mask, p)                                        # two surfaces at different heights?
    planes = [(m, -np.inf, np.inf)]
    if step:                                                                  # measure the slope on the surface
        planes = step.pop("planes")
        m, mask = step.pop("plane"), step.pop("mask")                         # with more points, not across the step
        rms = float(np.sqrt(np.mean((pdist(road[mask], m)) ** 2)))
    pts["inliers"] = road[mask]
    n_in = int(mask.sum())
    slope, along, side = angles(m[:3])                                        # 10: levelled = posture removed
    n_body = R_lb.T @ m[:3]                                                   # same plane in the tilted body frame
    slope_b, along_b, side_b = angles(n_body)
    out.update(a=m[0], b=m[1], c=m[2], d=m[3], a_body=n_body[0], b_body=n_body[1], c_body=n_body[2], d_body=m[3],
               n_inliers=n_in, inlier_ratio=n_in / len(road), plane_rms_m=rms,
               slope_deg=slope, slope_pct=100 * np.tan(np.radians(slope)), along_deg=along,
               along_pct=100 * np.tan(np.radians(along)), side_deg=side,
               slope_unlevelled_deg=slope_b, along_unlevelled_deg=along_b, side_unlevelled_deg=side_b,
               height_m=m[3] / m[2],
               dist_median_m=float(np.median(np.hypot(road[mask, 0], road[mask, 1]))))
    out.update(step_detected=int(bool(step)), **step)
    if step:
        out["step_front_dist_m"] = max(0.0, step["step_dist_m"] - FRONT_OFFSET_M)
    obstacles, obs_pts = find_obstacles(lev, cfg, planes, p)                  # things standing on the road
    pts["obstacles"], pts["planes"] = obs_pts, planes
    out["n_obstacles"] = len(obstacles)
    out["obstacles"] = obstacles
    if obstacles:
        o = obstacles[0]
        out.update(obstacle_dist_m=o["dist_m"], obstacle_front_dist_m=o["front_dist_m"], obstacle_height_m=o["height_m"],
                   obstacle_y_m=o["y"],
                   obstacle_size_x_m=o["size_x_m"], obstacle_size_y_m=o["size_y_m"])
    n_sup = step.get("step_inliers_both", n_in) if step else n_in            # with a step: points on either surface
    if n_sup < p["min_inliers"] or n_sup / len(road) < p["min_inlier_ratio"]:
        out.update(status="RANSAC_WEAK", reason=f"{n_sup} inliers ({100 * n_sup / len(road):.0f} %)")
    elif slope > p["max_tilt_deg"]:
        out.update(status="BAD_NORMAL", reason=f"plane tilted {slope:.1f} deg > {p['max_tilt_deg']}")
    else:
        q = road[mask]
        x_near = float(np.percentile(q[:, 0], 5))                  # nearest part of the plane that was seen
        z_near = -(m[0] * x_near + m[3]) / m[2]                     # plane height there (y = 0)
        k = np.tan(np.radians(p["max_slope_deg"]))
        lo = -body_height - p["band_below"] - k * x_near
        hi = -body_height + p["band_above"] + k * x_near
        out["plane_z_near_m"], out["x_near_m"] = z_near, x_near
        if not lo <= z_near <= hi:
            out.update(status="BAD_HEIGHT", reason=f"plane at z {z_near:+.2f} m (x {x_near:.2f} m), road expected "
                                                   f"{lo:+.2f}..{hi:+.2f} m")
        elif x_near < 0.6 and not p["height_range"][0] <= out["height_m"] <= p["height_range"][1]:
            out.update(status="BAD_HEIGHT", reason=f"floor {out['height_m']:.2f} m below base_link")
    return out, pts
