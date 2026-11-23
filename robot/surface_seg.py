"""Geometric surface segmentation of the camera image (no AI): every pixel with depth is classified by its height
above the road plane found by slope_core (RANSAC), so the colours are measurements, not guesses.

Classes (height h above the road surface under the pixel; thr = noise threshold, adaptive):
  FLOOR     |h| <= thr                      normal floor
  SUNKEN    -3 cm < h < -thr                sunken / pressed-down tile (loose tile)
  HOLE      h <= -3 cm                      hole, missing tile, drop (edge of a step down)
  RAISED    thr < h < 3 cm                  raised / lifted tile edge
  OBSTACLE  h >= 3 cm                       object on the floor (same 3 cm as slope_core.find_obstacles)
  STEP      on the second plane of a step   the other level of a step
  NONE      no depth / outside the ROI box
thr = clip(3 x robust sigma of the floor heights, 4 mm, 20 mm): the smallest sunken depth that can be told apart from
the depth noise today (D435: ~2-5 mm at 1 m, grows with distance^2 -> sunken tiles only near, 0.4-1.2 m).
SUNKEN / RAISED pixels are cleaned (opening 3x3) and grouped into regions; regions smaller than min_area are dropped.
"""
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

import slope_core as sc

NONE, FLOOR, SUNKEN, HOLE, RAISED, OBSTACLE, STEP = range(7)
COLORS = {FLOOR: (40, 200, 70), SUNKEN: (30, 120, 255), HOLE: (150, 40, 200), RAISED: (255, 210, 0),
          OBSTACLE: (235, 30, 30), STEP: (255, 140, 0)}
NAMES_KO = {FLOOR: "정상 바닥", SUNKEN: "함몰 (꺼진 타일)", HOLE: "구멍 / 깊은 낙차", RAISED: "돌출 (들뜬 타일)",
            OBSTACLE: "장애물", STEP: "단차 면"}
PARAMS = {"z_min": 0.2, "z_max": 4.0, "big": 0.03, "thr_min": 0.004, "thr_max": 0.02, "min_area_m2": 0.002,
          "roi_y": 0.8}
# StepMap dashboard level colours (RGB): a region at or above the caution / danger height gets this box colour, so
# the camera picture and the map use the same 위험 / 주의 as StepMap (step_warn_cm / step_danger_cm)
LEVEL_RGB = {2: (239, 100, 97), 1: (247, 181, 88)}
LEVEL_KO = {2: "위험", 1: "주의"}
AI_RGB = (255, 60, 220)        # damage AI outline (magenta: not one of the geometry class colours)
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fonts", "NotoSansCJKkr-Bold.otf")


_RAYS = {}


def rays(shape, K):
    """Cached unit-depth rays (x/z, y/z, 1) of every pixel, float32."""
    key = (shape, tuple(np.round(K, 3)))
    if key not in _RAYS:
        fx, fy, cx, cy = K
        u, v = np.meshgrid(np.arange(shape[1], dtype=np.float32), np.arange(shape[0], dtype=np.float32))
        _RAYS.clear()
        _RAYS[key] = np.dstack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)]).astype(np.float32)
    return _RAYS[key]


def backproject(depth, K):
    """depth HxW [m] -> HxWx3 points in the optical frame (z forward, x right, y down)."""
    return rays(depth.shape, K) * depth[:, :, None]


def _touching(mask, near):
    """Connected patches of `mask` that touch `near` (uint8 masks) -> bool mask of those patches."""
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros(mask.shape, bool)
    hit = np.zeros(n, bool)
    hit[np.unique(lab[near > 0])] = True
    hit[0] = False
    return hit[lab]


def segment(depth, K, R_lo, t_lo, planes, roi_x, p=PARAMS, thr_in=None):
    """depth [m], intrinsics K = (fx, fy, cx, cy), optical -> levelled (R_lo, t_lo), road planes from slope_core
    (list of (abcd, x_from, x_to); 1 plane, or 2 with a step). Returns class map, height map [m], thr, regions.
    Vectorised float32 + OpenCV morphology / connected components (~10x faster than the first scipy version)."""
    depth = depth.astype(np.float32)
    shape = depth.shape
    valid = np.isfinite(depth) & (depth > p["z_min"]) & (depth < p["z_max"])
    P = backproject(np.where(valid, depth, 0), K)
    L = sc.xform(P.reshape(-1, 3), R_lo, t_lo)
    X, Y = L[:, 0].reshape(shape), L[:, 1].reshape(shape)
    segment.last_xy = (X, Y)                         # levelled robot-frame position of every pixel (height map)
    if len(planes) == 1:
        m = planes[0][0].astype(np.float32)
        h = sc.pdist(L, m).reshape(shape)
    else:
        h = sc.heights_above(L.astype(np.float64), planes).astype(np.float32).reshape(shape)
    inroi = valid & (X >= roi_x[0]) & (X <= roi_x[1]) & (np.abs(Y) <= p["roi_y"])

    hf = h[inroi & (np.abs(h) < 0.02)]
    if len(hf) > 50:
        hf = hf[:: max(1, len(hf) // 5000)]
        sigma = 1.4826 * float(np.median(np.abs(hf - np.median(hf))))
    else:
        sigma = 0.005
    thr = float(np.clip(3 * sigma, p["thr_min"], p["thr_max"]))
    segment.last_thr_measured = thr                  # the live node smooths it over time and passes it back as thr_in
    if thr_in is not None:
        thr = float(thr_in)

    cls = np.zeros(shape, np.uint8)
    cls[inroi & (np.abs(h) <= thr)] = FLOOR
    cls[inroi & (h < -thr) & (h > -p["big"])] = SUNKEN
    cls[inroi & (h <= -p["big"])] = HOLE
    cls[inroi & (h > thr) & (h < p["big"])] = RAISED
    cls[inroi & (h >= p["big"]) & (h <= 2.0)] = OBSTACLE
    if len(planes) > 1:                              # pixels lying on the far level of a step
        m2, xa, _ = planes[1]
        on2 = inroi & (np.abs(sc.pdist(L, m2.astype(np.float32))).reshape(shape) <= thr) & (X >= xa - 0.15)
        cls[on2 & (cls != OBSTACLE)] = STEP

    # isolated noisy SUNKEN / RAISED pixels -> floor (opening 3x3)
    k3 = np.ones((3, 3), np.uint8)
    for kind in (SUNKEN, RAISED):
        mk = (cls == kind).astype(np.uint8)
        cls[(mk > 0) & (cv2.morphologyEx(mk, cv2.MORPH_OPEN, k3) == 0)] = FLOOR
    # pixels at the foot / edge of an obstacle mix floor and object depths: RAISED patches touching an obstacle
    # belong to it; SUNKEN patches touching it are "flying pixels" along its silhouette / the half-hidden floor
    # behind it (real test 2026-10-18: -8..-11 mm just behind a 6 cm box) -> dropped
    obs = (cls == OBSTACLE).astype(np.uint8)
    if obs.any():
        cls[_touching(cls == RAISED, cv2.dilate(obs, np.ones((5, 5), np.uint8)))] = OBSTACLE
        cls[_touching(cls == SUNKEN, cv2.dilate((cls == OBSTACLE).astype(np.uint8), np.ones((9, 9), np.uint8)))] = NONE

    # pixel footprint on the ground [m^2] ~ z^2 / (fx fy) / cos(incidence)
    n_o = (R_lo.T @ planes[0][0][:3]).astype(np.float32)          # floor normal in the optical frame
    R = rays(shape, K)
    cosi = np.abs(R @ n_o) / np.linalg.norm(R, axis=2)
    area_px = depth * depth / (K[0] * K[1]) / np.maximum(cosi, 0.2)

    regions = []
    for kind in (SUNKEN, RAISED, HOLE, OBSTACLE):
        mask = (cls == kind).astype(np.uint8)
        if not mask.any():
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = np.bincount(lab.ravel(), weights=np.nan_to_num(area_px).ravel(), minlength=n)
        for k in range(1, n):
            x0, y0, w, hh = stats[k, :4]
            sl = (slice(y0, y0 + hh), slice(x0, x0 + w))              # work inside the bounding box only
            r = lab[sl] == k
            if areas[k] < p["min_area_m2"]:
                if kind in (SUNKEN, RAISED):
                    cls[sl][r] = FLOOR
                continue
            hr = h[sl][r].astype(np.float64)
            xr, yr = X[sl][r], Y[sl][r]
            hval = sc.top_height(hr) if kind == OBSTACLE else float(np.median(hr))   # same height as slope_core
            regions.append({"kind": int(kind), "height_mm": float(1000 * hval),
                            "extreme_mm": float(1000 * (hr.min() if kind in (SUNKEN, HOLE) else hr.max())),
                            "area_m2": float(areas[k]), "x": float(xr.mean()), "y": float(yr.mean()),
                            "dist_m": float(np.hypot(xr, yr).min()),
                            "front_dist_m": float(max(0.0, xr.min() - sc.FRONT_OFFSET_M)),
                            "bbox": [int(x0), int(y0), int(x0 + w - 1), int(y0 + hh - 1)]})
    regions.sort(key=lambda g: g["dist_m"])
    return cls, h, thr, regions


_FONT_CACHE = {}


def _fonts(H):
    """Korean fonts for an image of height H, loaded once (the .otf is 17 MB: loading it per frame cost ~25 ms)."""
    if H not in _FONT_CACHE:
        try:
            _FONT_CACHE[H] = tuple(ImageFont.truetype(FONT, s) for s in (max(16, H // 24), max(14, H // 28), max(14, H // 26)))
        except OSError:
            _FONT_CACHE[H] = (ImageFont.load_default(),) * 3
    return _FONT_CACHE[H]


def level_of(g, levels):
    """StepMap level of a sunken / raised / hole region: 2 위험, 1 주의, 0 below (levels = (caution, danger) [m])."""
    if levels is None or g["kind"] == OBSTACLE:
        return 0
    if g["kind"] == HOLE:
        return 2
    a = abs(g["height_mm"]) / 1000.0
    return 2 if a >= levels[1] else 1 if a >= levels[0] else 0


def describe(g, levels=None):
    """One line of Korean text for a region: [level], kind, height (or depth), distance from the robot front."""
    lv = level_of(g, levels)
    pre = f"{LEVEL_KO[lv]} · " if lv else ""
    return pre + _describe(g)


def _describe(g):
    d = f"거리 {100 * g['front_dist_m']:.0f} cm"
    if g["kind"] == OBSTACLE:
        return f"장애물 · 높이 {g['height_mm'] / 10:.1f} cm · {d}"
    if g["kind"] == HOLE:
        return f"구멍 · 깊이 {abs(g['extreme_mm']) / 10:.1f} cm · {d}"
    if g["kind"] == SUNKEN:
        return f"함몰 · 깊이 {abs(g['height_mm']):.0f} mm · 면적 {1e4 * g['area_m2']:.0f} cm² · {d}"
    return f"돌출 · 높이 {g['height_mm']:.0f} mm · 면적 {1e4 * g['area_m2']:.0f} cm² · {d}"


_PANEL = {}


def ai_labels(found, max_items=4):
    """Pre-rendered Korean labels of damage AI findings (PIL once per AI result, pasted on every frame):
    [((x, y) top-left anchor in picture pixels, RGB bitmap)]. 파손: height, 균열: AI score; with a damage type
    (type_ko 파손 / 결손 / 이탈·분리) the type leads the label."""
    out = []
    if not found:
        return out
    f = _fonts(480)[1]
    for g in found[:max_items]:
        if g["kind_ko"] == "파손" and g.get("height_mm") is not None:
            t = f"파손 {abs(g['height_mm']) / 10:.1f} cm"
        else:
            t = f"균열 {g['score']:.2f}"
        if g.get("type_ko"):
            t = g["type_ko"] + (f" {abs(g['height_mm']) / 10:.1f} cm" if g.get("height_mm") is not None
                                else f" {g['score']:.2f}")
        w = int(f.getlength(t) if hasattr(f, "getlength") else f.getsize(t)[0]) + 12
        im = Image.new("RGB", (w, f.size + 10), AI_RGB)
        ImageDraw.Draw(im).text((6, 3), t, fill=(255, 255, 255), font=f)
        x0, y0 = g["bbox"][:2]
        out.append(((x0, y0), np.asarray(im)))
    return out


def overlay(rgb, cls, regions, thr, slope=None, alpha=0.40, max_items=5, panel_hz=5.0, note=None, levels=None,
            uneven_mm=None):
    """Camera image with the classified pixels coloured and each region boxed with a number; all text goes to an
    info panel UNDER the image (one line per region), so labels never overlap each other or the picture.
    Image part drawn with OpenCV, only the small Korean panel with PIL (fast enough for the camera rate).
    levels = (caution, danger) [m]: boxes of 위험 / 주의 regions in the StepMap level colours, thicker, and the
    level first in their panel line."""
    lut = np.zeros((256, 3), np.uint8)                                # class -> colour table
    for k, c in COLORS.items():
        lut[k] = c
    layer = np.dstack([cv2.LUT(cls, lut[:, i].copy()) for i in range(3)])   # OpenCV only: masked numpy
    blend = cv2.addWeighted(rgb, 1 - alpha, layer, alpha, 0)                # indexing took ~24 ms per frame
    img = rgb.copy()
    cv2.copyTo(blend, (cls > 0).astype(np.uint8), img)
    H, W = img.shape[:2]
    f_big, f, f_num = _fonts(H)
    items = regions[:max_items]
    # numbered boxes on the image (number inside the box corner, no long text)
    r = max(10, H // 40)
    for i, g in enumerate(items, 1):
        x0, y0, x1, y1 = g["bbox"]
        lv = level_of(g, levels)
        c = tuple(int(v) for v in (LEVEL_RGB[lv] if lv else COLORS[g["kind"]]))
        cv2.rectangle(img, (x0, y0), (x1, y1), c, 5 if lv == 2 else 3)
        cx, cy = min(max(x0 + r, r), W - r), min(max(y0 + r, r), H - r)
        cv2.circle(img, (cx, cy), r, c, -1)
        cv2.circle(img, (cx, cy), r, (255, 255, 255), 2)
        sc_ = r / 14.0
        (tw_, th_), _ = cv2.getTextSize(str(i), cv2.FONT_HERSHEY_SIMPLEX, sc_, 2)
        cv2.putText(img, str(i), (cx - tw_ // 2, cy + th_ // 2), cv2.FONT_HERSHEY_SIMPLEX, sc_, (0, 0, 0), 2, cv2.LINE_AA)
    # info panel (PIL: Korean text, ~20 ms) is re-rendered at most panel_hz times per second; the coloured image
    # itself is new every frame
    key = (W, len(items), note, levels is None, uneven_mm is None)
    now = time.time()
    cached = _PANEL.get(key)
    if cached is not None and now - cached[0] < 1.0 / panel_hz:
        return np.vstack([img, cached[1]])
    line = f.size + 8
    # FIXED height (room for max_items lines): a panel growing / shrinking with the number of regions changed the
    # image size every few frames and the viewer window jumped up and down
    panel_h = 12 + (f_big.size + 10) + line * max(1, max_items) + (f.size + 12)
    pan = Image.new("RGB", (W, panel_h), (18, 18, 22))
    d = ImageDraw.Draw(pan)

    def tw(t, font):
        return d.textlength(t, font=font) if hasattr(d, "textlength") else font.getsize(t)[0]

    y = 8
    head = f"경사도 {slope:.1f}°" if slope is not None else "경사도 –"
    if uneven_mm is not None:                        # one line at 640 px: the distance note makes room for 요철
        head += f"  ·  요철 {uneven_mm:.0f} mm  ·  판별 한계 ±{1000 * thr:.0f} mm"
    else:
        head += f"   ·   판별 한계 ±{1000 * thr:.0f} mm   ·   거리 = 로봇 앞부분 기준"
    if note:
        head = note
    d.text((10, y), head, fill=(255, 255, 255), font=f_big)
    y += f_big.size + 10
    if not items:
        d.text((10, y), "장애물 / 함몰 / 돌출 없음 — 정상 바닥", fill=(120, 220, 140), font=f)
        y += line
    for i, g in enumerate(items, 1):
        lv = level_of(g, levels)
        c = LEVEL_RGB[lv] if lv else COLORS[g["kind"]]
        rr = f.size // 2 + 2
        d.ellipse([10, y + 2, 10 + 2 * rr, y + 2 + 2 * rr], fill=c)
        d.text((10 + rr - tw(str(i), f) / 2, y), str(i), fill=(0, 0, 0), font=f)
        d.text((22 + 2 * rr, y), describe(g, levels), fill=c, font=f)
        y += line
    x = 10
    for k in (FLOOR, SUNKEN, HOLE, RAISED, OBSTACLE, STEP):
        d.rectangle([x, y + 4, x + 14, y + 4 + f.size - 6], fill=COLORS[k])
        name = NAMES_KO[k].split(" (")[0]
        d.text((x + 18, y), name, fill=(200, 200, 200), font=f)
        x += 30 + tw(name, f)
    pan = np.asarray(pan)
    _PANEL.clear()
    _PANEL[key] = (now, pan)
    return np.vstack([img, pan])


# ---------------------------------------------------------------------------------------------------------------------
# Damage view (StepMap, 2026-11-13): the live picture shows only what StepMap reports - the damage AI regions with
# their type (파손 / 결손 / 이탈·분리) and height, coloured by the StepMap level like the dashboard (위험 red, 주의 orange,
# below the caution height green). No floor / obstacle colouring (the geometry view is overlay(), live_view:=geometry).
OK_RGB = (165, 212, 140)                       # dashboard 정상 green
DAMAGE_TYPES = ("파손", "결손", "이탈·분리")
_DPANEL = {}


def damage_level(g, levels):
    """StepMap level of an AI region from its depth height: 2 위험, 1 주의, 0 below / no height."""
    if levels is None or g.get("height_mm") is None:
        return 0
    a = abs(g["height_mm"]) / 1000.0
    return 2 if a >= levels[1] else 1 if a >= levels[0] else 0


def damage_name(g):
    t = g.get("type_ko")
    return t if t in DAMAGE_TYPES else ("파손" if g.get("kind_ko") == "파손" else "손상")


def damage_text(g, levels):
    lv = damage_level(g, levels)
    h = g.get("height_mm")
    parts = [damage_name(g), f"{abs(h) / 10:.1f} cm" if h is not None else "높이 측정 안 됨",
             LEVEL_KO.get(lv, "기준 미만")]
    if g.get("x") is not None:
        parts.append(f"거리 {100 * g['x']:.0f} cm")
    return " · ".join(parts)


def damage_tags(found, levels, max_items=4):
    """Number tags of the AI regions, rendered once per AI result: [(anchor xy, RGB bitmap, colour)]."""
    out = []
    f = _fonts(480)[1]
    for i, g in enumerate(found[:max_items], 1):
        lv = damage_level(g, levels)
        c = LEVEL_RGB.get(lv, OK_RGB)
        h = g.get("height_mm")
        t = f"{i} {damage_name(g)}" + (f" {abs(h) / 10:.1f} cm" if h is not None else "")
        w = int(f.getlength(t) if hasattr(f, "getlength") else f.getsize(t)[0]) + 14
        im = Image.new("RGB", (w, f.size + 10), c)
        ImageDraw.Draw(im).text((7, 3), t, fill=(255, 255, 255) if lv else (30, 50, 30), font=f)
        out.append((tuple(g["bbox"][:2]), np.asarray(im), c))
    return out


def damage_overlay(rgb, ai, thr, slope=None, uneven_mm=None, levels=None, note=None, max_items=4, panel_hz=5.0,
                   range_m=(0.4, 1.2)):
    """ai = (labels or mask, found, tags, time) of the last damage AI result (or None); labels: int region ids
    (damage_ai.fuse_damage.labels, item "rid"), so touching regions are each drawn with their own pixels. Picture: each AI region filled lightly
    and outlined in its level colour with a number tag; panel under it: one line per region (type · height · level ·
    distance) and the level legend. Same panel height always (no jumping window)."""
    img = rgb.copy()
    H, W = img.shape[:2]
    found, tags, stamp = ([], [], 0.0) if ai is None else (ai[1][:max_items], ai[2], ai[3])
    if ai is not None:
        mask = ai[0]
        for g, (_, _, c) in zip(found, tags):
            x0, y0, x1, y1 = g["bbox"]
            win = mask[y0:y1 + 1, x0:x1 + 1]           # region labels (split regions: their own pixels only)
            sub = (win == g["rid"]).astype(np.uint8) if g.get("rid") and win.dtype != bool else win.astype(np.uint8)
            if not sub.any():
                continue
            roi = img[y0:y1 + 1, x0:x1 + 1]
            col = np.empty_like(roi)
            col[:] = c
            blend = cv2.addWeighted(roi, 0.6, col, 0.4, 0)
            cv2.copyTo(blend, sub, roi)
            cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(roi, cnts, -1, c, 3)
        placed = []                                  # tags of touching regions must not cover each other
        for (x, y), bmp, _ in tags:
            h_, w_ = bmp.shape[:2]
            x, y = int(min(max(0, x), W - w_)), int(min(max(0, y - h_ - 2), H - h_))
            for _ in range(8):
                hit = next((b for b in placed if x < b[2] and b[0] < x + w_ and y < b[3] and b[1] < y + h_), None)
                if hit is None:
                    break
                y = hit[3] + 2 if hit[3] + 2 + h_ <= H else max(0, hit[1] - h_ - 2)
            placed.append((x, y, x + w_, y + h_))
            img[y:y + h_, x:x + w_] = bmp
    f_big, f, _ = _fonts(H)
    key = ("damage", W, stamp, note, len(found))
    now = time.time()
    cached = _DPANEL.get("p")
    if cached is not None and cached[2] == key and now - cached[0] < 1.0 / panel_hz:
        return np.vstack([img, cached[1]])
    line = f.size + 8
    panel_h = 12 + (f_big.size + 10) + line * max_items + (f.size + 12)
    pan = Image.new("RGB", (W, panel_h), (18, 18, 22))
    d = ImageDraw.Draw(pan)

    def tw(t, font):
        return d.textlength(t, font=font) if hasattr(d, "textlength") else font.getsize(t)[0]

    y = 8
    head = note or (f"손상 탐지 (AI + 깊이)  ·  측정 범위 {range_m[0]:.1f}–{range_m[1]:.1f} m  ·  판별 한계 ±{1000 * thr:.0f} mm")
    d.text((10, y), head, fill=(255, 255, 255), font=f_big)
    y += f_big.size + 10
    if not note and not found:
        d.text((10, y), "손상 없음  (파손 · 결손 · 이탈·분리)", fill=OK_RGB, font=f)
    for i, (g, (_, _, c)) in enumerate(zip(found, tags), 1):
        rr = f.size // 2 + 2
        d.ellipse([10, y + 2, 10 + 2 * rr, y + 2 + 2 * rr], fill=c)
        d.text((10 + rr - tw(str(i), f) / 2, y), str(i), fill=(0, 0, 0), font=f)
        d.text((22 + 2 * rr, y), damage_text(g, levels), fill=c, font=f)
        y += line
    y = panel_h - (f.size + 10)
    x = 10
    lv_c = levels or (0.01, 0.02)
    for name, c in ((f"위험 ≥ {100 * lv_c[1]:.0f} cm", LEVEL_RGB[2]), (f"주의 ≥ {100 * lv_c[0]:.0f} cm", LEVEL_RGB[1]),
                    ("기준 미만", OK_RGB)):
        d.rectangle([x, y + 4, x + 14, y + 4 + f.size - 6], fill=c)
        d.text((x + 18, y), name, fill=(200, 200, 200), font=f)
        x += 34 + tw(name, f)
    extra = []
    if slope is not None:
        extra.append(f"경사 {slope:.1f}°")
    if uneven_mm is not None:
        extra.append(f"요철 {uneven_mm:.0f} mm")
    if extra:
        t = " · ".join(extra)
        d.text((W - 10 - tw(t, f), y), t, fill=(150, 150, 160), font=f)
    pan = np.asarray(pan)
    _DPANEL["p"] = (now, pan, key)
    return np.vstack([img, pan])
