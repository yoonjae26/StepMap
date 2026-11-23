"""Replay the robot's own damage AI on recorded 2026-11-13 frames (session 0023, 3 Hz colour + depth), drawn with the
robot's own damage_overlay (surface_seg.py) -> camera-card pictures as the live view showed them.
Models: models/damage_seg.onnx (model A, 6 classes) + models/damage_types.onnx (8 classes), copied from the robot
(~/terrain_dataset/models, the files it runs as TensorRT engines). Depth -> height above a RANSAC ground plane
(the robot levels with its TF/IMU; the plane fit is the offline stand-in). X, Y are not needed for the picture.
python replay_ai.py <session dir> <idx0> <idx1> <out dir>   -> out/<idx>.png + frames.json (idx, t)"""
import csv
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "robot"))
import damage_ai as dai  # noqa: E402
import surface_seg as ss  # noqa: E402

S, I0, I1, OUT = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
os.makedirs(OUT, exist_ok=True)
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
A = ort.InferenceSession(f"{HERE}/models/damage_seg.onnx", providers=["CPUExecutionProvider"])
TY = ort.InferenceSession(f"{HERE}/models/damage_types.onnx", providers=["CPUExecutionProvider"])


def run(sess, rgb):
    x = dai.preprocess(rgb, (480, 640))
    out = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
    if out.max() > 1.0 or out.min() < 0.0:                     # logits -> probabilities
        e = np.exp(out - out.max(0, keepdims=True))
        out = e / e.sum(0, keepdims=True)
    return out.astype(np.float32)


kd = np.array(json.load(open(f"{S}/camera_info_depth.json"))["k"]).reshape(3, 3)
kc = np.array(json.load(open(f"{S}/camera_info_color.json"))["k"]).reshape(3, 3)
rows = [r for r in csv.DictReader(open(f"{S}/frames.csv")) if I0 <= int(r["idx"]) <= I1]
rng = np.random.default_rng(0)
meta = []
for r in rows:
    bgr = cv2.imread(f"{S}/images/color/{r['color']}")
    rgb = bgr[:, :, ::-1].copy()
    d = cv2.imread(f"{S}/images/depth/{r['depth']}", cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.001
    H, W = d.shape
    vv, uu = np.mgrid[0:H, 0:W]
    P = np.stack([(uu - kd[0, 2]) / kd[0, 0] * d, (vv - kd[1, 2]) / kd[1, 1] * d, d], -1)
    ok = (d > 0.3) & (d < 2.5)
    Q = P[ok & (d < 1.6)]
    sub = Q[rng.choice(len(Q), min(8000, len(Q)), replace=False)]
    best, bn = None, 0
    for _ in range(300):
        a, b, c = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if abs(n[1]) < 0.6:
            continue
        cnt = int((np.abs((sub - a) @ n) < 0.008).sum())
        if cnt > bn:
            best, bn = (a, n), cnt
    a, n = best
    In = sub[np.abs((sub - a) @ n) < 0.008]
    c0 = In.mean(0)
    n = np.linalg.svd(In - c0, full_matrices=False)[2][2]
    n = n if n[1] < 0 else -n
    h = (P - c0) @ n                                             # metres above the plane
    h[~ok] = np.nan
    zax = np.array([0, 0, 1.0]); fwd = zax - (zax @ n) * n; fwd /= np.linalg.norm(fwd)   # optical axis on the ground
    foot = -((-c0) @ n) * n * 0 + (np.zeros(3) - ((np.zeros(3) - c0) @ n) * n)              # camera foot on the plane
    X = (P - foot) @ fwd + 0.30                                  # forward distance from the body centre (camera 0.30 m ahead)
    Pc = P - np.array([0.0147, 0, 0])
    z = np.where(ok, Pc[..., 2], 1)
    u = np.where(ok, np.round(Pc[..., 0] / z * kc[0, 0] + kc[0, 2]), -1).astype(int)
    v = np.where(ok, np.round(Pc[..., 1] / z * kc[1, 1] + kc[1, 2]), -1).astype(int)
    prob = run(A, rgb)
    mask, pd = dai.damage_mask(prob, rgb.shape[:2], 0.5)
    ptype = run(TY, rgb) if mask.any() else None
    found = dai.fuse_damage(mask, pd, u, v, h, X, np.zeros_like(X), 10.0, ptype=ptype, step_mm=6.0)
    levels = (0.010, 0.020)
    ai = (dai.fuse_damage.labels, found, ss.damage_tags(found, levels), float(r["t_jetson"]))
    img = ss.damage_overlay(rgb, ai, 0.015, None, None, levels=levels)
    cv2.imwrite(f"{OUT}/{int(r['idx']):06d}.png", img[:, :, ::-1])
    meta.append({"idx": int(r["idx"]), "t": float(r["t_jetson"]), "vx": float(r["vx"]),
                 "found": [(g.get("type_ko"), g.get("height_mm")) for g in found]})
    print(r["idx"], meta[-1]["found"])
json.dump(meta, open(f"{OUT}/frames.json", "w", encoding="utf-8"), ensure_ascii=False, indent=0)
