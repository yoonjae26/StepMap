"""StepMap as it runs on the Go2 (2026-11-18): model A (damage_seg.onnx = runs/ft_b1_v2_A) finds damage ->
type model (damage_types.onnx = runs/types_v5/c1_oldA) only when A found damage -> depth fusion (damage_ai.fuse_damage:
depth step at the region -> 이탈·분리). Same ONNX files and code as the robot (copied, unchanged).
Test set: split_old 'val' of index_types_v5 = 200 Go2 pictures that neither model trained on (33 with damage).
Depth: the recorded D435 frame of the same picture; levelling by a RANSAC ground plane (the robot uses TF/IMU).
Per label: positive = picture labelled L; predicted = a kept region has type L. Also 'damage yes/no' (A alone).
Two variants: with depth (the robot) and without depth (type model only, the old table's setting).
python eval_system.py -> eval/system_table.json"""
import collections
import glob
import json
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root: models/, robot/
sys.path.insert(0, os.path.join(HERE, "robot"))
import damage_ai as dai  # noqa: E402

A = ort.InferenceSession(f"{HERE}/models/" + os.environ.get("AMODEL", "damage_seg.onnx"), providers=["CPUExecutionProvider"])
TY = ort.InferenceSession(f"{HERE}/models/damage_types.onnx", providers=["CPUExecutionProvider"])
TYPES = ("파손", "결손", "이탈·분리")


def run(sess, rgb):
    out = sess.run(None, {sess.get_inputs()[0].name: dai.preprocess(rgb, (480, 640))})[0][0]
    if out.max() > 1.0 or out.min() < 0.0:
        e = np.exp(out - out.max(0, keepdims=True))
        out = e / e.sum(0, keepdims=True)
    return out.astype(np.float32)


def depth_geom(sess_dir, sid, rng):
    kd = np.array(json.load(open(f"{sess_dir}/camera_info_depth.json"))["k"]).reshape(3, 3)
    kc = np.array(json.load(open(f"{sess_dir}/camera_info_color.json"))["k"]).reshape(3, 3)
    d = cv2.imread(f"{sess_dir}/depth/{sid}.png", cv2.IMREAD_UNCHANGED).astype(np.float32) * 0.001
    H, W = d.shape
    vv, uu = np.mgrid[0:H, 0:W]
    P = np.stack([(uu - kd[0, 2]) / kd[0, 0] * d, (vv - kd[1, 2]) / kd[1, 1] * d, d], -1)
    ok = (d > 0.3) & (d < 2.5)
    Q = P[ok & (d < 1.6)]
    if len(Q) < 2000:
        return None
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
    if best is None:
        return None
    a, n = best
    In = sub[np.abs((sub - a) @ n) < 0.008]
    c0 = In.mean(0)
    n = np.linalg.svd(In - c0, full_matrices=False)[2][2]
    n = n if n[1] < 0 else -n
    h = (P - c0) @ n
    h[~ok] = np.nan
    zax = np.array([0, 0, 1.0]); fwd = zax - (zax @ n) * n; fwd /= np.linalg.norm(fwd)
    foot = -((-c0) @ n) * n
    X = (P - foot) @ fwd + 0.30
    Pc = P - np.array([0.0147, 0, 0])
    z = np.where(ok, Pc[..., 2], 1)
    u = np.where(ok, np.round(Pc[..., 0] / z * kc[0, 0] + kc[0, 2]), -1).astype(int)
    v = np.where(ok, np.round(Pc[..., 1] / z * kc[1, 1] + kc[1, 2]), -1).astype(int)
    return u, v, h, X


def table(use, pred):
    N, out = len(use), {}
    for t in TYPES:
        pos = [t in e["types"] for e in use]
        pr = [t in pred[e["id"]] for e in use]
        tp = sum(p and q for p, q in zip(pos, pr)); fp = sum((not p) and q for p, q in zip(pos, pr))
        tn = sum((not p) and (not q) for p, q in zip(pos, pr)); P = sum(pos)
        sp = {e["spot"] for e, p in zip(use, pos) if p}
        hit = {e["spot"] for e, p, q in zip(use, pos, pr) if p and q}
        out[t] = {"pictures": P, "places": len(sp), "recall": round(100 * tp / max(1, P), 1),
                  "precision": round(100 * tp / max(1, tp + fp), 1), "accuracy": round(100 * (tp + tn) / N, 1),
                  "places_found": f"{len(hit)}/{len(sp)}"}
    return out


def main():
    it = json.load(open(f"{HERE}/eval/index_types_v5.json", encoding="utf-8"))
    use = [e for e in it if e["src"] == "robot" and e.get("split_old") == "val" and e["kind"] != "ign"]
    rng = np.random.default_rng(0)
    det, with_d, no_d = {}, {}, {}
    for e in use:
        sid = e["id"]
        sd = glob.glob(os.path.join(os.environ.get("STEPMAP_FIELD", "data/field"), f"images/robot/*_{sid[:5]}"))
        sd = [s for s in sd if os.path.exists(f"{s}/color/{sid}.jpg")][0]
        rgb = cv2.imread(f"{sd}/color/{sid}.jpg")[:, :, ::-1].copy()
        mask, pd = dai.damage_mask(run(A, rgb), rgb.shape[:2], 0.5)
        det[sid] = bool(mask.any())
        with_d[sid], no_d[sid] = set(), set()
        if not mask.any():
            continue
        ptype = run(TY, rgb)
        g = depth_geom(sd, sid, rng)
        if g is not None:
            u, v, h, X = g
            for f in dai.fuse_damage(mask, pd, u, v, h, X, np.zeros_like(X), 10.0, ptype=ptype, step_mm=6.0):
                with_d[sid].add(f["type_ko"])
        H, W = mask.shape                                        # no depth: u = -1 everywhere -> type model only
        dummy = np.full((240, 320), -1, int)
        for f in dai.fuse_damage(mask, pd, dummy, dummy, np.full((240, 320), np.nan), np.zeros((240, 320)),
                                 np.zeros((240, 320)), 10.0, ptype=ptype, step_mm=6.0):
            no_d[sid].add(f["type_ko"])
        if g is None:
            with_d[sid] = set(no_d[sid])
    pos = [e["kind"] == "pos" for e in use]
    pr = [det[e["id"]] for e in use]
    tp = sum(p and q for p, q in zip(pos, pr)); fp = sum((not p) and q for p, q in zip(pos, pr))
    yn = {"pictures": sum(pos), "recall": round(100 * tp / sum(pos), 1), "precision": round(100 * tp / max(1, tp + fp), 1),
          "accuracy": round(100 * (tp + sum((not p) and (not q) for p, q in zip(pos, pr))) / len(use), 1)}
    detL = {e["id"]: (set(TYPES) if det[e["id"]] else set()) for e in use}     # A found damage = counts for every label
    out = {"N": len(use), "damage_yes_no_modelA": yn, "modelA_per_label_any_damage": table(use, detL), "system_with_depth": table(use, with_d),
           "type_model_only_no_depth": table(use, no_d), "groups": dict(collections.Counter(e.get("group") for e in use))}
    json.dump(out, open(f"{HERE}/eval/" + os.environ.get("AOUT", "system_table.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
