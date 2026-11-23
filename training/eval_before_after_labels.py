#!/usr/bin/env python3
"""Per-label detection BEFORE field data (AI Hub only model runs/b1) vs AFTER (c1 = AI Hub + field pictures),
on the same Go2 pictures (ft/index_types_v6.json, src robot, kind != ign; 2026-11-16, new file, nothing else changed).

b1 has no damage types: for b1 a label counts as "detected" when the picture's damage is found at all (the robot's
rule, finetune_robot.detect: 640x480, damaged argmax and p >= 0.5, opening 5 px, region >= 400 px). b1 never saw a
Go2 picture, so every picture is unseen. c1: out-of-fold predictions of the 5 place folds (runs/types_v6/c1s1000_f*,
step 1000), the same pictures; reported both as "damage found" (same rule as b1) and "right type".
Per label: pictures with it, found, recall, places found; plus false alarms on pictures without damage.
  CUDA_VISIBLE_DEVICES=0 python3 eval_before_after_labels.py [--out runs/before_after_labels.json]"""
import argparse
import collections
import json
import os

import cv2
import torch

import train_seg as T
from eval_seg import load
from finetune_robot import detect

DSV = os.environ.get("STEPMAP_FIELD", "data/field")
TYPES = ("파손", "결손", "이탈·분리")


@torch.no_grad()
def found(model, path, dev):
    img = cv2.imread(path, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
    x = T.to_tensor(cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA))[None].to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model(pixel_values=x).logits.float()
    return bool(detect(lg.softmax(1)[0].cpu().numpy())[0])


def summarize(use, det_of, type_of=None):
    out = {}
    for t in TYPES:
        rows = [e for e in use if t in e["types"]]
        hit = [e for e in rows if det_of(e)]
        spots, sf = set(), set()
        for e in rows:
            s = e.get("spot") or e.get("zone")
            spots.add(s)
            if det_of(e):
                sf.add(s)
        d = {"pictures": len(rows), "found": len(hit), "recall": round(len(hit) / max(1, len(rows)), 3),
             "places_found": "%d/%d" % (len(sf), len(spots))}
        if type_of is not None:
            right = [e for e in rows if t in type_of(e)]
            d["right_type"] = len(right)
            d["right_type_recall"] = round(len(right) / max(1, len(rows)), 3)
        out[t] = d
    pos = [e for e in use if e["kind"] == "pos"]
    neg = [e for e in use if e["kind"] != "pos"]
    out["손상(전체)"] = {"pictures": len(pos), "found": sum(map(det_of, pos)),
                       "recall": round(sum(map(det_of, pos)) / max(1, len(pos)), 3)}
    fa = sum(map(det_of, neg))
    out["정상(오탐)"] = {"pictures": len(neg), "false_alarms": fa, "false_rate": round(fa / max(1, len(neg)), 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/before_after_labels.json")
    ap.add_argument("--c1", default="runs/types_v6/c1s1000")
    ap.add_argument("--step", type=int, default=1000)
    a = ap.parse_args()
    idx = json.load(open(f"{DSV}/ft/index_types_v6.json", encoding="utf-8"))
    items = idx if isinstance(idx, list) else idx.get("items", [])
    use = [e for e in items if e["src"] == "robot" and e["kind"] != "ign"]
    res = {}
    for k in range(5):
        for line in open("%s_f%d/preds.jsonl" % (a.c1, k), encoding="utf-8"):
            r = json.loads(line)
            if r["step"] == a.step:
                res.update(r["res"])
    use = [e for e in use if e["id"] in res]                   # the same pictures for both models
    dev = "cuda"
    model = load("runs/b1/best.pt", dev, "nvidia/segformer-b1-finetuned-ade-512-512").to(dev).eval()
    b1 = {e["id"]: found(model, f"{DSV}/{e['img']}", dev) for e in use}
    out = {"pictures": len(use), "kinds": dict(collections.Counter(e["kind"] for e in use)),
           "before_aihub_only_b1": summarize(use, lambda e: b1[e["id"]]),
           "after_c1": summarize(use, lambda e: bool(res[e["id"]]["det"]), lambda e: res[e["id"]]["types"])}
    json.dump(out, open(a.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
