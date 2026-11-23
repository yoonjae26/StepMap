#!/usr/bin/env python3
"""Check a trained damage model on pictures without labels: the user's examples (eval_user, every one shows broken /
cracked paving) and robot camera frames (eval_robot, firm lab floor and stairs: should show no damage).

  python eval_seg.py runs/b2/best.pt --out runs/b2/eval [--size 768] [--tta]

Per picture: share of the paving (surface_ok + damaged) predicted damaged and the largest damaged region; a contact
sheet with the damage drawn in red. Summary: user pictures with damage found (>= min_frac of the paving or a region
>= min_region of the picture) / robot frames with damage (false alarms) and the test-split damaged IoU (labelled).
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import train_seg as T


def load(ckpt, dev, base="nvidia/segformer-b2-finetuned-ade-512-512"):
    from transformers import SegformerConfig, SegformerForSemanticSegmentation
    d = os.path.dirname(ckpt)
    cfg = (SegformerConfig.from_pretrained(d) if os.path.isfile(os.path.join(d, "config.json"))
           else SegformerConfig.from_pretrained(base, num_labels=len(T.NAMES)))
    m = SegformerForSemanticSegmentation(cfg)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    return m.to(dev).eval()


@torch.no_grad()
def predict(model, img, size, dev, tta=False):
    """img RGB uint8 HxW -> class probabilities (6, H, W) at the picture size; short side resized to `size`."""
    H, W = img.shape[:2]
    s = size / min(H, W)
    x = cv2.resize(img, (int(round(W * s / 32) * 32), int(round(H * s / 32) * 32)), interpolation=cv2.INTER_AREA)
    xt = T.to_tensor(x)[None].to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = model(pixel_values=xt).logits.float()
        if tta:
            lg = lg + model(pixel_values=xt.flip(-1)).logits.float().flip(-1)
    return F.interpolate(lg, size=(H, W), mode="bilinear", align_corners=False).softmax(1)[0].cpu().numpy()


def damage_stats(prob, thr=0.5):
    pred = prob.argmax(0)
    dmg = (pred == T.DAMAGED) & (prob[T.DAMAGED] >= thr)
    paving = (pred == 1) | (pred == T.DAMAGED)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dmg.astype(np.uint8), connectivity=8)
    biggest = st[1:, cv2.CC_STAT_AREA].max() / dmg.size if n > 1 else 0.0
    return dmg, {"damaged_of_paving": round(float(dmg.sum() / max(1, paving.sum())), 4),
                 "largest_region": round(float(biggest), 4), "mean_conf": round(float(prob[T.DAMAGED][dmg].mean()) if dmg.any() else 0.0, 3)}


def draw(img, prob, dmg, title):
    pred = prob.argmax(0)
    col = T.COLORS[pred]
    out = img.copy()
    m = pred > 0
    out[m] = (0.65 * img[m] + 0.35 * col[m]).astype(np.uint8)
    cnts, _ = cv2.findContours(dmg.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cnts, -1, (255, 30, 30), max(2, img.shape[1] // 300))
    out = cv2.resize(out, (480, int(480 * img.shape[0] / img.shape[1])))
    cv2.putText(out, title, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def sheet(tiles, path, cols=4):
    h = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    g = np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])
    cv2.imwrite(path, g[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--base", default="nvidia/segformer-b2-finetuned-ade-512-512", help="architecture if no config.json")
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--thr", type=float, default=0.5, help="damaged probability threshold")
    ap.add_argument("--min_frac", type=float, default=0.03)
    ap.add_argument("--min_region", type=float, default=0.01)
    ap.add_argument("--test", type=int, default=600, help="labelled test pictures for the IoU (0 = skip)")
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(args.ckpt), "eval")
    os.makedirs(out, exist_ok=True)
    dev = "cuda"
    model = load(args.ckpt, dev, args.base)
    res = {}
    for group, pat in (("user", f"{T.WORK}/eval_user/*.png"), ("robot", f"{T.WORK}/eval_robot/**/*.jpg")):
        tiles, rows = [], []
        for f in sorted(glob.glob(pat, recursive=True)):
            img = cv2.imread(f, cv2.IMREAD_COLOR)[:, :, ::-1].copy()
            if max(img.shape[:2]) > 2000:
                k = 2000 / max(img.shape[:2])
                img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            size = args.size if group == "user" else min(args.size, 480)
            prob = predict(model, img, size, dev, args.tta)
            dmg, st = damage_stats(prob, args.thr)
            found = st["damaged_of_paving"] >= args.min_frac or st["largest_region"] >= args.min_region
            rows.append(dict(st, file=os.path.basename(f), found=bool(found)))
            tiles.append(draw(img, prob, dmg, f"{'DAMAGE' if found else 'ok'} {100 * st['damaged_of_paving']:.0f}%"))
        sheet(tiles, f"{out}/{group}.jpg")
        res[group] = {"n": len(rows), "found": sum(r["found"] for r in rows), "pictures": rows}
    if args.test:
        idx = json.load(open(f"{T.WORK}/damaged_index.json"))
        names = sorted(idx["test"])[:: max(1, len(idx["test"]) // args.test)]
        loader = torch.utils.data.DataLoader(T.Surface("test", names, train=False), batch_size=8, num_workers=16)
        res["test"] = T.evaluate(model, loader, dev)
        # picture level on the labelled test split: all pictures with damage + as many without
        pos = [n for n, v in idx["test"].items() if v > 0]
        neg = sorted(n for n, v in idx["test"].items() if v == 0)[:: max(1, (len(idx["test"]) - len(pos)) // len(pos))][:len(pos)]
        ds = T.Surface("test", pos + neg, train=False)
        tp = fp = 0
        for i in range(len(ds)):
            img, _ = ds.load(i)
            img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
            _, st = damage_stats(predict(model, np.ascontiguousarray(img), args.size, dev, args.tta), args.thr)
            found = st["damaged_of_paving"] >= args.min_frac or st["largest_region"] >= args.min_region
            tp += found and i < len(pos)
            fp += found and i >= len(pos)
        res["test_pictures"] = {"with_damage": len(pos), "found": int(tp), "recall": round(tp / len(pos), 3),
                                "without_damage": len(neg), "false_alarms": int(fp), "false_rate": round(fp / len(neg), 3)}
    json.dump(res, open(f"{out}/eval.json", "w"), indent=1)
    print(json.dumps({"user_found": f"{res['user']['found']}/{res['user']['n']}",
                      "robot_false": f"{res['robot']['found']}/{res['robot']['n']}",
                      "test_damaged_iou": (res.get("test") or {}).get("damaged_iou"),
                      "test_pictures": res.get("test_pictures")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
