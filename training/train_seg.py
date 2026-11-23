#!/usr/bin/env python3
"""Sidewalk damage segmentation (SegFormer) on the AI Hub 서피스마스킹 data (sidewalk_package/surface).

Goal: find broken / cracked / missing paving (파손·균열) in the robot camera picture. 22 surface classes are merged into
6 classes for the robot:
  0 background (road, buildings, other)   1 surface_ok (sidewalk / alley surfaces, crosswalk, bike lane)
  2 damaged (sidewalk / alley / braille damaged)   3 braille_ok   4 fixture (manhole, tree zone, grating)
  5 caution (stairs, repair zone)
Fixtures are their own class so that manholes and gratings are learned as "not damage".

Robot view: the Go2 camera is ~0.3 m high and looks down 26 deg, the data was taken at eye height. Training crops
are therefore often zoomed in (random scale 1-3x) and perspective-warped so close-up tiles look like the robot view.
Images with damage are drawn 4x more often (only ~8 % of images contain any); loss = class-weighted CE + Dice on the
damaged class. Validation: per-class IoU on val (full images at 960x540) every eval_every steps; best damaged IoU
is kept (best.pt).

  python train_seg.py --model nvidia/segformer-b2-finetuned-ade-512-512 --out runs/b2 --steps 20000
"""
import argparse
import json
import math
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = os.environ.get("STEPMAP_AIHUB", "data/aihub_surface")   # AI Hub 인도보행 영상 · 서피스마스킹
WORK = os.environ.get("STEPMAP_WORK", ".")
NAMES = ["background", "surface_ok", "damaged", "braille_ok", "fixture", "caution"]
COLORS = np.array([[0, 0, 0], [70, 200, 90], [239, 60, 60], [250, 210, 40], [60, 120, 255], [200, 80, 255]], np.uint8)
LUT = np.full(256, 255, np.uint8)
for src, dst in {0: 0, 12: 0, 13: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 8: 1, 9: 1, 10: 1, 14: 1,
                 7: 2, 11: 2, 16: 2, 15: 3, 17: 4, 18: 4, 19: 4, 20: 5, 21: 5}.items():
    LUT[src] = dst
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255
DAMAGED = 2


def to_tensor(img):
    return torch.from_numpy(((img.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1).copy())


class Surface(Dataset):
    def __init__(self, split, names, size=512, train=True):
        self.split, self.names, self.size, self.train = split, names, size, train

    def __len__(self):
        return len(self.names)

    def load(self, i):
        n = self.names[i]
        img = cv2.imread(f"{ROOT}/images/{self.split}/{n}.jpg", cv2.IMREAD_COLOR)[:, :, ::-1]
        m = LUT[cv2.imread(f"{ROOT}/masks/{self.split}/{n}.png", cv2.IMREAD_UNCHANGED)]
        return img, m

    def __getitem__(self, i):
        img, m = self.load(i)
        if not self.train:
            img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
            m = cv2.resize(m, (960, 540), interpolation=cv2.INTER_NEAREST)
            return to_tensor(img), torch.from_numpy(m.astype(np.int64))
        return self.augment(img, m)

    def augment(self, img, m):
        H, W = m.shape
        S = self.size
        # zoom: crop side = S..W/1 scaled; zoomed-in crops (small side) look like the low robot camera
        base = min(H, W)
        side = int(base / random.uniform(1.0, 3.0))
        side = max(side, S // 2)
        # centre the crop on damage half of the time (if any)
        dm = np.argwhere(m[::8, ::8] == DAMAGED)
        if len(dm) and random.random() < 0.5:
            cy, cx = dm[random.randrange(len(dm))] * 8
            y0 = int(np.clip(cy - side // 2 + random.randint(-side // 4, side // 4), 0, H - side))
            x0 = int(np.clip(cx - side // 2 + random.randint(-side // 4, side // 4), 0, W - side))
        else:
            y0, x0 = random.randint(0, H - side), random.randint(0, W - side)
        # perspective: the crop is a trapezoid (top wider = looking down at the ground)
        k = random.uniform(0.0, 0.25) * side
        src = np.float32([[x0 - k, y0], [x0 + side + k, y0], [x0 + side, y0 + side], [x0, y0 + side]])
        dst = np.float32([[0, 0], [S, 0], [S, S], [0, S]])
        P = cv2.getPerspectiveTransform(src, dst)
        img = cv2.warpPerspective(img, P, (S, S), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        m = cv2.warpPerspective(m, P, (S, S), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        if random.random() < 0.5:
            img, m = img[:, ::-1], m[:, ::-1]
        img = img.astype(np.float32)
        img = img * random.uniform(0.6, 1.4) + random.uniform(-30, 30)            # brightness / contrast
        if random.random() < 0.3:                                                   # colour cast
            img = img * np.random.uniform(0.85, 1.15, 3)
        if random.random() < 0.3:                                                   # motion / focus blur
            kk = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (kk, kk), 0)
        if random.random() < 0.3:                                                   # sensor noise
            img = img + np.random.normal(0, random.uniform(2, 8), img.shape)
        img = np.clip(img, 0, 255)
        return to_tensor(img), torch.from_numpy(np.ascontiguousarray(m).astype(np.int64))


def dice_loss(logits, y, cls=DAMAGED):
    p = logits.softmax(1)[:, cls]
    valid = (y != 255).float()
    t = (y == cls).float() * valid
    p = p * valid
    inter = (p * t).sum((1, 2))
    return (1 - (2 * inter + 1) / (p.sum((1, 2)) + t.sum((1, 2)) + 1)).mean()


@torch.no_grad()
def evaluate(model, loader, dev, max_batches=None):
    model.eval()
    conf = torch.zeros(len(NAMES), len(NAMES), dtype=torch.int64, device=dev)
    for b, (x, y) in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(pixel_values=x).logits
        pred = F.interpolate(lg.float(), size=y.shape[-2:], mode="bilinear", align_corners=False).argmax(1)
        k = y != 255
        conf += torch.bincount(y[k] * len(NAMES) + pred[k], minlength=len(NAMES) ** 2).view(len(NAMES), -1)
    model.train()
    c = conf.double()
    iou = (c.diag() / (c.sum(0) + c.sum(1) - c.diag()).clamp(min=1)).tolist()
    prec = (c[DAMAGED, DAMAGED] / c[:, DAMAGED].sum().clamp(min=1)).item()
    rec = (c[DAMAGED, DAMAGED] / c[DAMAGED, :].sum().clamp(min=1)).item()
    return {"iou": dict(zip(NAMES, [round(v, 4) for v in iou])), "miou": round(float(np.mean(iou)), 4),
            "damaged_iou": round(iou[DAMAGED], 4), "damaged_precision": round(prec, 4), "damaged_recall": round(rec, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/segformer-b2-finetuned-ade-512-512")
    ap.add_argument("--out", default=f"{WORK}/runs/b2")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-5)
    ap.add_argument("--damaged_weight", type=float, default=4.0, help="sampling weight of images with damage")
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--init", default="", help="start from this checkpoint (fine-tuning)")
    ap.add_argument("--cw_damaged", type=float, default=4.0, help="cross-entropy weight of the damaged class")
    ap.add_argument("--dice", type=float, default=0.5, help="weight of the Dice loss on the damaged class")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    random.seed(0)
    dev = "cuda"
    idx = json.load(open(f"{WORK}/damaged_index.json"))
    tr_names = sorted(idx["train"])
    w = [args.damaged_weight if idx["train"][n] > 0 else 1.0 for n in tr_names]
    sampler = WeightedRandomSampler(w, num_samples=args.steps * args.batch, replacement=True)
    train = DataLoader(Surface("train", tr_names, args.size), batch_size=args.batch, sampler=sampler,
                       num_workers=args.workers, pin_memory=True, drop_last=True, persistent_workers=True)
    va_names = sorted(idx["val"])
    val = DataLoader(Surface("val", va_names, train=False), batch_size=8, num_workers=16, pin_memory=True)

    from transformers import SegformerForSemanticSegmentation
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model, num_labels=len(NAMES), ignore_mismatched_sizes=True,
        id2label=dict(enumerate(NAMES)), label2id={n: i for i, n in enumerate(NAMES)}).to(dev)
    model.config.save_pretrained(args.out)              # eval_seg.py can load best.pt while training runs
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=dev))
    head = [p for n, p in model.named_parameters() if n.startswith("decode_head")]
    body = [p for n, p in model.named_parameters() if not n.startswith("decode_head")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.lr * 10}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 500) * (1 - s / args.steps) ** 0.9)
    cw = torch.tensor([0.5, 1.0, args.cw_damaged, 1.5, 2.0, 2.0], device=dev)   # damaged pixels are rare
    best, log = -1.0, open(f"{args.out}/log.jsonl", "a")
    t0, step = time.time(), 0
    model.train()
    for x, y in train:
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(pixel_values=x).logits
        lg = F.interpolate(lg.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
        loss = F.cross_entropy(lg, y, weight=cw, ignore_index=255) + args.dice * dice_loss(lg, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} {(time.time() - t0) / step:.3f} s/step", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            r = evaluate(model, val, dev)
            r.update(step=step, loss=round(loss.item(), 4), minutes=round((time.time() - t0) / 60, 1))
            print(json.dumps(r), flush=True)
            log.write(json.dumps(r) + "\n")
            log.flush()
            torch.save(model.state_dict(), f"{args.out}/last.pt")
            if r["damaged_iou"] > best:
                best = r["damaged_iou"]
                torch.save(model.state_dict(), f"{args.out}/best.pt")
                json.dump(r, open(f"{args.out}/best.json", "w"), indent=1)
        if step >= args.steps:
            break
    model.config.save_pretrained(args.out)
    print("done, best damaged IoU", best, flush=True)


if __name__ == "__main__":
    main()
