#!/usr/bin/env python3
"""Fine-tune the B1 damage model for the robot at night: AI Hub 서피스마스킹 + dataset_v2 (robot 2026-10-26 + user pictures).

Half of every batch is AI Hub (keeps what B1 knows; half of those pictures are darkened / blurred like the robot at
night), half is dataset_v2 (ft/index.json, tools/build_ft_set.py): pos / neg in equal parts.
Most damaged dataset_v2 pictures have no checked polygon, so two losses:
  pixel  CE (class weights) + Dice on damaged, where a mask exists (255 = ignore)
  image  BCE on the mean of the top 1 % damaged probabilities: high for damage pictures, low for no-damage pictures
Robot pictures keep their whole view (crop >= 85 %, resized to 512 x 384, padded to 512): the damage of an image-only
picture must stay inside the crop.

Validation (dataset_v2 split = val, by place) with the robot's own rule (damage_ai.damage_mask: 480 x 640 input,
damaged argmax and p >= 0.5, opening 5 px, region >= 400 px): a picture is "detected" when one region survives.
Damage recall on D, false alarms on N, F1 -> best.pt; AI Hub val damaged IoU (1,200 pictures) checks forgetting.
Localisation (frames with loc_mask = depth damage mask, tools/depth_masks.py): of the damaged pixels predicted on the
labelled area, the share on depth damage (loc_precision); of the depth damage, the share predicted (loc_recall).
--index ft/index_depth.json also trains on those depth masks. --eval_only scores --init without training.

  CUDA_VISIBLE_DEVICES=0 python finetune_robot.py --init runs/b1/best.pt --out runs/ft_b1_v2 [--no_blur_pos]
"""
import argparse
import json
import os
import random
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

import train_seg as T

DS = os.environ.get("STEPMAP_FIELD", "data/field")   # Go2 + smartphone field data
S, RW, RH = 512, 512, 384


def motion_blur(img, n):
    k = np.zeros((n, n), np.float32)
    k[n // 2] = 1.0
    k = cv2.warpAffine(k, cv2.getRotationMatrix2D((n / 2 - 0.5, n / 2 - 0.5), random.uniform(0, 180), 1.0), (n, n))
    return cv2.filter2D(img, -1, k / max(k.sum(), 1e-6))


def night(img):
    """float32 RGB 0..255 -> the robot's night look: dark, warm street light, noise, sometimes motion blur."""
    img = 255.0 * (img / 255.0) ** random.uniform(1.2, 2.2) * random.uniform(0.25, 0.6)
    img = img * np.array([random.uniform(1.0, 1.2), 1.0, random.uniform(0.75, 0.95)], np.float32)
    if random.random() < 0.4:
        img = motion_blur(img, random.choice([5, 7, 9, 11, 15]))
    return img + np.random.normal(0, random.uniform(3, 10), img.shape)


class AiHub(T.Surface):
    def __getitem__(self, i):
        x, y = super().__getitem__(i)                    # crop / perspective / colour from train_seg
        if random.random() < 0.5:
            img = x.numpy().transpose(1, 2, 0) * T.STD + T.MEAN
            x = T.to_tensor(np.clip(night(img), 0, 255))
        return x, y, torch.ones(y.shape, dtype=torch.bool), torch.tensor(-1.0)


class Robot(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        e = self.items[i]
        img = cv2.imread(f"{DS}/{e['img']}", cv2.IMREAD_COLOR)[:, :, ::-1]
        H, W = img.shape[:2]
        m = (cv2.imread(f"{DS}/{e['mask']}", cv2.IMREAD_UNCHANGED) if e["mask"]
             else np.full((H, W), 255, np.uint8))
        ch, cw = int(H * random.uniform(0.85, 1.0)), int(W * random.uniform(0.85, 1.0))
        y0, x0 = random.randint(0, H - ch), random.randint(0, W - cw)
        img = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (RW, RH), interpolation=cv2.INTER_AREA)
        m = cv2.resize(m[y0:y0 + ch, x0:x0 + cw], (RW, RH), interpolation=cv2.INTER_NEAREST)
        if random.random() < 0.5:
            img, m = img[:, ::-1], m[:, ::-1]
        img = img.astype(np.float32) * random.uniform(0.7, 1.3) + random.uniform(-15, 15)
        if random.random() < 0.25:
            img = motion_blur(img, random.choice([3, 5, 7, 9]))
        if random.random() < 0.3:
            img = img + np.random.normal(0, random.uniform(2, 6), img.shape)
        x = np.zeros((S, S, 3), np.float32) + T.MEAN       # pad to 512 x 512 (normalises to 0)
        x[:RH] = np.clip(img, 0, 255)
        y = np.full((S, S), 255, np.uint8)
        y[:RH] = m
        valid = np.zeros((S, S), bool)
        valid[:RH] = True
        lab = 1.0 if e["kind"].startswith("pos") else 0.0
        return T.to_tensor(x), torch.from_numpy(y.astype(np.int64)), torch.from_numpy(valid), torch.tensor(lab)


def image_loss(lg, valid, lab, top=0.01):
    """Top-1 % damaged probability per picture vs. the picture label (-1 = no image label)."""
    use = lab >= 0
    if not use.any():
        return lg.new_zeros(())
    p = lg[use].float().softmax(1)[:, T.DAMAGED]
    v = F.interpolate(valid[use, None].float(), size=p.shape[-2:], mode="nearest")[:, 0] > 0
    p = p.masked_fill(~v, -1.0).flatten(1)
    k = max(1, int(top * v[0].sum()))
    s = p.topk(k, dim=1).values.mean(1).clamp(1e-4, 1 - 1e-4)
    return F.binary_cross_entropy(s, lab[use])


def detect(prob, thr=0.5, min_area=400, mask=False):
    """damage_ai.damage_mask on a (6, h, w) probability map at the robot input size 480 x 640 -> detected, area
    (mask=True: the kept damaged mask itself)."""
    p = cv2.resize(prob[T.DAMAGED], (640, 480), interpolation=cv2.INTER_LINEAR)
    am = cv2.resize(prob.argmax(0).astype(np.uint8), (640, 480), interpolation=cv2.INTER_NEAREST)
    m = cv2.morphologyEx(((am == T.DAMAGED) & (p >= thr)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if mask:
        keep = np.zeros(n, bool)
        keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area
        return keep[lab]
    area = int(st[1:, cv2.CC_STAT_AREA][st[1:, cv2.CC_STAT_AREA] >= min_area].sum()) if n > 1 else 0
    return area > 0, area


@torch.no_grad()
def eval_robot(model, items, dev):
    model.eval()
    res = {}
    loc = np.zeros(3)                                    # predicted on damage, predicted on labelled area, damage
    for e in items:
        img = cv2.imread(f"{DS}/{e['img']}", cv2.IMREAD_COLOR)[:, :, ::-1]
        x = T.to_tensor(cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA))[None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(pixel_values=x).logits.float()
        pr = lg.softmax(1)[0].cpu().numpy()
        det, area = detect(pr)
        res[e["id"]] = (e["src"], e["kind"].startswith("pos"), det, area)
        if e.get("loc_mask"):
            lm = cv2.resize(cv2.imread(f"{DS}/{e['loc_mask']}", cv2.IMREAD_UNCHANGED), (640, 480), interpolation=cv2.INTER_NEAREST)
            pd = detect(pr, mask=True)
            loc += [(pd & (lm == 2)).sum(), (pd & (lm != 255)).sum(), (lm == 2).sum()]
    model.train()
    out = {}
    for name, src in (("robot", ("robot",)), ("user", ("user",)), ("all", ("robot", "user"))):
        r = [v for v in res.values() if v[0] in src]
        tp = sum(1 for v in r if v[1] and v[2])
        fn = sum(1 for v in r if v[1] and not v[2])
        fp = sum(1 for v in r if not v[1] and v[2])
        tn = sum(1 for v in r if not v[1] and not v[2])
        rec, prec = tp / max(1, tp + fn), tp / max(1, tp + fp)
        out[name] = {"recall": round(rec, 3), "false_alarm": round(fp / max(1, fp + tn), 3),
                     "precision": round(prec, 3), "f1": round(2 * rec * prec / max(1e-9, rec + prec), 3),
                     "tp": tp, "fn": fn, "fp": fp, "tn": tn}
    out["loc_precision"] = round(loc[0] / max(1, loc[1]), 3)
    out["loc_recall"] = round(loc[0] / max(1, loc[2]), 3)
    return out, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/segformer-b1-finetuned-ade-512-512")
    ap.add_argument("--init", default=f"{T.WORK}/runs/b1/best.pt")
    ap.add_argument("--out", default=f"{T.WORK}/runs/ft_b1_v2")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--img_w", type=float, default=0.5, help="weight of the image loss")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--no_blur_pos", action="store_true", help="leave out the 21 blurred damage frames")
    ap.add_argument("--index", default="ft/index.json", help="dataset_v2 list (ft/index_depth.json: + depth masks)")
    ap.add_argument("--eval_index", default="ft/index_depth.json", help="val list (has the loc masks)")
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    dev = "cuda"
    items = json.load(open(f"{DS}/{args.index}"))
    if args.no_blur_pos:
        items = [e for e in items if not e["blur_pos"]]
    tr = [e for e in items if e["split"] == "train"]
    va = [e for e in json.load(open(f"{DS}/{args.eval_index}")) if e["split"] == "val"
          and not (args.no_blur_pos and e["blur_pos"])]
    idx = json.load(open(f"{T.WORK}/damaged_index.json"))
    hub_names = sorted(idx["train"])
    hub_w = np.array([4.0 if idx["train"][n] > 0 else 1.0 for n in hub_names])
    pos = np.array([e["kind"].startswith("pos") for e in tr])
    rob_w = np.where(pos, 0.25 / pos.sum(), 0.25 / (~pos).sum())
    w = np.concatenate([0.5 * hub_w / hub_w.sum(), rob_w])
    data = ConcatDataset([AiHub("train", hub_names, S), Robot(tr)])
    sampler = WeightedRandomSampler(w.tolist(), num_samples=args.steps * args.batch, replacement=True)
    train = DataLoader(data, batch_size=args.batch, sampler=sampler, num_workers=args.workers, pin_memory=True,
                       drop_last=True, persistent_workers=True)
    va_names = sorted(idx["val"])
    random.Random(1).shuffle(va_names)
    hub_val = DataLoader(T.Surface("val", va_names[:1200], train=False), batch_size=8, num_workers=16)
    print(f"train: AI Hub {len(hub_names)}, dataset_v2 {len(tr)} (pos {pos.sum()}); val dataset_v2 {len(va)}", flush=True)

    from transformers import SegformerForSemanticSegmentation
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model, num_labels=len(T.NAMES), ignore_mismatched_sizes=True,
        id2label=dict(enumerate(T.NAMES)), label2id={n: i for i, n in enumerate(T.NAMES)}).to(dev)
    model.config.save_pretrained(args.out)
    model.load_state_dict(torch.load(args.init, map_location=dev))
    head = [p for n, p in model.named_parameters() if n.startswith("decode_head")]
    body = [p for n, p in model.named_parameters() if not n.startswith("decode_head")]
    opt = torch.optim.AdamW([{"params": body, "lr": args.lr}, {"params": head, "lr": args.lr * 10}], weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * (1 - s / args.steps) ** 0.9)
    cw = torch.tensor([0.5, 1.0, 4.0, 1.5, 2.0, 2.0], device=dev)
    log = open(f"{args.out}/log.jsonl", "a")

    def run_eval(step, loss=None):
        rv, per = eval_robot(model, va, dev)
        hub = T.evaluate(model, hub_val, dev)
        r = {"step": step, "val": rv, "aihub_damaged_iou": hub["damaged_iou"], "aihub_miou": hub["miou"],
             "loss": loss, "minutes": round((time.time() - t0) / 60, 1)}
        print(json.dumps(r), flush=True)
        log.write(json.dumps(r) + "\n")
        log.flush()
        return r, per

    t0 = time.time()
    r0, per0 = run_eval(0)
    json.dump({"metrics": r0, "per_picture": per0}, open(f"{args.out}/baseline.json", "w"), indent=0)
    if args.eval_only:
        return
    best, step = -1.0, 0
    model.train()
    for x, y, valid, lab in train:
        x, y, valid, lab = (t.to(dev, non_blocking=True) for t in (x, y, valid, lab))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(pixel_values=x).logits
        up = F.interpolate(lg.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
        loss_pix = F.cross_entropy(up, y, weight=cw, ignore_index=255) if (y != 255).any() else up.sum() * 0
        loss = loss_pix + 0.5 * T.dice_loss(up, y) + args.img_w * image_loss(lg, valid, lab)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} {(time.time() - t0) / step:.3f} s/step", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            r, per = run_eval(step, round(loss.item(), 4))
            torch.save(model.state_dict(), f"{args.out}/last.pt")
            score = r["val"]["all"]["f1"]
            if score > best:
                best = score
                torch.save(model.state_dict(), f"{args.out}/best.pt")
                json.dump({"metrics": r, "per_picture": per}, open(f"{args.out}/best.json", "w"), indent=0)
        if step >= args.steps:
            break
    print("done, best val F1", best, flush=True)


if __name__ == "__main__":
    main()
