#!/usr/bin/env python3
"""Fine-tune the B1 damage model to tell the three damage types apart: 파손 / 결손 / 이탈·분리.

8 classes: the 6 of train_seg.NAMES with "damaged" (2) split in three (dataset_v3/tools/import_types_v5.py):
  0 background  1 surface_ok  2 파손  3 braille_ok  4 fixture  5 caution  6 결손  7 이탈·분리
The 6-class checkpoint (--init) is widened: the 결손 / 이탈·분리 output channels start as copies of "damaged".
Training batches: half AI Hub (only "damage of any type" is known there: its loss merges the three type logits with
logsumexp, so AI Hub keeps teaching damage vs. surface without saying which type), half robot / user pictures from
ft/index_types_v5.json (one quarter damage, one quarter no damage). Damage pictures are drawn per place and type:
every (damage place, type) pair gets the same weight, shared by the pictures of that place, so a place seen in 50
frames does not count 50 times. No-damage pictures keep their old 6-class masks: their "damaged" pixels are ignored.
Losses: pixel CE over the 8 classes (255 = ignore) + Dice on "any damage" + image BCE (top 1 % any-damage probability
vs. the picture label), as in finetune_robot.py.

Split (--split): fold<k>  train on the other folds, validate on fold k (5 folds by place, split_types_v5.py)
                 old      train on split_old = train, validate on split_old = val (the places model A never saw)
Validation, robot rule (damage_ai.damage_mask at 480 x 640): damage pixels = argmax in {2, 6, 7} and any-damage
probability >= 0.5, opening 5 px, regions >= 400 px. A picture is detected when a region survives; each region gets
the type with the highest summed probability over it, the picture's predicted types are those of its regions.
Reported: any-damage recall / false alarm / F1 (comparable with model A) and per type recall / precision / F1 at the
picture level and recall at the place level. Every evaluation's per-picture predictions go to preds.jsonl so the
folds can be pooled at one common step (picking the step per fold on its own val would be optimistic).
--eval_only with a 6-class --init scores any-damage only (the old model A).

  CUDA_VISIBLE_DEVICES=0 python finetune_types.py --split fold0 --init runs/b1/best.pt --out runs/types_v5/b1_f0
"""
import argparse
import collections
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
from finetune_robot import DS, RH, RW, S, AiHub, motion_blur

NAMES = ["background", "surface_ok", "파손", "braille_ok", "fixture", "caution", "결손", "이탈·분리"]
TYPES = {"파손": 2, "결손": 6, "이탈·분리": 7}
DMG = [2, 6, 7]


class Robot(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        e = self.items[i]
        img = cv2.imread(f"{DS}/{e['img']}", cv2.IMREAD_COLOR)[:, :, ::-1]
        H, W = img.shape[:2]
        if e["mask"]:
            m = cv2.imread(f"{DS}/{e['mask']}", cv2.IMREAD_UNCHANGED)
            if e["old_mask"]:
                m = m.copy()
                m[m == 2] = 255                                # old 6-class mask of a no-damage picture
        else:
            m = np.full((H, W), 255, np.uint8)
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
        x = np.zeros((S, S, 3), np.float32) + T.MEAN
        x[:RH] = np.clip(img, 0, 255)
        y = np.full((S, S), 255, np.uint8)
        y[:RH] = m
        valid = np.zeros((S, S), bool)
        valid[:RH] = True
        lab = {"pos": 1.0, "neg": 0.0}.get(e["kind"], -1.0)       # "ign" (only 무시 left): no picture label
        return T.to_tensor(x), torch.from_numpy(y.astype(np.int64)), torch.from_numpy(valid), torch.tensor(lab), \
            torch.tensor(0)


class Hub(AiHub):
    def __getitem__(self, i):
        x, y, valid, lab = super().__getitem__(i)
        return x, y, valid, lab, torch.tensor(1)


def any_damage(lg):
    """8-class logits -> 6-class logits with the three types merged (logsumexp) into class 2."""
    d = torch.logsumexp(lg[:, DMG], 1, keepdim=True)
    return torch.cat([lg[:, :2], d, lg[:, 3:6]], 1)


def losses(lg, y, valid, lab, hub, cw8, cw6, img_w):
    up = F.interpolate(lg.float(), size=y.shape[-2:], mode="bilinear", align_corners=False)
    loss = up.sum() * 0
    if (~hub).any():
        yr = y[~hub]
        if (yr != 255).any():
            loss = loss + F.cross_entropy(up[~hub], yr, weight=cw8, ignore_index=255)
    if hub.any():
        yh = y[hub]
        if (yh != 255).any():
            loss = loss + F.cross_entropy(any_damage(up[hub]), yh, weight=cw6, ignore_index=255)
    # Dice on any damage: target 2 for AI Hub, {2, 6, 7} for the robot pictures
    m6 = any_damage(up)
    ya = y.clone()
    ya[(~hub)[:, None, None].expand_as(y) & ((y == 6) | (y == 7))] = 2
    loss = loss + 0.5 * T.dice_loss(m6, ya)
    # picture label
    use = lab >= 0
    if use.any():
        p = m6[use].softmax(1)[:, 2]
        v = valid[use]
        p = p.masked_fill(~v, -1.0).flatten(1)
        k = max(1, int(0.01 * v[0].sum()))
        s = p.topk(k, dim=1).values.mean(1).clamp(1e-4, 1 - 1e-4)
        loss = loss + img_w * F.binary_cross_entropy(s, lab[use])
    return loss


def regions(prob, ncls, thr=0.5, min_area=400):
    """prob (C, 480, 640) -> list of (area, type name or None) for the regions the robot rule keeps."""
    if ncls == 8:
        pany = prob[DMG].sum(0)
        # argmax over the merged classes: damage split over three types must not lose to surface_ok
        merged = np.concatenate([prob[:2], pany[None], prob[3:6]], 0)
        dmg_arg = merged.argmax(0) == 2
    else:
        pany = prob[2]
        dmg_arg = prob.argmax(0) == 2
    m = cv2.morphologyEx((dmg_arg & (pany >= thr)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < min_area:
            continue
        t = None
        if ncls == 8:
            r = lab == k
            t = list(TYPES)[int(np.argmax([prob[c][r].sum() for c in DMG]))]
        out.append((int(st[k, cv2.CC_STAT_AREA]), t))
    return out


@torch.no_grad()
def predict(model, items, dev, ncls):
    model.eval()
    res = {}
    for e in items:
        img = cv2.imread(f"{DS}/{e['img']}", cv2.IMREAD_COLOR)[:, :, ::-1]
        x = T.to_tensor(cv2.resize(img, (640, 480), interpolation=cv2.INTER_AREA))[None].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(pixel_values=x).logits.float()
        pr = F.interpolate(lg, size=(480, 640), mode="bilinear", align_corners=False).softmax(1)[0].cpu().numpy()
        rg = regions(pr, ncls)
        res[e["id"]] = {"det": bool(rg), "types": sorted({t for _, t in rg if t}), "area": sum(a for a, _ in rg)}
    model.train()
    return res


def score(items, res):
    """any-damage picture metrics per source + per type picture / place metrics."""
    out = {}
    for name, src in (("robot", ("robot",)), ("user", ("user",)), ("phone", ("phone",)),
                      ("all", ("robot", "user", "phone"))):
        r = [(e["kind"] == "pos", res[e["id"]]["det"]) for e in items if e["src"] in src and e["kind"] != "ign"]
        tp = sum(g and d for g, d in r)
        fn = sum(g and not d for g, d in r)
        fp = sum(d and not g for g, d in r)
        tn = sum(not g and not d for g, d in r)
        rec, prec = tp / max(1, tp + fn), tp / max(1, tp + fp)
        out[name] = {"recall": round(rec, 3), "false_alarm": round(fp / max(1, fp + tn), 3), "precision": round(prec, 3),
                     "f1": round(2 * rec * prec / max(1e-9, rec + prec), 3), "tp": tp, "fn": fn, "fp": fp, "tn": tn}
    f1s = []
    for t in TYPES:
        g = [e for e in items if t in e["types"]]
        p = [e for e in items if t in res[e["id"]]["types"]]
        tp = sum(t in res[e["id"]]["types"] for e in g)
        rec, prec = tp / max(1, len(g)), tp / max(1, len(p))
        f1 = 2 * rec * prec / max(1e-9, rec + prec)
        spots = collections.defaultdict(bool)
        for e in g:
            spots[e["spot"]] |= t in res[e["id"]]["types"]
        out[t] = {"recall": round(rec, 3), "precision": round(prec, 3), "f1": round(f1, 3), "n": len(g), "pred": len(p),
                  "place_recall": round(sum(spots.values()) / max(1, len(spots)), 3), "places": len(spots)}
        f1s.append(f1)
    det = [e for e in items if e["kind"] == "pos" and res[e["id"]]["det"]]
    out["type_hit"] = round(sum(bool(set(e["types"]) & set(res[e["id"]]["types"])) for e in det) / max(1, len(det)), 3)
    out["type_f1_macro"] = round(float(np.mean(f1s)), 3)
    return out


def widen(sd6):
    """6-class state dict -> 8 classes: classifier channels 6, 7 = copies of 2 (+ tiny noise)."""
    sd = dict(sd6)
    w, b = sd["decode_head.classifier.weight"], sd["decode_head.classifier.bias"]
    extra_w = w[2:3].repeat(2, 1, 1, 1) + 1e-3 * torch.randn_like(w[2:3].repeat(2, 1, 1, 1))
    sd["decode_head.classifier.weight"] = torch.cat([w, extra_w])
    sd["decode_head.classifier.bias"] = torch.cat([b, b[2:3].repeat(2)])
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="nvidia/segformer-b1-finetuned-ade-512-512")
    ap.add_argument("--init", default=f"{T.WORK}/runs/b1/best.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="fold0", help="fold0..fold4 or old")
    ap.add_argument("--index", default="ft/index_types_v5.json")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--head_mult", type=float, default=10.0)
    ap.add_argument("--hub_share", type=float, default=0.5)
    ap.add_argument("--img_w", type=float, default=0.5)
    ap.add_argument("--type_w", type=float, default=4.0, help="CE weight of the three damage classes")
    ap.add_argument("--px_balance", type=float, default=0.0,
                    help="type CE weights x (mean / sampled pixel share of the type) ** this (0 = off; mean stays type_w)")
    ap.add_argument("--phone_w", type=float, default=1.0, help="sampling weight factor of the phone pictures")
    ap.add_argument("--freeze_enc", type=int, default=0, help="freeze the encoder for the first N steps")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--hub_val", type=int, default=300, help="AI Hub val pictures with damage (forgetting check)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_only", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(f"{args.out}/args.json", "w"), indent=1)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    dev = "cuda"
    items = json.load(open(f"{DS}/{args.index}", encoding="utf-8"))
    if args.split == "old":
        tr = [e for e in items if e["split_old"] == "train"]
        va = [e for e in items if e["split_old"] == "val"]
    else:
        k = int(args.split[4:])
        tr = [e for e in items if e["fold"] != k]
        va = [e for e in items if e["fold"] == k]

    # sampling weights: AI Hub | damage (per place x type) | no damage
    idx = json.load(open(f"{T.WORK}/damaged_index.json"))
    hub_names = sorted(idx["train"])
    hub_w = np.array([4.0 if idx["train"][n] > 0 else 1.0 for n in hub_names])
    per = collections.Counter((e["spot"], t) for e in tr if e["kind"] == "pos" for t in e["types"])
    pw = np.array([sum(1.0 / per[(e["spot"], t)] for t in e["types"]) if e["kind"] == "pos" else 0.0 for e in tr])
    nw = np.array([1.0 if e["kind"] == "neg" else 0.0 for e in tr])
    fw = np.array([args.phone_w if e["src"] == "phone" else 1.0 for e in tr])
    pw, nw = pw * fw, nw * fw
    # sampled pixel share of each type (type pixels / picture pixels, weighted like the damage draws)
    tshare = {t: 0.0 for t in TYPES}
    for i, e in enumerate(tr):
        if e["kind"] == "pos" and e.get("px"):
            hw = np.prod(cv2.imread(f"{DS}/{e['mask']}", cv2.IMREAD_UNCHANGED).shape[:2])
            for t in TYPES:
                tshare[t] += pw[i] / pw.sum() * e["px"].get(t, 0) / hw
    ph = np.array([e["src"] == "phone" for e in tr])
    print("sampled type pixel share", {t: round(v, 5) for t, v in tshare.items()},
          "phone share of damage draws %.2f" % (pw[ph].sum() / pw.sum()), flush=True)
    rob_share = (1 - args.hub_share) / 2
    w = np.concatenate([args.hub_share * hub_w / hub_w.sum(), rob_share * pw / pw.sum() + rob_share * nw / nw.sum()])
    data = ConcatDataset([Hub("train", hub_names, S), Robot(tr)])
    sampler = WeightedRandomSampler(w.tolist(), num_samples=args.steps * args.batch, replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed))
    train = DataLoader(data, batch_size=args.batch, sampler=sampler, num_workers=args.workers, pin_memory=True,
                       drop_last=True, persistent_workers=True)
    va_names = sorted(n for n in idx["val"] if idx["val"][n] > 0)          # AI Hub val pictures with damage
    random.Random(1).shuffle(va_names)
    hub_val = DataLoader(T.Surface("val", va_names[:args.hub_val], train=False), batch_size=8, num_workers=16)
    print(f"{args.split}: train robot/user {len(tr)} (damage {int((pw > 0).sum())}, places x types {len(per)}), "
          f"val {len(va)} (damage {sum(e['kind'] == 'pos' for e in va)})", flush=True)

    from transformers import SegformerForSemanticSegmentation
    sd = None if args.init == "none" else torch.load(args.init, map_location="cpu")   # none: ADE checkpoint only
    ncls = 8 if sd is None else sd["decode_head.classifier.weight"].shape[0]
    if args.eval_only and ncls == 6:
        names = T.NAMES
    else:
        names, ncls = NAMES, 8
        if sd is not None and sd["decode_head.classifier.weight"].shape[0] == 6:
            sd = widen(sd)
    model = SegformerForSemanticSegmentation.from_pretrained(
        args.model, num_labels=len(names), ignore_mismatched_sizes=True,
        id2label=dict(enumerate(names)), label2id={n: i for i, n in enumerate(names)}).to(dev)
    model.config.save_pretrained(args.out)
    if sd is not None:
        model.load_state_dict(sd)
    log = open(f"{args.out}/log.jsonl", "a")
    preds = open(f"{args.out}/preds.jsonl", "a")

    def run_eval(step, loss=None):
        res = predict(model, va, dev, ncls)
        r = {"step": step, "val": score(va, res), "loss": loss, "minutes": round((time.time() - t0) / 60, 1)}
        if ncls == 8:
            m6 = model                                         # AI Hub forgetting check on merged classes
            m6.eval()
            conf = torch.zeros(6, 6, dtype=torch.int64, device=dev)
            with torch.no_grad():
                for x, y in hub_val:
                    x, y = x.to(dev), y.to(dev)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        lg = m6(pixel_values=x).logits.float()
                    pr = any_damage(F.interpolate(lg, size=y.shape[-2:], mode="bilinear", align_corners=False)).argmax(1)
                    kk = y != 255
                    conf += torch.bincount(y[kk] * 6 + pr[kk], minlength=36).view(6, 6)
            m6.train()
            c = conf.double()
            r["aihub_damaged_iou"] = round((c[2, 2] / (c[2].sum() + c[:, 2].sum() - c[2, 2]).clamp(min=1)).item(), 4)
        print(json.dumps(r, ensure_ascii=False), flush=True)
        log.write(json.dumps(r, ensure_ascii=False) + "\n")
        log.flush()
        preds.write(json.dumps({"step": step, "res": res}, ensure_ascii=False) + "\n")
        preds.flush()
        return r

    t0 = time.time()
    run_eval(0)
    if args.eval_only:
        return
    head = [p for n, p in model.named_parameters() if n.startswith("decode_head")]
    enc = [p for n, p in model.named_parameters() if not n.startswith("decode_head")]
    opt = torch.optim.AdamW([{"params": enc, "lr": args.lr}, {"params": head, "lr": args.lr * args.head_mult}],
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 200) * (1 - s / args.steps) ** 0.9)
    tw = {t: args.type_w for t in TYPES}
    if args.px_balance > 0:
        mean = np.mean(list(tshare.values()))
        r = {t: (mean / max(tshare[t], 1e-6)) ** args.px_balance for t in TYPES}
        m = np.mean(list(r.values()))
        tw = {t: args.type_w * r[t] / m for t in TYPES}
    print("type CE weights", {t: round(float(v), 2) for t, v in tw.items()}, flush=True)
    cw8 = torch.tensor([0.5, 1.0, float(tw["파손"]), 1.5, 2.0, 2.0, float(tw["결손"]), float(tw["이탈·분리"])], device=dev)
    cw6 = torch.tensor([0.5, 1.0, 4.0, 1.5, 2.0, 2.0], device=dev)
    step = 0
    model.train()
    for p in enc:
        p.requires_grad_(args.freeze_enc == 0)
    for x, y, valid, lab, hub in train:
        if step == args.freeze_enc and args.freeze_enc:
            for p in enc:
                p.requires_grad_(True)
        x, y, valid, lab = (t.to(dev, non_blocking=True) for t in (x, y, valid, lab))
        hub = hub.to(dev).bool()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(pixel_values=x).logits
        loss = losses(lg, y, valid, lab, hub, cw8, cw6, args.img_w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        step += 1
        if step % 100 == 0:
            print(f"step {step} loss {loss.item():.4f} {(time.time() - t0) / step:.3f} s/step", flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            run_eval(step, round(loss.item(), 4))
            torch.save(model.state_dict(), f"{args.out}/step{step}.pt")
        if step >= args.steps:
            break
    print("done", flush=True)


if __name__ == "__main__":
    main()
