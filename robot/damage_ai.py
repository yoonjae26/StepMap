"""Sidewalk damage AI (SegFormer trained on AI Hub 서피스마스킹, server seg/train_seg.py) + fusion with the depth.

Runtime on the Jetson: TensorRT engine (GPU, FP16) built once from the ONNX export:
  /usr/src/tensorrt/bin/trtexec --onnx=damage_seg.onnx --fp16 --saveEngine=damage_seg.engine
Python has tensorrt but no pycuda: CUDA memory through ctypes (libcudart). Elsewhere (server test) the ONNX model can
be run with onnxruntime or the PyTorch checkpoint (TorchModel).

Model output: class probabilities (6 classes, 1/4 of the input size): 0 background, 1 paving ok, 2 damaged,
3 braille blocks, 4 fixture (manhole, grating, tree zone), 5 caution (stairs, repair zone).

Fusion (fuse_damage): each damaged region of the picture is looked up in the depth heights (height above the local
road surface of the depth pixels that fall into the region):
  |height| (median of the 10 % most extreme pixels) >= min_height_mm  -> "파손" (broken / missing block, height in cm)
  flat                                                                -> "균열" (crack, score = mean AI probability)
Geometry-only findings (함몰 / 돌출 / 구멍) stay as they are in surface_seg.segment.

Damage type (2026-11-06, optional second engine damage_types.engine, server seg/finetune_types.py, 8 classes:
NAMES8): model A above still decides WHERE the damage is; each A region gets a type ("type_ko"):
  이탈·분리  the depth grid has a block-edge step (height jump >= step_mm between depth pixels step_px apart) within
            the region grown by ~15 colour px (>= ovl_px step depth pixels)
  파손 / 결손  otherwise the larger summed probability of the type model's 파손 (2) / 결손 (6) channel over the region
"type_share" = 파손 / (파손 + 결손), "step_px" = step pixels found. Checked on the places model A never trained on
(server hybrid_types.py): 파손 R 0.80 P 0.74, 결손 R 0.46 P 0.86, 이탈·분리 R 0.21 P 0.40 (7-8 places per type).
"kind_ko" (파손 = height >= min_height_mm / 균열 = flat) is unchanged, so StepMap grading stays as it was.
"""
import ctypes
import os

import cv2
import numpy as np

NAMES = ["background", "surface_ok", "damaged", "braille_ok", "fixture", "caution"]
NAMES8 = ["background", "surface_ok", "파손", "braille_ok", "fixture", "caution", "결손", "이탈·분리"]
DAMAGED = 2
MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255


def preprocess(rgb, size_hw):
    x = cv2.resize(rgb, (size_hw[1], size_hw[0]), interpolation=cv2.INTER_AREA) if rgb.shape[:2] != tuple(size_hw) else rgb
    return np.ascontiguousarray(((x.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None])


class TrtModel:
    """TensorRT engine with one input (1, 3, H, W) float32 and one output (1, C, H/4, W/4) float32 probabilities
    (C = 6 for damage_seg.engine, 8 for damage_types.engine)."""

    def __init__(self, engine_path):
        import tensorrt as trt
        self.cudart = ctypes.CDLL("libcudart.so")
        logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
        self.ctx = self.engine.create_execution_context()
        self.shapes, self.ptrs, self.host = [], [], []
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            n = int(np.prod(shape)) * 4
            p = ctypes.c_void_p()
            self._check(self.cudart.cudaMalloc(ctypes.byref(p), ctypes.c_size_t(n)))
            self.shapes.append(shape)
            self.ptrs.append(p)
            self.host.append(np.empty(shape, np.float32))
        self.in_hw = self.shapes[0][2:]

    def _check(self, err):
        if err != 0:
            raise RuntimeError(f"CUDA error {err}")

    def __call__(self, rgb):
        x = preprocess(rgb, self.in_hw)
        H2D, D2H = 1, 2
        self._check(self.cudart.cudaMemcpy(self.ptrs[0], x.ctypes.data_as(ctypes.c_void_p), ctypes.c_size_t(x.nbytes), H2D))
        self.ctx.execute_v2([p.value for p in self.ptrs])
        out = self.host[1]
        self._check(self.cudart.cudaMemcpy(out.ctypes.data_as(ctypes.c_void_p), self.ptrs[1], ctypes.c_size_t(out.nbytes), D2H))
        return out[0]


class TorchModel:
    """Same output from the PyTorch checkpoint (server / laptop tests)."""

    def __init__(self, ckpt, base="nvidia/segformer-b2-finetuned-ade-512-512", in_hw=(480, 640), device=None):
        import torch
        from transformers import SegformerConfig, SegformerForSemanticSegmentation
        d = os.path.dirname(ckpt)
        cfg = (SegformerConfig.from_pretrained(d) if os.path.isfile(os.path.join(d, "config.json"))
               else SegformerConfig.from_pretrained(base, num_labels=len(NAMES)))
        self.m = SegformerForSemanticSegmentation(cfg)
        self.m.load_state_dict(torch.load(ckpt, map_location="cpu"))
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.m.to(self.dev).eval()
        self.in_hw, self.torch = in_hw, torch

    def __call__(self, rgb):
        with self.torch.no_grad():
            x = self.torch.from_numpy(preprocess(rgb, self.in_hw)).to(self.dev)
            return self.m(pixel_values=x).logits.softmax(1)[0].float().cpu().numpy()


def damage_mask(prob, shape_hw, thr=0.5, min_area_px=400):
    """Probabilities (6, h, w) -> damaged mask at the picture size (H, W) bool, damaged probability map (H, W)."""
    H, W = shape_hw
    p = cv2.resize(prob[DAMAGED], (W, H), interpolation=cv2.INTER_LINEAR)
    am = cv2.resize(prob.argmax(0).astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
    m = ((am == DAMAGED) & (p >= thr)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_area_px
    return keep[lab], p


def depth_steps(h, step_mm=6.0, step_px=2, min_px=6):
    """Block-edge steps on the depth grid: |height difference| >= step_mm between pixels step_px apart (4 directions,
    no wrap-around), small specks removed. h: heights [m] (NaN = none). -> bool mask, same shape as h."""
    H, W = h.shape
    ok = np.isfinite(h)
    f = np.where(ok, h, 0.0).astype(np.float32) * 1000.0
    s = np.zeros((H, W), bool)
    d = step_px
    for dy, dx in ((0, d), (d, 0), (d, d), (d, -d)):
        a = (slice(0, H - dy), slice(max(0, -dx), W - max(0, dx)))
        b = (slice(dy, H), slice(max(0, dx), W - max(0, -dx)))
        s[a] |= ok[a] & ok[b] & (np.abs(f[a] - f[b]) >= step_mm)
    n, lab, st, _ = cv2.connectedComponentsWithStats(s.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = st[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[lab]


def _nearest_seed(region, seeds):
    """region (bool) pixels -> label of the nearest seed (seeds: int32, 0 = none, >0 seed ids inside region)."""
    src = (seeds == 0).astype(np.uint8)
    _, near = cv2.distanceTransformWithLabels(src, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
    # DIST_LABEL_PIXEL: every zero pixel (seed pixel) gets its own label; map labels -> seed id
    lut = np.zeros(int(near.max()) + 1, np.int32)
    zy, zx = np.nonzero(src == 0)
    lut[near[zy, zx]] = seeds[zy, zx]
    out = np.zeros_like(seeds)
    out[region] = lut[near[region]]
    return out


def split_regions(mask, ptype=None, erode_px=4, min_area_px=400, type_min_frac=0.2):
    """Damaged mask -> region labels (int32, 0 background), split where two damages only touch (2026-11-13):
    1. thin bridges: the mask is eroded by erode_px; a component with >= 2 eroded cores of >= min_area_px / 4 is
       split, every pixel going to its nearest core (the 1/4-resolution model output blurs gaps of a few cm shut).
       Long thin cracks erode away entirely and stay one region.
    2. damage type: with the type probabilities (p2 파손, p6 결손 at picture size), a region whose 파손 part and 결손
       part are each >= min_area_px and >= type_min_frac of it is split along the type boundary (the big pieces of
       each side are the seeds, small pieces join the nearest seed).
    Every component is processed inside its own bounding box (fast on the Jetson). Returns (labels, n)."""
    m = mask.astype(np.uint8)
    n0, lab0, st0, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = np.zeros(m.shape, np.int32)
    if n0 <= 1:
        return out, 1
    k = 2 * erode_px + 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    t_all = None if ptype is None else (ptype[0] >= ptype[1])
    nxt = 1
    for r in range(1, n0):
        x0, y0, w, hh = (int(v_) for v_ in st0[r, :4])
        sl = (slice(y0, y0 + hh), slice(x0, x0 + w))
        reg = lab0[sl] == r
        regu = reg.astype(np.uint8)
        # 1. thin bridges
        core = cv2.erode(cv2.copyMakeBorder(regu, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0), ker)[1:-1, 1:-1]
        nc, labc, stc, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
        big = [j for j in range(1, nc) if stc[j, cv2.CC_STAT_AREA] >= max(1, min_area_px // 4)]
        if len(big) >= 2:
            seeds = np.zeros(reg.shape, np.int32)
            for i, j in enumerate(big, 1):
                seeds[labc == j] = i
            parts = _nearest_seed(reg, seeds)
            pieces = [parts == i for i in range(1, len(big) + 1)]
        else:
            pieces = [reg]
        # 2. damage type boundary (파손 / 결손)
        if t_all is not None:
            t = t_all[sl]
            split2 = []
            for pc in pieces:
                a = int(pc.sum())
                a2 = int((pc & t).sum())
                if min(a2, a - a2) < max(min_area_px, type_min_frac * a):
                    split2.append(pc)
                    continue
                seeds = np.zeros(reg.shape, np.int32)
                sid = 1
                for side in (pc & t, pc & ~t):
                    nn, pl, ps, _ = cv2.connectedComponentsWithStats(side.astype(np.uint8), connectivity=8)
                    for j in range(1, nn):
                        if ps[j, cv2.CC_STAT_AREA] >= min_area_px:
                            seeds[pl == j] = sid
                            sid += 1
                if sid <= 2:
                    split2.append(pc)
                    continue
                near = _nearest_seed(pc, seeds)
                split2 += [near == i for i in range(1, sid)]
            pieces = split2
        win = out[sl]
        for pc in pieces:
            if pc.any():
                win[pc] = nxt
                nxt += 1
    return out, nxt


def fuse_damage(mask, pdmg, u, v, h, X, Y, min_height_mm=10.0, min_depth_px=15, ptype=None, step_mm=6.0,
                ovl_px=3, grow_px=15, split=True):
    """Damaged regions of the picture + depth -> list of findings.

    mask, pdmg: picture-size damaged mask / probability. u, v: colour pixel of every depth pixel (int arrays, -1 if
    not visible), h: its height above the road surface [m], X, Y: its levelled robot-frame position [m] (same shape
    as u). Returns dicts: kind_ko 파손 / 균열, height_mm (signed, most extreme 10 %), score, area_px, x, y, bbox;
    with ptype (8-class type probabilities, (8, h', w')) also type_ko 파손 / 결손 / 이탈·분리, type_share, step_px."""
    H, W = mask.shape
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H) & np.isfinite(h)
    if ptype is not None:
        p2 = cv2.resize(ptype[2], (W, H), interpolation=cv2.INTER_LINEAR)
        p6 = cv2.resize(ptype[6], (W, H), interpolation=cv2.INTER_LINEAR)
    # regions: touching damages split (thin bridges, 파손 / 결손 boundary); "rid" = label in fuse_damage.labels
    if split:
        lab, n = split_regions(mask, (p2, p6) if ptype is not None else None)
    else:
        n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        lab = lab.astype(np.int32)
    fuse_damage.labels = lab
    st = np.zeros((max(n, 1), 5), np.int64)                    # x0, y0, w, h, area per region (one pass)
    area = np.bincount(lab.ravel(), minlength=max(n, 1))
    from scipy import ndimage
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        if sl is not None and k < n:
            st[k] = (sl[1].start, sl[0].start, sl[1].stop - sl[1].start, sl[0].stop - sl[0].start, area[k])
    reg = np.zeros(u.shape, np.int32)
    reg[ok] = lab[v[ok], u[ok]]
    if ptype is not None:
        st_d = depth_steps(h, step_mm) & ok                      # step depth pixels with a colour pixel
        sm = np.zeros((H, W), np.uint8)
        sm[v[st_d], u[st_d]] = 1                                  # step pixels in picture coordinates
        ker = np.ones((2 * grow_px + 1, 2 * grow_px + 1), np.uint8)
    out = []
    for k in range(1, n):
        x0, y0, w, hh, area = [int(v_) for v_ in st[k]]
        if area == 0:
            continue
        sel = reg == k
        score = float(pdmg[lab == k].mean())
        item = {"rid": int(k), "score": round(score, 3), "area_px": int(area), "bbox": [int(x0), int(y0), int(x0 + w - 1), int(y0 + hh - 1)],
                "height_mm": None, "x": None, "y": None, "kind_ko": "균열"}
        if sel.sum() >= min_depth_px:
            hv = h[sel] * 1000.0
            lo, hi = np.percentile(hv, 10), np.percentile(hv, 90)
            ext = float(np.median(hv[hv <= lo])) if abs(lo) > abs(hi) else float(np.median(hv[hv >= hi]))
            item.update(height_mm=round(ext, 1), x=float(np.median(X[sel])), y=float(np.median(Y[sel])))
            if abs(ext) >= min_height_mm:
                item["kind_ko"] = "파손"
        if ptype is not None:
            r = lab == k
            s2, s6 = float(p2[r].sum()), float(p6[r].sum())
            share = s2 / max(s2 + s6, 1e-9)
            x1, y1 = max(0, x0 - grow_px), max(0, y0 - grow_px)        # grow the region only inside its box
            sub = r[y1:y0 + hh + grow_px, x1:x0 + w + grow_px].astype(np.uint8)
            near = cv2.dilate(sub, ker)
            n_st = int((sm[y1:y0 + hh + grow_px, x1:x0 + w + grow_px] & near).sum())
            item.update(type_share=round(share, 3), step_px=n_st,
                        type_ko="이탈·분리" if n_st >= ovl_px else ("파손" if share >= 0.5 else "결손"))
        out.append(item)
    out.sort(key=lambda g: -g["area_px"])
    return out


def unevenness(h, cls, X, floor_classes, x_range=(0.4, 1.2), min_px=800):
    """요철 (flatness) index of the road surface ahead: spread (95th - 5th percentile) of the heights of floor-like
    pixels (normal floor, sunken, raised) between x_range [m] ahead [mm]. None if too few pixels."""
    m = np.isin(cls, floor_classes) & np.isfinite(h) & (X >= x_range[0]) & (X <= x_range[1])
    if m.sum() < min_px:
        return None
    hv = h[m] * 1000.0
    return float(np.percentile(hv, 95) - np.percentile(hv, 5))
