#!/usr/bin/env python3
"""Export the 8-class damage-type model (finetune_types.py) to ONNX for the Jetson, same layout as export_onnx.py:
input (1, 3, 480, 640) normalised RGB, output (1, 8, 120, 160) softmax probabilities, opset 16.
  python export_types_onnx.py runs/types_v5/c1_oldA/step1000.pt --out runs/types_v5/c1_oldA/damage_types.onnx
"""
import argparse

import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation

from finetune_types import NAMES


class Probs(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(pixel_values=x).logits.softmax(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    m = SegformerForSemanticSegmentation.from_pretrained(
        "nvidia/segformer-b1-finetuned-ade-512-512", num_labels=len(NAMES), ignore_mismatched_sizes=True)
    m.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    m = Probs(m.eval()).eval()
    x = torch.randn(1, 3, 480, 640)
    torch.onnx.export(m, x, args.out, input_names=["image"], output_names=["probs"], opset_version=16, dynamo=False)
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(args.out))
    with torch.no_grad():
        ref = m(x).numpy()
    got = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
    print("shape", got.shape, "max abs diff vs torch", float(np.abs(got - ref).max()))


if __name__ == "__main__":
    main()
