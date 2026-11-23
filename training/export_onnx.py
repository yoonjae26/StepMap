#!/usr/bin/env python3
"""Export a trained damage model to ONNX for the Jetson (TensorRT): input (1, 3, 480, 640) float32 normalised RGB,
output (1, 6, 120, 160) class probabilities (softmax inside, so damage_ai.TrtModel needs no extra step).

  python export_onnx.py runs/b2/best.pt --out runs/b2/damage_seg.onnx
On the Jetson:  /usr/src/tensorrt/bin/trtexec --onnx=damage_seg.onnx --fp16 --saveEngine=damage_seg.engine
"""
import argparse

import numpy as np
import torch

import eval_seg


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
    ap.add_argument("--base", default="nvidia/segformer-b2-finetuned-ade-512-512")
    ap.add_argument("--hw", default="480,640")
    ap.add_argument("--opset", type=int, default=16,
                    help="16 for the Jetson (TensorRT 8.5 has no LayerNormalization op; opset 16 exports it as basic ops)")
    args = ap.parse_args()
    H, W = (int(v) for v in args.hw.split(","))
    m = Probs(eval_seg.load(args.ckpt, "cpu", args.base)).eval()
    x = torch.randn(1, 3, H, W)
    torch.onnx.export(m, x, args.out, input_names=["image"], output_names=["probs"], opset_version=args.opset,
                      dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(args.out))
    with torch.no_grad():
        ref = m(x).numpy()
    try:
        import onnxruntime as ort
        got = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
        print("onnxruntime max diff", float(np.abs(got - ref).max()))
    except ImportError:
        pass
    print("exported", args.out, "output", ref.shape)


if __name__ == "__main__":
    main()
