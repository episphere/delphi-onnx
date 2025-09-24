import argparse
import json
import os
import sys
from typing import Any, Dict, Tuple

import torch

from model import Delphi, DelphiConfig


def _resolve_state_dict(ckpt: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if isinstance(ckpt, dict):
        model_args = None
        for k in ["model_args", "config", "hparams"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                model_args = ckpt[k]
                break

        for k in ["model", "model_state_dict", "state_dict"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k], (model_args or {})
                
        looks_like_state_dict = all(isinstance(v, torch.Tensor) for v in ckpt.values())
        if looks_like_state_dict:
            return ckpt, (model_args or {})
    
class DelphiForONNX(torch.nn.Module):
    def __init__(self, model: Delphi):
        super().__init__()
        self.model = model

    def forward(self, idx: torch.Tensor, age: torch.Tensor) -> torch.Tensor:
        logits, _, _ = self.model(idx, age, targets=None, targets_age=None)
        return logits

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint (.pt/.pth)")
    parser.add_argument("--output", default="model.onnx", help="Path to write ONNX file")
    parser.add_argument("--seq-len", type=int, default=128, help="Dummy sequence length for export")
    parser.add_argument("--batch-size", type=int, default=1, help="Dummy batch size for export")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version (e.g., 17 or 18)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device to instantiate model")
    parser.add_argument("--strict-load", action="store_true", help="Use strict=True when loading state_dict")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict, model_args = _resolve_state_dict(ckpt)

    cfg = DelphiConfig()

    for k, v in (model_args or {}).items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, v)
            except Exception:
                pass

    print("Resolved model config:")
    for k in ["block_size", "vocab_size", "n_layer", "n_head", "n_embd", "dropout", "token_dropout", "t_min", "bias", "mask_ties"]:
        if hasattr(cfg, k):
            print(f"  {k}: {getattr(cfg, k)}")

    model = Delphi(cfg).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=args.strict_load)
    print("Loaded state_dict (strict={}):".format(args.strict_load))
    if missing:
        print("  Missing keys:", missing)
    if unexpected:
        print("  Unexpected keys:", unexpected)

    model.eval()

    B = args.batch_size
    T = min(args.seq_len, cfg.block_size if hasattr(cfg, "block_size") else args.seq_len)
    example_idx = torch.zeros(B, T, dtype=torch.long, device=device)
    example_age = torch.zeros(B, T, dtype=torch.float32, device=device)

    wrapper = DelphiForONNX(model).to(device)

    input_names = ["idx", "age"]
    output_names = ["logits", "age"]
    dynamic_axes = {
        "idx": {0: "batch", 1: "seq"},
        "age": {0: "batch", 1: "seq"},
        "logits": {0: "batch", 1: "seq"},
    }


    if T > 0:
        example_idx[:, 0] = 1

    print(f"Exporting to ONNX: {args.output} (opset={args.opset})")
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (example_idx, example_age),
            args.output,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=args.opset
        )

    print("ONNX export complete.")
    print(f"Saved: {args.output}")
    
    import onnxruntime as ort
    import numpy as np

    B, T = 1, 128
    idx = torch.zeros(B, T, dtype=torch.long); idx[:,0] = 1
    age = torch.zeros(B, T, dtype=torch.float32)

    wrapper.eval()
    with torch.no_grad():
        pt_logits = wrapper(idx, age).cpu().numpy()

    sess = ort.InferenceSession("delphi.onnx", providers=["CPUExecutionProvider"])
    onnx_logits = sess.run(["logits"], {"idx": idx.numpy(), "age": age.numpy()})[0]

    print("Max abs diff:", np.max(np.abs(pt_logits - onnx_logits)))

if __name__ == "__main__":
    main()
