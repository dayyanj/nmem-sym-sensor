#!/usr/bin/env python3
"""Export PatchJEPA v2 encoder to ONNX for CPU inference.

Exports only the encode() path (CLS token output, 256-dim).
The DualStreamEncoder handles the 256->192 projection at runtime.

Usage:
    python scripts/export_jepa_v2_onnx.py \
        --checkpoint /path/to/jepa_v2_best.pt \
        --output models/jepa_v2.onnx \
        --img-size 128
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from nmem_sym_sensor.patch_jepa_v2 import PatchJEPAv2


class JEPAEncoder(nn.Module):
    """Wrapper that exposes only the encode() path for ONNX export."""

    def __init__(self, model: PatchJEPAv2):
        super().__init__()
        self.patch_embed = model.patch_embed
        self.pos_embed = model.pos_embed
        self.cls_token = model.cls_token
        self.encoder = model.encoder
        self.encoder_norm = model.encoder_norm
        self.output_proj = model.output_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        patches = self.patch_embed(x) + self.pos_embed
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, patches], dim=1)
        for block in self.encoder:
            tokens = block(tokens)
        tokens = self.encoder_norm(tokens)
        cls_out = tokens[:, 0]
        return self.output_proj(cls_out)


def main():
    parser = argparse.ArgumentParser(description="Export JEPA v2 to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--output", default="models/jepa_v2.onnx", help="Output ONNX path")
    parser.add_argument("--img-size", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--output-dim", type=int, default=256)
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.checkpoint}")
    model = PatchJEPAv2(
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        output_dim=args.output_dim,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=False)
    model.eval()

    encoder = JEPAEncoder(model)
    encoder.eval()

    dummy = torch.randn(1, 3, args.img_size, args.img_size)
    with torch.no_grad():
        test_out = encoder(dummy)
    print(f"Test output shape: {test_out.shape}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    print(f"Exporting to: {args.output}")
    torch.onnx.export(
        encoder,
        dummy,
        args.output,
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={
            "image": {0: "batch"},
            "embedding": {0: "batch"},
        },
        opset_version=17,
    )

    # Verify
    import onnxruntime as ort
    session = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    onnx_out = session.run(None, {"image": dummy.numpy()})[0]
    diff = np.abs(test_out.numpy() - onnx_out).max()
    print(f"Max difference (PyTorch vs ONNX): {diff:.6f}")
    assert diff < 1e-4, f"ONNX output diverges: max diff {diff}"
    print("Export verified successfully.")


if __name__ == "__main__":
    main()
