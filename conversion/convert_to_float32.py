"""Convert HeavyBuilder2's float64 state dicts to float32, losslessly.

Usage: python convert_to_float32.py <input dir> <output dir>

The published weights were saved as float64, but every value is exactly
representable in float32, and the model casts them to float32 on loading.
Refuse to write a file unless the conversion is exact.
"""

import sys
from pathlib import Path

import torch

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)
for name in ("heavy_chain_model_1", "heavy_chain_model_2", "heavy_chain_model_3"):
    state = torch.load(src / name, map_location="cpu", weights_only=True)
    converted = {}
    for key, tensor in state.items():
        if tensor.dtype == torch.float64:
            single = tensor.float()
            if not torch.equal(single.double(), tensor):
                sys.exit(f"{name}: {key} is not exactly representable in float32")
            converted[key] = single
        else:
            converted[key] = tensor
    torch.save(converted, dst / name)
    reloaded = torch.load(dst / name, map_location="cpu", weights_only=True)
    assert all(torch.equal(reloaded[k].double(), state[k].double()) for k in state)
    dtypes = sorted({str(t.dtype) for t in state.values()})
    params = sum(t.numel() for t in state.values())
    print(f"{name}: {params:,} values, {dtypes} -> float32, exact")
