# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# See the LICENSE file in the repo root for license information.

"""BEV lift-splat correctness test.

对比 kernel_function 输出与 problem.Model (eager 物化外积+scatter_add) 参考。
allclose(rtol=1e-2, atol=1e-2)——atomic scatter 顺序不同导致浮点差，放松容差。
"""

import sys
import torch

from problem import Model, get_inputs, get_init_inputs
from kernel import kernel_function


def test_kernel():
    device = "cuda"
    dtype = torch.float32
    model = Model(*get_init_inputs()).to(device).to(dtype)
    feat, dw = get_inputs()
    feat = feat.to(device).to(dtype)
    dw = dw.to(device).to(dtype)

    ref = model(feat, dw)  # [C, bev_x, bev_y]
    out = kernel_function(feat, dw, *get_init_inputs())  # [C, bev_x, bev_y]

    if out.shape != ref.shape:
        print(f"shape mismatch: out {out.shape} vs ref {ref.shape}")
        sys.exit(1)

    if not torch.allclose(out, ref, rtol=1e-2, atol=1e-2):
        diff = (out - ref).abs()
        nz = ref.abs() > 1e-6
        rel = (diff[nz] / ref[nz].abs()).max() if nz.any() else diff.max()
        print(f"NUMERICAL MISMATCH: max abs diff {diff.max()}, max rel {rel}")
        print(f"out[nz][:5] {out[nz].flatten()[:5]}")
        print(f"ref[nz][:5] {ref[nz].flatten()[:5]}")
        sys.exit(1)

    print("BEV lift-splat test passed.")
    sys.exit(0)


if __name__ == "__main__":
    test_kernel()
