# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# See the LICENSE file in the repo root for license information.

"""Voxelization kernel correctness test.

对比 kernel_function 输出与 problem.Model (eager scatter_add) 参考，
allclose(rtol=1e-2, atol=1e-2)——原子累加顺序不同导致微小浮点差，
放松容差。空 voxel 两者都为 0。
"""

import sys
import torch

from problem import Model, get_inputs, get_init_inputs
from kernel import kernel_function


def test_kernel():
    device = "cuda"
    dtype = torch.float32
    model = Model(*get_init_inputs()).to(device).to(dtype)
    (x,) = get_inputs()
    x = x.to(device).to(dtype)

    ref = model(x)  # eager reference [Vx, Vy, Vz, PD]
    out = kernel_function(x, *get_init_inputs())  # [Vx, Vy, Vz, PD]

    if out.shape != ref.shape:
        print(f"shape mismatch: out {out.shape} vs ref {ref.shape}")
        sys.exit(1)

    # non-empty voxels: must match (mean pooling)
    # empty voxels: both should be 0 (ref divides by clamped 1 → 0/1 = 0;
    # kernel normalizes by max(cnt,1) so 0-sum voxel stays 0)
    if not torch.allclose(out, ref, rtol=1e-2, atol=1e-2):
        diff = (out - ref).abs()
        nz = ref.abs() > 1e-6
        if nz.any():
            rel = (diff[nz] / ref[nz].abs()).max()
        else:
            rel = diff.max()
        print(f"NUMERICAL MISMATCH: max abs diff {diff.max()}, max rel {rel}")
        print(f"out[nz][:5] {out[nz].flatten()[:5]}")
        print(f"ref[nz][:5] {ref[nz].flatten()[:5]}")
        sys.exit(1)

    print("Voxelization test passed.")
    sys.exit(0)


if __name__ == "__main__":
    test_kernel()
