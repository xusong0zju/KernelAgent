# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# See the LICENSE file in the repo root for license information.

"""初始（保守）Triton voxelization kernel —— 作为优化 baseline。

朴素实现：每个 point 一个 program，直接 atomic add 到全局 voxel_sum / voxel_cnt。
所有点竞争同一原子操作 → 高原子竞争，慢。优化空间：
1. block 级先做局部 reduce（同 voxel 的点在 block 内先归约），再少量 atomic。
2. 预排序点（按 voxel）让同 voxel 点连续 → 消除原子竞争。
3. 融合 count 归一化进同一 kernel（省一次 HBM 往返）。

baseline 故意保守，便于 KernelAgent 闭环测出优化收益。
1D load per-point-feature 避免 2D tensor 切片（Triton 不支持列切片）。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _voxel_accumulate(
    px_ptr, py_ptr, pz_ptr, pf_ptr, sum_ptr, cnt_ptr,
    N, Vx, Vy, Vz,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(px_ptr + offs, mask=mask, other=0.0)
    y = tl.load(py_ptr + offs, mask=mask, other=0.0)
    z = tl.load(pz_ptr + offs, mask=mask, other=0.0)
    f = tl.load(pf_ptr + offs, mask=mask, other=0.0)
    xi = x.to(tl.int64)
    yi = y.to(tl.int64)
    zi = z.to(tl.int64)
    ix = tl.maximum(0, tl.minimum(xi, Vx - 1))
    iy = tl.maximum(0, tl.minimum(yi, Vy - 1))
    iz = tl.maximum(0, tl.minimum(zi, Vz - 1))
    lin = ix * Vy * Vz + iy * Vz + iz  # [BLOCK]
    tl.atomic_add(sum_ptr + lin * 4 + 0, x, mask=mask)
    tl.atomic_add(sum_ptr + lin * 4 + 1, y, mask=mask)
    tl.atomic_add(sum_ptr + lin * 4 + 2, z, mask=mask)
    tl.atomic_add(sum_ptr + lin * 4 + 3, f, mask=mask)
    tl.atomic_add(cnt_ptr + lin, 1.0, mask=mask)


@triton.jit
def _voxel_normalize(sum_ptr, cnt_ptr, out_ptr, NV,
                     BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NV
    c = tl.load(cnt_ptr + offs, mask=mask, other=0.0)
    inv = 1.0 / tl.maximum(c, 1.0)
    for d in tl.static_range(4):
        s = tl.load(sum_ptr + offs * 4 + d, mask=mask, other=0.0)
        tl.store(out_ptr + offs * 4 + d, s * inv, mask=mask)


def kernel_function(points: torch.Tensor, voxel_size=(256, 256, 8), point_dim=4) -> torch.Tensor:
    Vx, Vy, Vz = voxel_size
    NV = Vx * Vy * Vz
    # per-dim views (points is [N, 4]: x,y,z,feat)
    px = points[:, 0].contiguous()
    py = points[:, 1].contiguous()
    pz = points[:, 2].contiguous()
    pf = points[:, 3].contiguous()
    voxel_sum = torch.zeros(NV, 4, device=points.device, dtype=points.dtype)
    voxel_cnt = torch.zeros(NV, device=points.device, dtype=torch.float32)
    N = points.shape[0]
    BLOCK = 256
    grid = (triton.cdiv(N, BLOCK),)
    _voxel_accumulate[grid](px, py, pz, pf, voxel_sum, voxel_cnt, N, Vx, Vy, Vz, BLOCK=BLOCK)
    grid2 = (triton.cdiv(NV, 1024),)
    out = torch.empty(NV, 4, device=points.device, dtype=points.dtype)
    _voxel_normalize[grid2](voxel_sum, voxel_cnt, out, NV, BLOCK=1024)
    return out.reshape(Vx, Vy, Vz, 4)
