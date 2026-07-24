# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# See the LICENSE file in the repo root for license information.

import torch
import triton
import triton.language as tl


@triton.jit
def _voxel_accumulate_faster(
    points_ptr, sum_ptr, cnt_ptr,
    N,
    Vx: tl.constexpr, Vy: tl.constexpr, Vz: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Accumlate points directly from a (N,4) tensor.
    Each block processes BLOCK points, computes voxel index,
    and atomically adds coordinates and feature to global buffers.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    # points_ptr points to a (N,4) row-major contiguous tensor of f32
    base = points_ptr + offs * 4
    x = tl.load(base + 0, mask=mask, other=0.0)
    y = tl.load(base + 1, mask=mask, other=0.0)
    z = tl.load(base + 2, mask=mask, other=0.0)
    f = tl.load(base + 3, mask=mask, other=0.0)

    # conver to int32 – the voxel grid size fts in 32 bits
    xi = x.to(tl.int32)
    yi = y.to(tl.int32)
    zi = z.to(tl.int32)

    # clamp to vald range [0, size-1] (use maximum/minimum, not clamp)
    ix = tl.maximum(0, tl.minimum(xi, Vx - 1))
    iy = tl.maximum(0, tl.minimum(yi, Vy - 1))
    iz = tl.maximum(0, tl.minimum(zi, Vz - 1))

    # linear voxel index: ix * (Vy*Vz) + iy * Vz + iz
    # Vy*Vz and Vz are compile-time constants
    lin = ix * (Vy * Vz) + iy * Vz + iz

    # atomic add into the 4-channel sum
    tl.atomic_add(sum_ptr + lin * 4 + 0, x, mask=mask)
    tl.atomic_add(sum_ptr + lin * 4 + 1, y, mask=mask)
    tl.atomic_add(sum_ptr + lin * 4 + 2, z, mask=mask)
    tl.atomic_add(sum_ptr + lin * 4 + 3, f, mask=mask)

    # atomically increment counter for this voxel
    tl.atomic_add(cnt_ptr + lin, 1.0, mask=mask)


@triton.jit
def _voxel_normalize_faster(
    sum_ptr, cnt_ptr, out_ptr,
    NV: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """
    Normalize accumulated sums by the per-voxel count.
    Processes NV voxels, each 4 channels.
    """
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NV

    c = tl.load(cnt_ptr + offs, mask=mask, other=0.0)
    inv = 1.0 / tl.maximum(c, 1.0)  # avoid div by zero, counts are at least 0

    # unrolled loads for the 4 channels of the voxel
    s0 = tl.load(sum_ptr + offs * 4 + 0, mask=mask, other=0.0)
    s1 = tl.load(sum_ptr + offs * 4 + 1, mask=mask, other=0.0)
    s2 = tl.load(sum_ptr + offs * 4 + 2, mask=mask, other=0.0)
    s3 = tl.load(sum_ptr + offs * 4 + 3, mask=mask, other=0.0)

    # store normalized values
    tl.store(out_ptr + offs * 4 + 0, s0 * inv, mask=mask)
    tl.store(out_ptr + offs * 4 + 1, s1 * inv, mask=mask)
    tl.store(out_ptr + offs * 4 + 2, s2 * inv, mask=mask)
    tl.store(out_ptr + offs * 4 + 3, s3 * inv, mask=mask)


def kernel_function(points: torch.Tensor, voxel_size=(256, 256, 8), point_dim=4) -> torch.Tensor:
    Vx, Vy, Vz = voxel_size
    NV = Vx * Vy * Vz

    # points are expected to be (N,4) float32 tensor, already contiguous
    N = points.shape[0]

    # output buffers
    voxel_sum = torch.zeros(NV, 4, device=points.device, dtype=points.dtype)
    voxel_cnt = torch.zeros(NV, device=points.device, dtype=torch.float32)

    BLOCK = 512  # good balance between occupancy and atomic pressure
    grid = (triton.cdiv(N, BLOCK),)

    _voxel_accumulate_faster[grid](
        points, voxel_sum, voxel_cnt,
        N,
        Vx=Vx, Vy=Vy, Vz=Vz,
        BLOCK=BLOCK,
    )

    BLOCK_NORM = 1024
    grid2 = (triton.cdiv(NV, BLOCK_NORM),)
    out = torch.empty(NV, 4, device=points.device, dtype=points.dtype)
    _voxel_normalize_faster[grid2](
        voxel_sum, voxel_cnt, out,
        NV=NV,
        BLOCK=BLOCK_NORM,
    )

    return out.reshape(Vx, Vy, Vz, 4)