# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# See the LICENSE file in the repo root for license information.

import torch
import triton
import triton.language as tl


@triton.jit
def _voxel_accumulate_blocked(
    px_ptr, py_ptr, pz_ptr, pf_ptr, sum_ptr, cnt_ptr,
    N, Vx, Vy, Vz,
    BLOCK: tl.constexpr,
    BLOCK_REDUCE: tl.constexpr,
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
    lin = ix * Vy * Vz + iy * Vz + iz

    # Block-level reduction: each thread accumulates into registers,
    # then we atomically flush only unique voxels per warp/block.
    # For simplicity, we do a small local reduction: each thread
    # atomically adds its own point; but we can reduce contention
    # by staggering atomic adds across threads in a block.
    # Actually, we can do a block-local gather using shared memory
    # via registers + atomic on global. Since tl.shared is not allowed,
    # we use a simple per-thread atomic with reduced contention
    # by having each block process points that are spatially coherent
    # (points are not sorted here, but we rely on grid stride to spread).

    # Use 4 separate atomic adds to allow more parallelism
    tl.atomic_add(sum_ptr + lin * 4 + 0, x, mask=mask)
    tl.atomic_add(sum_ptr + lin * 4 + 1, y, mask=mask)
    tl.atomic_add(sum_ptr + lin * 4 + 2, z, mask=mask)
    tl.atomic_add(sum_ptr + lin * 4 + 3, f, mask=mask)
    tl.atomic_add(cnt_ptr + lin, 1.0, mask=mask)


@triton.jit
def _voxel_accumulate_fused(
    px_ptr, py_ptr, pz_ptr, pf_ptr, out_ptr,
    N, Vx, Vy, Vz,
    BLOCK: tl.constexpr,
):
    """
    Fused accumulate + normalize in one pass.
    Each block processes BLOCK points, accumulates into a local
    buffer in registers, then atomically adds to global memory.
    After all blocks finish, a separate normalize pass is still needed
    because we cannot synchronize across blocks.
    """
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
    lin = ix * Vy * Vz + iy * Vz + iz

    # Direct atomic add to global memory
    tl.atomic_add(out_ptr + lin * 4 + 0, x, mask=mask)
    tl.atomic_add(out_ptr + lin * 4 + 1, y, mask=mask)
    tl.atomic_add(out_ptr + lin * 4 + 2, z, mask=mask)
    tl.atomic_add(out_ptr + lin * 4 + 3, f, mask=mask)
    # We'll use a separate counter array and normalize later


@triton.jit
def _voxel_normalize_fast(sum_ptr, cnt_ptr, out_ptr, NV,
                          BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NV
    c = tl.load(cnt_ptr + offs, mask=mask, other=0.0)
    inv = 1.0 / tl.maximum(c, 1.0)
    # Unrolled loop for 4 channels
    s0 = tl.load(sum_ptr + offs * 4 + 0, mask=mask, other=0.0)
    s1 = tl.load(sum_ptr + offs * 4 + 1, mask=mask, other=0.0)
    s2 = tl.load(sum_ptr + offs * 4 + 2, mask=mask, other=0.0)
    s3 = tl.load(sum_ptr + offs * 4 + 3, mask=mask, other=0.0)
    tl.store(out_ptr + offs * 4 + 0, s0 * inv, mask=mask)
    tl.store(out_ptr + offs * 4 + 1, s1 * inv, mask=mask)
    tl.store(out_ptr + offs * 4 + 2, s2 * inv, mask=mask)
    tl.store(out_ptr + offs * 4 + 3, s3 * inv, mask=mask)


@triton.jit
def _voxel_accumulate_2d(
    px_ptr, py_ptr, pz_ptr, pf_ptr, sum_ptr, cnt_ptr,
    N, Vx, Vy, Vz,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Each program processes a 2D tile of points for better memory coalescing.
    """
    pid = tl.program_id(0)
    # Use 2D grid: pid_x = pid % grid_x, pid_y = pid // grid_x
    # But we keep it simple: 1D grid with larger blocks
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
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
    lin = ix * Vy * Vz + iy * Vz + iz

    tl.atomic_add(sum_ptr + lin * 4 + 0, x, mask=mask)
    tl.atomic_add(sum_ptr + lin * 4 + 1, y, mask=mask)
    tl.atomic_add(sum_ptr + lin * 4 + 2, z, mask=mask)
    tl.atomic_add(sum_ptr + lin * 4 + 3, f, mask=mask)
    tl.atomic_add(cnt_ptr + lin, 1.0, mask=mask)


def kernel_function(points: torch.Tensor, voxel_size=(256, 256, 8), point_dim=4) -> torch.Tensor:
    Vx, Vy, Vz = voxel_size
    NV = Vx * Vy * Vz

    px = points[:, 0].contiguous()
    py = points[:, 1].contiguous()
    pz = points[:, 2].contiguous()
    pf = points[:, 3].contiguous()

    voxel_sum = torch.zeros(NV, 4, device=points.device, dtype=points.dtype)
    voxel_cnt = torch.zeros(NV, device=points.device, dtype=torch.float32)

    N = points.shape[0]
    # Use larger block size for better occupancy and fewer atomic conflicts
    BLOCK = 512
    grid = (triton.cdiv(N, BLOCK),)
    _voxel_accumulate_2d[grid](px, py, pz, pf, voxel_sum, voxel_cnt, N, Vx, Vy, Vz, BLOCK_SIZE=BLOCK)

    # Normalize with larger block size
    BLOCK_NORM = 1024
    grid2 = (triton.cdiv(NV, BLOCK_NORM),)
    out = torch.empty(NV, 4, device=points.device, dtype=points.dtype)
    _voxel_normalize_fast[grid2](voxel_sum, voxel_cnt, out, NV, BLOCK=BLOCK_NORM)

    return out.reshape(Vx, Vy, Vz, 4)