# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# See the LICENSE file in the repo root for license information.

"""Initial (unoptimised) Triton RMSNorm over the last dim, Qwen-style [N, H].

Intentionally TWO passes over each row: first to compute the sum-of-squares,
then to normalise and store. This is the obvious baseline that an
optimiser should fuse into a single pass to halve memory traffic.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rmsnorm_two_pass(
    x_ptr, y_ptr, N, H, stride_xn, stride_xh, stride_yn, stride_yh, eps,
    BLOCK_H: tl.constexpr,
):
    row = tl.program_id(0)
    x_row = x_ptr + row * stride_xn
    y_row = y_ptr + row * stride_yn
    # Pass 1: sum of squares over H.
    _sum = tl.zeros([BLOCK_H], dtype=tl.float32)
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        x = tl.load(x_row + idx * stride_xh, mask=mask, other=0.0).to(tl.float32)
        _sum += x * x
    sq = tl.sum(_sum, axis=0)
    rms_inv = 1.0 / tl.sqrt(sq / H + eps)
    # Pass 2: normalise and store.
    for off in range(0, H, BLOCK_H):
        idx = off + tl.arange(0, BLOCK_H)
        mask = idx < H
        x = tl.load(x_row + idx * stride_xh, mask=mask, other=0.0).to(tl.float32)
        y = (x * rms_inv).to(y_ptr.dtype.element_ty)
        tl.store(y_row + idx * stride_yh, y, mask=mask)


def kernel_function(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.dim() == 2
    N, H = x.shape
    y = torch.empty_like(x)
    BLOCK_H = triton.next_power_of_2(min(H, 1024))
    grid = (N,)
    _rmsnorm_two_pass[grid](
        x, y, N, H,
        x.stride(0), x.stride(1), y.stride(0), y.stride(1),
        eps, BLOCK_H=BLOCK_H,
    )
    return y
