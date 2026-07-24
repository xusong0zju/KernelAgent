# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# See the LICENSE file in the repo root for license information.

"""初始（保守）BEV lift-splat Triton kernel —— 优化 baseline。

朴素：遍历每个 (b,h,w)，对每个 d 算 lifted=feat·dw[d] 并 atomic-add
到 BEV。优化空间 = fused 不物化外积（已部分融合）+ block 级 reduce
减少 atomic + 更好的并行粒度。

坐标变换与 problem.Model **严格一致**：bev_x = w 索引线性映射，
bev_y = (h + d) 线性映射，clamp 到网格内。两者用同一套坐标，差只在
"eager 物化外积再 scatter" vs "kernel 逐点 fused scatter"。
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _liftsplat(
    feat_ptr, dw_ptr, cx_ptr, cy_ptr, bev_ptr,
    B, C, H, W, D, BY,
    sB, sC, sH, sW,
    BC: tl.constexpr,
):
    # one program per (b, h, w)
    pid = tl.program_id(0)
    b = pid // (H * W)
    hw = pid % (H * W)
    h = hw // W
    w = hw % W

    bx = tl.load(cx_ptr + w)
    # load feat [C] for (b,h,w) via contiguous load of the C-vector at (b,:,h,w)
    c_off = tl.arange(0, BC)  # [BC]
    fbase = b * sB + h * sH + w * sW
    feat = tl.load(feat_ptr + fbase + c_off * sC, mask=c_off < C, other=0.0)

    for d in range(0, D):
        wgt = tl.load(dw_ptr + b * H * W * D + h * W * D + w * D + d)
        by = tl.load(cy_ptr + h * D + d)
        lin = bx * BY + by
        # atomic-add feat * wgt to bev[lin, :]
        tl.atomic_add(bev_ptr + lin * C + c_off, feat * wgt, mask=c_off < C)


def _coord_tables(h, w, d, bev_x, bev_y, device):
    bx = (torch.arange(w, device=device) * bev_x // w).long()
    by = ((torch.arange(h, device=device).reshape(-1, 1) +
           torch.arange(d, device=device).reshape(1, -1)) * bev_y // (h + d - 1)).long().clamp(0, bev_y - 1)
    return bx, by


def kernel_function(feat, dw, c=32, h=24, w=48, d=16, bev_x=48, bev_y=48) -> torch.Tensor:
    B, C, H, W = feat.shape
    bx, by = _coord_tables(H, W, d, bev_x, bev_y, feat.device)
    bev = torch.zeros(bev_x * bev_y, C, device=feat.device, dtype=feat.dtype)
    grid = (B * H * W,)
    _liftsplat[grid](feat, dw, bx, by, bev, B, C, H, W, d, bev_y,
                     feat.stride(0), feat.stride(1), feat.stride(2), feat.stride(3),
                     BC=triton.next_power_of_2(C))
    return bev.reshape(bev_x, bev_y, C).permute(2, 0, 1)
