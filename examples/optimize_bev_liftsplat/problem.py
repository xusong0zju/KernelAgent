# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# See the LICENSE file in the repo root for license information.

"""BEV Lift-Splat（LSS / BEVFusion 风格）简化版。

自动驾驶 BEV 感知核心：把 2D 图像特征"lift"成 3D frustum（特征 × 离散
深度权重），再"splat"（scatter-add）到 BEV 网格。调研报告 §4.4 标注的
Triton 蓝海（★ 可行性高），优化空间 = fused 不物化巨大外积张量。

为可单卡验证 + 确定性 test，做简化：
- 输入 feat[B, C, H, W] + depth_weight[B, H, W, D]（深度离散权重）
- lift: 对每个 (b,h,w,d)，特征向量 = feat[b,:,h,w] × depth_weight[b,h,w,d]  （外积，朴素会物化 [B,C,H,W,D] 巨张量）
- splat: 按 (h,w,d)→(bev_x,bev_y) 的固定变换（预计算坐标表）scatter-add 到 BEV[b, C, X, Y]

朴素 eager 用 torch.einsum 物化外积再 scatter_add —— 显存峰值 = B·C·H·W·D。
fused kernel 不物化外积，逐 (h,w,d) 累加到 BEV，省掉这个巨张量 —— 这就是优化空间。

shape（2080Ti 友好）：B=2 相机(简化为batch), C=32, H=24, W=48, D=16
外积 = 2·32·24·48·16 ≈ 59MB（朴素物化这个）；BEV 网格 [48,48]×C=32。
放大到 B=6,C=64,H=48,W=120,D=24 时外积 ~530MB（朴素爆）—— 但本 test 用小 shape 便于验证。
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """简化 lift-splat：feat × depth_weight → scatter-add 到 BEV。"""

    def __init__(self, c=32, h=24, w=48, d=16, bev_x=48, bev_y=48):
        super().__init__()
        self.c, self.h, self.w, self.d = c, h, w, d
        self.bev_x, self.bev_y = bev_x, bev_y
        # 固定坐标变换表：(h, w, d) -> (bev_x, bev_y)，clamp 到网格内
        # 简单线性映射：bev_x = w 索引映射，bev_y = (h + d) 映射
        bx = (torch.arange(w) * bev_x // w).long()
        by = ((torch.arange(h).reshape(-1, 1) + torch.arange(d).reshape(1, -1)) * bev_y // (h + d - 1)).long()
        by = by.clamp(0, bev_y - 1)
        self.register_buffer("coord_x", bx)  # [w]
        self.register_buffer("coord_y", by)  # [h, d]

    def forward(self, feat: torch.Tensor, depth_weight: torch.Tensor) -> torch.Tensor:
        # feat: [B, C, H, W]; depth_weight: [B, H, W, D]
        B, C, H, W = feat.shape
        _, _, _, D = depth_weight.shape
        # lift: outer product per (b,h,w,d) -> lifted[b,c,h,w,d] = feat[b,c,h,w]*dw[b,h,w,d]
        # 朴素：物化 lifted = einsum('bchw,bhwd->bchw d')... reshape
        lifted = feat.unsqueeze(-1) * depth_weight.unsqueeze(1)  # [B,C,H,W,D]
        # splat: scatter-add to BEV by coord
        bx = self.coord_x  # [W]
        by = self.coord_y  # [H, D]
        bx_exp = bx.view(1, 1, 1, W, 1)  # broadcast
        by_exp = by.view(1, H, 1, 1, D)
        bx_exp = bx_exp.expand(B, H, 1, W, D).reshape(-1)
        by_exp = by_exp.expand(B, H, 1, W, D).reshape(-1)
        lifted_flat = lifted.permute(0, 2, 3, 4, 1).reshape(-1, C)  # [B*H*W*D, C]
        lin = bx_exp * self.bev_y + by_exp  # [B*H*W*D]
        bev = torch.zeros(self.bev_x * self.bev_y, C, device=feat.device, dtype=feat.dtype)
        bev.index_add_(0, lin, lifted_flat)
        return bev.reshape(self.bev_x, self.bev_y, C).permute(2, 0, 1)  # [C, bev_x, bev_y]


C, H, W, D, BEV_X, BEV_Y = 32, 24, 48, 16, 48, 48


def get_inputs():
    feat = torch.randn(2, C, H, W)
    dw = torch.softmax(torch.randn(2, H, W, D), dim=-1)  # 深度权重归一化
    return [feat.cuda(), dw.cuda()]


def get_init_inputs():
    return [C, H, W, D, BEV_X, BEV_Y]
