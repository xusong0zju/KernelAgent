# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# See the LICENSE file in the repo root for license information.

"""点云 Voxelization（PointPillars / VoxelNet 风格预处理）。

具身/3D 感知的第一步：把 N 个无序 3D 点（x, y, z, feature）按空间网格
分桶，每桶取均值池化成 voxel feature。LiDAR 感知（自动驾驶/机器人）
的标准预处理算子，是调研报告 §4.3 标注的 Triton 蓝海（★ 可行性高）。

优化空间（报告 §4.3）：fused hash + pool，消除中间 hash 表物化，
单 kernel 完成"算 voxel 索引 + 累加 + 计数 + 归一化"，省掉朴素实现的
多次 HBM 往返 + 中间索引张量。

shape/显存（2080Ti 友好）：N=200K 点 × 4 维 fp32 = 3.2MB；
voxel 网格 [256, 256, 8] × 4 = 8MB。零显存压力。
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """点云 voxelization：把点特征按 voxel 网格做均值池化。

    输入 points: [N, 4]  (x, y, z, feat)，坐标已归一化到 [0, voxel_range)。
    输出 voxel_features: [Vx, Vy, Vz, 4]，每 voxel 内点的均值。
    空 voxel 输出 0。
    """

    def __init__(self, voxel_size=(256, 256, 8), point_dim=4):
        super().__init__()
        self.voxel_size = voxel_size
        self.point_dim = point_dim

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        Vx, Vy, Vz = self.voxel_size
        # 点坐标取整成 voxel 索引（clamp 到网格内）
        idx = points[:, :3].long().clamp(0)
        idx[:, 0] = idx[:, 0].clamp(0, Vx - 1)
        idx[:, 1] = idx[:, 1].clamp(0, Vy - 1)
        idx[:, 2] = idx[:, 2].clamp(0, Vz - 1)
        linear_idx = idx[:, 0] * Vy * Vz + idx[:, 1] * Vz + idx[:, 2]  # [N]

        voxel_sum = torch.zeros(Vx * Vy * Vz, self.point_dim, device=points.device, dtype=points.dtype)
        voxel_cnt = torch.zeros(Vx * Vy * Vz, device=points.device, dtype=torch.float32)
        voxel_sum.index_add_(0, linear_idx, points)
        voxel_cnt.index_add_(0, linear_idx, torch.ones(points.shape[0], device=points.device))
        voxel_feat = voxel_sum / voxel_cnt.clamp(min=1.0).unsqueeze(-1)
        return voxel_feat.reshape(Vx, Vy, Vz, self.point_dim)


N = 200000
VOXEL_SIZE = (256, 256, 8)


def get_inputs():
    # 随机点云，坐标在 voxel 网格范围内，第 4 维是特征。
    points = torch.rand(N, 4) * torch.tensor([VOXEL_SIZE[0], VOXEL_SIZE[1], VOXEL_SIZE[2], 1.0])
    return [points.cuda()]


def get_init_inputs():
    return [VOXEL_SIZE, 4]
