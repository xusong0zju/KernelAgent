# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# See the LICENSE file in the repo root for license information.

"""Qwen-style RMSNorm problem.

Qwen (and LLaMA-family) models use RMSNorm as the normalisation layer,
applied to activations of shape [batch*seq, hidden_size] reducing along
the hidden dimension (dim=1). This is the real shape the optimiser
should target — NOT the NCHW variant in optimize_02.

hidden_size = 4096 matches Qwen2-7B. We use 4096 rows so the kernel
has enough parallelism to saturate 68 SMs of an RTX 2080 Ti while
staying well within 22 GB (4096*4096 fp32 ≈ 64 MB per tensor).
"""

import torch
import torch.nn as nn


class Model(nn.Module):
    """RMSNorm over the last (hidden) dimension, Qwen-style."""

    def __init__(self, hidden_size: int = 4096, eps: float = 1e-6):
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, H]; reduce mean of squares along H, normalise.
        rms = torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + self.eps)
        return x / rms


N = 4096
H = 4096


def get_inputs():
    return [torch.randn(N, H)]


def get_init_inputs():
    return [H]
