# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0 (the "License");
# See the LICENSE file in the repo root for license information.

"""CUDA voxelization 朴素版 —— 公平对比编译器质量（无 block 归约）。

与 Triton naive 同结构：每点 5 个 atomicAdd（4 sum + 1 cnt），无段归约、
无 shared mem。用于隔离"CUDA v1 的快是归约带来还是 nvcc 编译器质量带来"。
"""

import torch

_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <int BLOCK>
__global__ void voxel_naive_kernel(
    const float4* __restrict__ points,
    float* __restrict__ sum,
    float* __restrict__ cnt,
    int N, int Vx, int Vy, int Vz)
{
    int tid = blockIdx.x * BLOCK + threadIdx.x;
    if (tid >= N) return;
    float4 p = points[tid];
    int ix = min(max((int)p.x, 0), Vx - 1);
    int iy = min(max((int)p.y, 0), Vy - 1);
    int iz = min(max((int)p.z, 0), Vz - 1);
    int lin = ix * (Vy * Vz) + iy * Vz + iz;
    atomicAdd(sum + lin * 4 + 0, p.x);
    atomicAdd(sum + lin * 4 + 1, p.y);
    atomicAdd(sum + lin * 4 + 2, p.z);
    atomicAdd(sum + lin * 4 + 3, p.w);
    atomicAdd(cnt + lin, 1.0f);
}

__global__ void voxel_normalize_kernel(
    const float* __restrict__ sum, const float* __restrict__ cnt,
    float* __restrict__ out, int NV)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= NV) return;
    float c = cnt[i];
    float inv = 1.0f / fmaxf(c, 1.0f);
    int b = i * 4;
    out[b + 0] = sum[b + 0] * inv;
    out[b + 1] = sum[b + 1] * inv;
    out[b + 2] = sum[b + 2] * inv;
    out[b + 3] = sum[b + 3] * inv;
}

torch::Tensor voxel_naive_cuda(torch::Tensor points, int Vx, int Vy, int Vz) {
    int N = points.size(0);
    long NV = (long)Vx * Vy * Vz;
    auto opts = points.options();
    auto sum = torch::zeros({NV, 4}, opts);
    auto cnt = torch::zeros({NV}, opts);
    auto out = torch::empty({NV, 4}, opts);
    {
        const int BLOCK = 256;
        int grid = (N + BLOCK - 1) / BLOCK;
        voxel_naive_kernel<BLOCK><<<grid, BLOCK>>>(
            (const float4*)points.data_ptr<float>(),
            sum.data_ptr<float>(), cnt.data_ptr<float>(), N, Vx, Vy, Vz);
    }
    {
        const int BLOCK = 256;
        long grid = (NV + BLOCK - 1) / BLOCK;
        voxel_normalize_kernel<<<grid, BLOCK>>>(
            sum.data_ptr<float>(), cnt.data_ptr<float>(), out.data_ptr<float>(), NV);
    }
    return out.reshape({Vx, Vy, Vz, 4});
}
"""

_CPP_SRC = r"""
torch::Tensor voxel_naive_cuda(torch::Tensor points, int Vx, int Vy, int Vz);
"""

_module = None
def _mod():
    global _module
    if _module is None:
        import os
        env_bin = "/root/miniconda3/envs/ka_gpu/bin"
        if os.path.isdir(env_bin):
            os.environ["PATH"] = env_bin + os.pathsep + os.environ.get("PATH", "")
        from torch.utils.cpp_extension import load_inline
        _module = load_inline(
            name="voxel_cuda_naive",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=["voxel_naive_cuda"],
            verbose=False,
        )
    return _module


def kernel_function(points, voxel_size=(256, 256, 8), point_dim=4):
    Vx, Vy, Vz = voxel_size
    pts = points.contiguous().float().cuda()
    if pts.dtype != torch.float32:
        pts = pts.float()
    return _mod().voxel_naive_cuda(pts, Vx, Vy, Vz)
