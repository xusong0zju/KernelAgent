---
name: atomic-scatter-triton-vs-cuda-attribution
description: |
  Correctly diagnose WHY a hand-written CUDA kernel beats an equivalent
  Triton kernel on atomic-bound scatter-add workloads (voxelization, BEV
  lift-splat, point-cloud hashing). Use when: (1) you measured CUDA faster
  than Triton and are about to attribute it to "Triton lacks shared-memory
  / block-reduce expressivity", (2) a Triton scatter kernel is 2-3x slower
  than a same-algorithm CUDA one, (3) you want to know if Triton can match
  CUDA by vectorizing. Covers: the naive same-structure A/B test that
  isolates compiler quality from algorithm, the random-vs-sorted input test
  that isolates atomic-locality from reduction, and the common WRONG
  attribution ("Triton expressivity") vs the RIGHT one ("compiler quality
  on high-contention atomics").
author: Claude Code
version: 1.0.0
date: 2026-07-25
---

# Atomic scatter：Triton vs CUDA 的正确归因

## 问题

你为 atomic-bound 的 scatter kernel（voxelization 点→voxel 原子累加、BEV lift-splat、点云 hash）手写了 CUDA kernel（经 torch cpp_extension），它比 Triton 版快 2-3x。你（或你的 agent）可能得出结论："Triton 缺 shared-memory / block-reduce 表达力，所以慢"。**这个归因通常是错的。**

## 触发条件

- scatter-add kernel：每个 thread 对一个共享 buffer 按"计算出的索引"（voxel id / BEV cell）做 N 次 atomicAdd。
- CUDA 版用 `float4` load + `atomicAdd`；Triton 版用 `tl.load` + 带 mask 的 `tl.atomic_add`。
- 随机/无序输入下 CUDA 快 2-2.5x。
- 有人声称"Triton 做不了 block 归约，因 `tl.static_shared`/`tl.shared` 不存在" → 错误跳跃。

## 根因（正确归因）

对**高冲突（随机）**输入的 atomic-bound scatter，CUDA-vs-Triton 差距主要来自**编译器质量**，不是算法或表达力：

- **nvcc** 把 `atomicAdd` lowering 到最优 `red.global` 路径，合并同 cache line 的多个 atomicAdd；`float4` load = 一次 16B 事务。
- **Triton 的 LLVM 后端** 生成的 atomic lowering 次优，且不从带 mask 的 `tl.atomic_add` 合并同 line 原子；向量化 load 要 2D block ptr 形式（多余索引运算）。
- block 级归约（shared mem 段合并）在**点稀疏** shape（平均 <1 点/voxel）只贡献 ~10%——同 voxel 邻居太少没得合并。所以"CUDA 写归约"不是主要收益。

经典错误归因："CUDA 快是因为 block 归约把原子数从 N 降到 unique-voxels/block"。稀疏输入下几乎没有同 voxel 连续段，归约几乎不触发。

## 诊断流程

在断言"Triton 表达力限制"前，跑这三个测试。一个下午，避免错误结论。

### 测试 1：归约到底有没有生效？（random vs sorted 输入）

同一 kernel，两种输入：随机点 vs **按 voxel 索引预排序的点**（voxelization 对点序不敏感，排序是合法变换）。都测时间。

- 若 sorted ≪ random → 归约/局部性在邻居相邻时触发。random 的慢部分是"无局部性"，不是"无归约"。
- 若 sorted ≈ random → 归约本就没生效（稀疏），那把 CUDA 的快归功于归约就是错的。

### 测试 2（决定性）：naive 同结构 A/B（隔离编译器）

写 **CUDA naive** = 去掉 block 归约和 shared mem 的 CUDA kernel，纯每点 5 atomicAdd，结构上和 Triton naive 完全相同。然后三者在 random/sorted 下对比：

```
                random   sorted
triton_naive    0.557    0.264   # Triton，无归约
cuda_naive      0.274    0.104   # CUDA， 无归约，同结构
cuda_v1(归约)   0.246    0.100   # CUDA， 有 block 归约
```

读法：
- `cuda_naive` vs `cuda_v1`：归约贡献 = random ~10%、sorted ~0%。**归约不是主要收益**。
- `triton_naive` vs `cuda_naive`（**同算法、不同编译器**）：CUDA 快 2.0-2.5x → **差距是编译器质量**，不是表达力/归约。

### 测试 3：Triton 向量化能否追平？（把优势移植回 Triton）

Triton kernel 用 **2D block ptr load + 2D `tl.atomic_add`**（一次 [BLOCK,4] load + 一次 [BLOCK,4] atomic 替代 4 次标量）——把 CUDA 的 `float4`+合并原子优势移植回 Triton。取列要用 `tl.where(d==k, pts, 0).sum(axis=1)`（Triton 禁止 `pts[:,i]`）。

```
                random   sorted
triton_naive    0.557    0.264
triton_vec      0.370    0.118   # 2D load+atomic
cuda_naive      0.274    0.104
```

- **sorted** 输入：triton_vec 0.118 ≈ cuda_naive 0.104 → 向量化**追平**（原子有局部性时编译器差距几乎消失）。
- **random** 输入：triton_vec 0.370 仍 > cuda_naive 0.274 → 残余 1.35x 是 nvcc 在高冲突下的 cache-line 原子合并，Triton 后端做不到。

## 决策规则

三个测试后：

1. **输入天然有序**（LiDAR 扫描线、预排序 batch）？→ Triton 向量化 ≈ CUDA。**不必上 CUDA**，Triton 够。
2. **输入随机、高原子冲突** → CUDA 快 1.35-2.5x，差距是编译器（atomic lowering + cache-line 合并），**不是表达力**。若 1.5x 重要就用 CUDA；别怪"Triton 表达力"。
3. **点稀疏 shape**（平均 <1 点/bin）→ block 归约只值 ~10%。别为追归约强上复杂结构，先向量化（便宜 1.5x）。
4. **没做测试 2 同结构 A/B，别下"Triton 表达力限制"结论**。`tl.static_shared`/`tl.shared` 缺失是红鲱鱼——Triton 有 `tl.sort`/`tl.reduce`/`tl.sum(axis=)` 内部就实现了 block 归约。

## 验证

上面三张数字表就是验证。在单卡 GPU（RTX 2080 Ti、Triton 3.7、CUDA 13）跑 `examples/optimize_voxelization/` 的 `triton_naive.py` / `triton_vec.py` / `kernel_cuda_naive.py` / `kernel_cuda.py`（+ problem.py 里 argsort 的 sorted 变体）即可复现。

## 注意

- 混淆点：cuda_naive 用了 `float4` load（不是纯编译器对照，load 向量化混进来了）。要完全隔离，写用 4×标量 `__ldg` 的 cuda_naive，和 float4 版比，隔离 load 成本 vs atomic-lowering 成本。（实践中"编译器+向量化"混合归因才是决策要的。）
- `tl.sort`（3.7）返回单值排序 tensor（无 perm）；按一个 key 排序多数组要 `tl.join`/broadcast 技巧或 sort-then-recompute——所以多列 block 归约在 Triton 里别扭（不是不可能）。但测试 2 表明稀疏输入下归约本不是瓶颈。
- 0.022ms 带宽下限（13.7MB @ 616GB/s）对 atomic-bound scatter **不可达**——原子会序列化。真实下限 ≈ atomic 吞吐 × N，远高于带宽下限。别在 scatter kernel 上追带宽数。

## 参考

- 实测于 RTX 2080 Ti（SM7.5）+ Triton 3.7.1 + CUDA 13.0，2026-07-25。数字：triton_naive 0.557/0.264，cuda_naive 0.274/0.104，triton_vec 0.370/0.118（random/sorted）。
- KernelAgent `docs/具身3D算子优化实录.md` "Voxelization 的 CUDA vs Triton 深度对照"——完整表格 + 错→对归因修正。
- 关联：`reasoning-model-thinking-budget` skill（不同领域，同样教训：别接受第一个看似合理的归因——跑隔离实验）。
