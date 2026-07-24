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

# Atomic-scatter: Triton vs CUDA attribution

## Problem

You hand-wrote a CUDA kernel (via torch cpp_extension) for an
atomic-bound scatter kernel — e.g. voxelization (point → atomic-add into
voxel grid), BEV lift-splat, point-cloud hashing. It runs 2-3x faster
than the Triton version. You (or your agent) are tempted to conclude
"Triton lacks the expressivity for block-level reduction / shared memory,
that's why it's slow." **This attribution is usually wrong.**

## Context / Trigger Conditions

- A scatter-add kernel where each thread does N atomicAdd into a shared
  buffer keyed by a computed index (voxel id, BEV cell).
- CUDA version uses `float4` loads + `atomicAdd`; Triton uses `tl.load` +
  `tl.atomic_add` with mask.
- CUDA measured 2-2.5x faster on random/unsorted input.
- Someone claims "Triton can't do block reduction because
  `tl.static_shared`/`tl.shared` don't exist" → WRONG leap.

## Root Cause (the right attribution)

For atomic-bound scatter on **high-contention (random)** input, the
CUDA-vs-Triton gap is dominated by **compiler quality**, not algorithm or
expressivity:

- **nvcc** lowers `atomicAdd` to optimal `red.global` paths and coalesces
  multiple `atomicAdd` to the same cache line; `float4` load = one 16B
  transaction.
- **Triton's LLVM backend** emits less optimal atomic lowering and does
  NOT coalesce same-line atomics from masked `tl.atomic_add`; vectorized
  loads require the 2D-block-pointer form (extra index arithmetic).
- Block-level reduction (shared-mem segment merge) contributes only ~10%
  on **point-sparse** shapes (avg <1 point/voxel) — too few same-voxel
  neighbors to merge. So "writing reduction in CUDA" is NOT the main win.

The classic WRONG attribution: "CUDA fast because block reduction cuts
atomic count from N to unique-voxels/block." On sparse input there are
almost no runs of same-voxel neighbors, so reduction barely fires.

## Solution / Diagnostic procedure

Before claiming "Triton expressivity limits", run these three tests. They
take one afternoon and prevent the wrong conclusion.

### Test 1: Does the reduction even fire? (random vs sorted input)

Same kernel, two inputs: random points vs points **pre-sorted by voxel
index** (voxelization is permutation-invariant, so sorting is a legal
transform). Measure both.

- If sorted ≪ random → reduction/locality fires when neighbors are
  adjacent. The random-time gap is partly "no locality", not "no
  reduction".
- If sorted ≈ random → reduction never fired anyway (sparse), so
  crediting reduction for CUDA's speed is wrong.

### Test 2 (decisive): naive same-structure A/B (isolate compiler)

Write a **CUDA naive** = the CUDA kernel with the block-reduction and
shared-mem REMOVED — pure per-point 5 atomicAdd, structurally identical
to the Triton naive. Now compare three on random & sorted:

```
                random   sorted
triton_naive    0.557    0.264   # Triton, no reduction
cuda_naive      0.274    0.104   # CUDA,  no reduction, SAME structure
cuda_v1(reduce) 0.246    0.100   # CUDA,  with block reduction
```

Read it as:
- `cuda_naive` vs `cuda_v1`: reduction's contribution = ~10% (random),
  ~0% (sorted). Reduction is NOT the main win.
- `triton_naive` vs `cuda_naive` (SAME algorithm, different compiler):
  CUDA 2.0-2.5x faster → **the gap is compiler quality**, not
  expressivity/reduction.

### Test 3: Can Triton close the gap by vectorizing? (port the win back)

Implement the Triton kernel with **2D block-pointer load + 2D
`tl.atomic_add`** (one [BLOCK,4] load + one [BLOCK,4] atomic instead of
4 scalar each) — porting CUDA's `float4`+coalesced-atomic advantage back
into Triton. Column extraction needs `tl.where(d==k, pts, 0).sum(axis=1)`
since Triton forbids `pts[:,i]`.

```
                random   sorted
triton_naive    0.557    0.264
triton_vec      0.370    0.118   # 2D load+atomic
cuda_naive      0.274    0.104
```

- On **sorted** input: triton_vec 0.118 ≈ cuda_naive 0.104 → vectorizing
  **closes the gap** (compiler difference nearly vanishes when atomics
  have locality).
- On **random** input: triton_vec 0.370 still > cuda_naive 0.274 → the
  residual 1.35x is nvcc's cache-line atomic coalescing on high
  contention, which Triton's backend can't replicate.

## Decision rule

After the three tests:

1. **Input naturally ordered** (LiDAR scan lines, sorted batches)? →
   Triton vectorized ≈ CUDA. **No need for CUDA.** Triton is enough.
2. **Input random, high atomic contention** → CUDA ~1.35-2.5x faster,
   gap is compiler (atomic lowering + cache-line coalescing), NOT
   expressivity. Use CUDA if the 1.5x matters; don't blame Triton
   "expressivity".
3. **Point-sparse shape** (avg <1 point/bin) → block reduction gives
   only ~10%. Don't over-engineer reduction; vectorize first (cheap 1.5x).
4. Never conclude "Triton expressivity limit" without the Test-2
   same-structure A/B. The `tl.static_shared`/`tl.shared` absence is a
   red herring — Triton has `tl.sort`/`tl.reduce`/`tl.sum(axis=)` that
   implement block reduction internally.

## Verification

The three numeric tables above are the verification. Reproduce on a
single GPU (RTX 2080 Ti, Triton 3.7, CUDA 13) by running the kernels in
`examples/optimize_voxelization/`:
`triton_naive.py`, `triton_vec.py`, `kernel_cuda_naive.py`,
`kernel_cuda.py` (+ sorted-input variant by argsort in problem.py).

## Notes

- A subtle confound: CUDA's `float4` load vs Triton's 4×`tl.load`. Test 2
  uses `float4` in cuda_naive (so it's not a pure compiler-only
  comparison — load vectorization is mixed in). To fully isolate, write
  cuda_naive with 4×scalar `__ldg` loads; the gap to float4-version
  isolates load cost from atomic-lowering cost. (In practice the combined
  "compiler+vectorization" attribution is what matters for the decision.)
- `tl.sort` returns a single sorted tensor (no permutation in 3.7); to
  sort multiple arrays by one key you need the `tl.join`/broadcast trick
  or sort-then-recompute — which is why multi-column block reduction is
  awkward (not impossible) in Triton. But Test 2 shows reduction isn't
  the bottleneck anyway on sparse input.
- The 0.022ms bandwidth floor (13.7MB @ 616GB/s) is NOT achievable for
  atomic-bound scatter — atomics serialize. Real floor ≈ atomic
  throughput × N, far above bandwidth floor. Don't chase the bandwidth
  number on scatter kernels.

## References

- Empirically established on RTX 2080 Ti (SM7.5) + Triton 3.7.1 + CUDA
  13.0, 2026-07-25. Numbers: triton_naive 0.557/0.264, cuda_naive
  0.274/0.104, triton_vec 0.370/0.118 (random/sorted).
- KernelAgent `docs/具身3D算子优化实录.md` "Voxelization 的 CUDA vs Triton
  深度对照" — full tables and the wrong→right attribution correction.
- Related: the `reasoning-model-thinking-budget` skill (different domain,
  same lesson: don't accept the first plausible attribution — run the
  isolating experiment).
