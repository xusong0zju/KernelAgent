# 用 CUDA 当性能探针优化 Triton——一次跨范式突破的复盘

> 记一次反直觉的优化经历：为了优化 Triton 算子，我们先写了 CUDA，再把 CUDA 的技巧翻译回 Triton，反而让 Triton 更快。本文复盘全过程，提炼出"跨语言探针"的方法论，并讨论对 KernelAgent（单语言自动优化 agent）的设计启示。
>
> 写于 2026-07-25。配套 skill：`docs/skills/cross-language-probe-paradigm-escape.md`。完整实测数据：`docs/具身3D算子优化实录.md`。

---

## 一、先澄清一个误解

有人会读成"用 CUDA 之后 Triton 才更优"，好像 CUDA 教会了 Triton。时间线其实是这样：

1. **先**用 Triton（LLM 闭环）优化 voxelization → 0.456ms，触顶
2. **然后**手写 CUDA → 0.246ms
3. **最后**把 CUDA 的招数移植回 Triton → triton_vec **0.370ms**

所以"更优"是相对 **Triton 自己的朴素版**（0.557 → 0.370，快 1.5x），不是相对 CUDA（random 下 CUDA 0.274 仍领先）。Triton 全程没超过 CUDA。

真正的问题是：**为什么写完 CUDA，才写出更好的 Triton？** 答案有三层。

---

## 二、第一层：CUDA 充当"性能探针"，拆开了混在一起的瓶颈

写 CUDA 前，我只知道"Triton 慢、0.456 触顶"，**不知道慢在哪**。是 atomic 次数？归约没做？编译器？算索引开销？一团迷雾。

写 CUDA 的过程**逼着我把每件事单独做一遍**，等于做了一组受控实验：

| CUDA 版本 | 单独做的事 | 隔离出的变量 |
|---|---|---|
| CUDA 用 `float4` 一次读 16B | load 向量化 | "load 向量化"是独立变量 |
| CUDA 用原生 `atomicAdd` | atomic lowering 质量 | "atomic 质量"是独立变量 |
| CUDA 做 block 归约 | 段归约 | 后来发现它只值 ~10% |

**CUDA 不是"更好的实现"，是"把混在一起的因素拆开的工具"**。没写它，我看不到"float4 load + cache-line atomic 合并"才是主因，只会继续在"是不是该加 block 归约"上打转——而归约其实只值 10%。

这正是物理实验里"标准样品"的作用：**一个已知答案的参照系，能让你隔离变量**。单框架里所有因素混在一起（一次 `tl.atomic_add` 同时体现 atomic 次数、lowering 质量、load 是否向量化、cache-line 行为），你看的是总时间，分不出谁贡献多少。

---

## 三、第二层：换语言 = 换一套"什么显然、什么不显然"的直觉

写 Triton 时，思维落在 Triton 原语里：`tl.atomic_add` 逐标量、`tl.static_range` 循环、要不要 `tl.sort`……在这个范式里，"每次 atomic 加一个标量"是默认写法，**不觉得它慢**——因为 Triton 文档例子都这么写。

写 CUDA 时范式完全不同。CUDA 程序员的本能是：

- "**能用 `float4` 就别用 4 个 float**"
- "**同 cache line 的 atomic 要尽量合并**"

这是 CUDA 生态几十年的肌肉记忆，写在每个教程里。

移植回 Triton 时，我带上了这副"CUDA 眼镜"，开始问："Triton 里有没有等价 `float4` 的东西？"——于是找到了 **2D block ptr load + 2D `tl.atomic_add`**（一次加 [BLOCK,4]）。

**这招 Triton 一直支持，但纯在 Triton 范式里想不到去用**——因为 Triton 文档没强调"scatter 要向量化 atomic"，它的例子都是 elementwise。

> **关键认知：换个语言/范式，等于换一套"什么显然、什么不显然"的直觉。** 在 Triton 里"4 次标量 atomic"是显然的；在 CUDA 里"4 次标量 atomicAdd 而不用 float4"是反直觉的、一看就知道亏。这个直觉差，纯待在 Triton 里永远跨不过去。

---

## 四、完整数据复盘（RTX 2080 Ti, Triton 3.7, CUDA 13）

| | random | sorted | 说明 |
|---|---|---|---|
| triton_naive（朴素 5-atomic） | 0.557 | 0.264 | Triton 范式的"自然写法" |
| **triton_vec（移植 CUDA 技巧后）** | **0.370** | **0.118** | 2D load+atomic，Triton 一直支持但原没想到用 |
| cuda_naive（同结构，无归约） | 0.274 | 0.104 | 隔离出"编译器差距" |
| cuda_v1（block 归约） | 0.246 | 0.100 | 隔离出"归约只值 ~10%" |

读法：
- **triton_naive → triton_vec**：0.557 → 0.370（random 快 1.5x）、0.264 → 0.118（sorted 快 4.7x）。这是"跨语言探针"拿到的收益。
- **triton_vec vs cuda_naive**：sorted 下 0.118 ≈ 0.104（追平）；random 下 0.370 仍输 0.274。**移植能拿一部分，拿不到全部**——残余 1.35x 是 nvcc 的 cache-line atomic 合并，Triton 后端做不到的硬优势。
- **cuda_naive vs cuda_v1**：归约只值 ~10%。推翻了"归约是主因"的最初猜测。

---

## 五、对 KernelAgent 的设计启示

这对你做 KernelAgent（LLM 自动优化算子）有直接启示。

### 问题：单语言 agent 有范式盲区

如果 KernelAgent **只生成 Triton**，它困在"Triton 范式的显然性"里：
- 会优化 BLOCK 大小、加 mask、试 num_warps（A 范式内的旋钮）
- **想不到用 2D 向量化 atomic**（A 本就支持，但训练数据里 Triton scatter 例子都不这么写）

它需要一个"外部参照"才能跳出。我们这次是"LLM 在 Triton 里 3 轮到顶 → 切 CUDA → 再回 Triton"才突破的——**纯在 Triton 里再 30 轮也到不了 0.370**，因为 0.370 需要的技巧不在"Triton 旋钮空间"的搜索方向上。

### 三个可落地方向

1. **给 KernelAgent 加"CUDA 参照通道"**：对难优化算子，先让 agent（或人）写个 CUDA 探针版拆变量、定位主因，再让 LLM"把 CUDA 的关键技巧翻译回 Triton"。这次就是这流程的手动版。
2. **在 prompt 里显式注入跨语言直觉**：比如告诉 LLM"scatter-add 算子，优先尝试 2D 向量化 load+atomic，而非逐标量"——把 CUDA 生态的肌肉记忆显式喂给 Triton 优化提示。这能让 LLM 不绕道 CUDA 也拿到部分收益。
3. **单范式闭环多轮无突破时，换视角**：与其继续微调，不如换语言/换算法骨架。设一个"breakthrough detector"——N 轮无提升就触发范式切换，而非无限微调。

### 一个诚实的边界

跨语言探针**不保证追平**。这次 random 下 triton_vec 0.370 仍输 cuda 0.274——nvcc 的 cache-line atomic 合并是 Triton 后端做不到的硬优势。所以：
- 探针能帮你拿到**可移植部分**的收益（0.557→0.370，1.5x）
- 拿不到**编译器硬差距**（0.370 vs 0.274 的 1.35x）
- 若那 1.35x 关键，就用 CUDA 上线；否则 Triton 够用且更易维护、agent 能生成

---

## 六、方法论提炼（"跨语言探针"四步）

1. **在 B（常 CUDA/C）写参照实现**——不为上线，当探针。刻意把每件事单独做一遍。
2. **在 B 里做"拆变量"对照**——写多个只差一个变量的版本互比，把混在一起的因素拆开。这步必须在 B 里做（B 给你足够低层控制权）。
3. **把 B 的关键技巧翻译回 A**——带"B 的眼镜"重看 A：问"B 里起作用的技巧，A 有没有等价物？"这个翻译纯在 A 范式里想不到。
4. **诚实评估边界**——哪些能移植（A≈B，用 A）、哪些是 B 硬优势（A 追不上，关键就用 B）。

---

## 七、反例：什么时候不要用探针

- **简单算子**：A 一轮就优化好，不必绕道 B。探针有成本（写参照、对照实验）。
- **B 和 A 范式差异不够**：如 TVM→Triton 可能差异不足，探针看不出新东西。B 要选范式差异大、控制权低层的（CUDA/C 对 Triton/Python 是好选择）。
- **A 还没到顶**：A 里还有明显旋钮没试，先试完。探针是"A 调到顶、说不清瓶颈"时才上的手段。

---

## 八、与已有沉淀的关系

- **`docs/skills/cross-language-probe-paradigm-escape.md`**：本文的方法论精炼版（skill 形式，agent 可自动命中）。
- **`docs/skills/atomic-scatter-triton-vs-cuda-attribution.md`**：本文"第 2 步拆变量"在 atomic-scatter 场景的具体化（三个对照实验）。
- **`docs/具身3D算子优化实录.md`**：完整实测数据与代码。
- 同源教训：`docs/skills/reasoning-model-thinking-budget.md`——别接受第一个看似合理的归因，跑隔离实验。本文是这条原则的"跨语言"扩展。

---

## 九、一句话总结

写 CUDA 不是为了让 Triton 更优，而是 CUDA 当了"性能探针 + 范式转换器"：它把混在一起的瓶颈拆开（让我看到 float4+atomic 合并才是主因、归约只值 10%），又带来 CUDA 生态"向量化 atomic 是本能"的直觉。这两件事，纯待在 Triton 范式里都得不到。对 KernelAgent 的启示：**单语言 agent 有范式盲区，需要跨语言参照或显式注入跨语言直觉才能突破**。
