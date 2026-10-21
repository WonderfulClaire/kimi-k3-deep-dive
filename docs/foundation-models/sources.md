# 官方来源与阅读定位

核对日期：**2026-10-21**。正文固定采用 **Kimi K3 技术报告 v1** 的架构与训练说明；Kimi Linear 和 AttnRes 官方仓库用于追溯方法与实验归属。下面的日期区分论文版本日期与仓库访问日期。

## 1. 文献与版本

| 标题 | 日期 / 版本 | URL | 本仓库怎样使用 |
| --- | --- | --- | --- |
| Kimi K3: Open Frontier Intelligence | v1：2026-07-27；版本页另列 v2：2026-08-07 | [固定 v1 HTML](https://arxiv.org/html/2607.24653v1) · [版本记录](https://arxiv.org/abs/2607.24653) | 本轮逐节阅读的是 v1；规格、公式和系统配置按该版本引用，没有声称已逐项比较 v1/v2 |
| Kimi Linear: An Expressive, Efficient Attention Architecture | v1：2025-10-30；v2：2025-11-01 | [arXiv](https://arxiv.org/abs/2510.26692) · [官方仓库](https://github.com/MoonshotAI/Kimi-Linear) | 阅读摘要、官方 README 的 Overview、模型表及性能图说明；用于厘清 KDA 沿革与 75% / 6.3× 的实验归属 |
| Attention Residuals | v1：2026-03-16 | [arXiv](https://arxiv.org/abs/2603.15031) · [官方仓库](https://github.com/MoonshotAI/Attention-Residuals) | 阅读摘要、官方 README 的方法、伪代码和实验设置；K3 的具体 block 配置另引 K3 v1 §2.2 |
| Kimi Linear 官方 README | 仓库访问：2026-10-21 | [README](https://github.com/MoonshotAI/Kimi-Linear#overview) · [raw](https://raw.githubusercontent.com/MoonshotAI/Kimi-Linear/master/README.md) | 48B total / 3B activated、1M context；最多 75% KV reduction 与性能图中的 6.3× TPOT；不作为 K3 实测 |
| Attention Residuals 官方 README | 仓库访问：2026-10-21 | [README](https://github.com/MoonshotAI/Attention-Residuals#overview) · [raw](https://raw.githubusercontent.com/MoonshotAI/Attention-Residuals/master/README.md) | pseudo-query、depth softmax、Block AttnRes、伪代码；48B/A3B 与 1.4T token 的独立实验设置 |

## 2. Kimi K3 v1 的逐节支持表

下表中的标题均为 *Kimi K3: Open Frontier Intelligence*，版本日期均为 **2026-07-27**。每个 URL 的 fragment 已对照公开 HTML 中的 id 核对。

| 小节 / 图表 | 原文 URL | 支持什么 | 正文位置 |
| --- | --- | --- | --- |
| §2 / Figure 2；§2.1 Hybrid Attention | [§2](https://arxiv.org/html/2607.24653v1#S2) · [§2.1](https://arxiv.org/html/2607.24653v1#S2.SS1) | 三种 mixing、3:1 周期、额外末层 Gated MLA、视觉输入路径 | [架构 §1](../architecture.md#1-先把模型规格与计算路径对上) |
| §3.2 / Table 1 | [§3.2](https://arxiv.org/html/2607.24653v1#S3.SS2) | 2.78T / 104.2B、93 层、69 KDA + 24 MLA、7,168/3,584、专家与视觉规格 | 架构规格表 |
| §2.1.1 / Eq. 1–2 | [Kimi Delta Attention](https://arxiv.org/html/2607.24653v1#S2.SS1.SSS1) | 单头维度、decay 与 delta-rule 顺序、Q/K/V 投影 | 架构 §2.1–2.2 |
| §2.1.1 / Eq. 3–4 | [Chunkwise parallel form](https://arxiv.org/html/2607.24653v1#S2.SS1.SSS1.Px1) | 块内并行、块间状态、UT 产生的 pseudo-values、因果对角线 | 架构 §2.3；预训练 §7 |
| §2.1.1 / Eq. 5 / Figure 3 | [Lower-bounded decay](https://arxiv.org/html/2607.24653v1#S2.SS1.SSS1.Px2) | scaled sigmoid、g_min=-5、16-token tile 的范围、对角 tile GEMM 路径 | 架构 §2.4；预训练 §7–8 |
| §2.1.1 / Eq. 6 | [Full-rank gate](https://arxiv.org/html/2607.24653v1#S2.SS1.SSS1.Px3) | KDA 输出门的 full-rank 参数化与 head-wise RMSNorm | 架构 §2.4 |
| §2.1.2 / Eq. 7 | [Gated MLA](https://arxiv.org/html/2607.24653v1#S2.SS1.SSS2) | latent KV、NoPE、输出门、训练输出 FP32 与 kernel 配套 | 架构 §3、§7；预训练 §8 |
| §2.2 / Eq. 8–10 | [Attention Residuals](https://arxiv.org/html/2607.24653v1#S2.SS2) · [Block form](https://arxiv.org/html/2607.24653v1#S2.SS2.SSS0.Px2) | pseudo-query、RMSNorm keys、depth softmax、8 个计算块、embedding 来源与状态复杂度 | 架构 §4 |
| §2.3–2.3.1 / Eq. 11 | [Stable LatentMoE](https://arxiv.org/html/2607.24653v1#S2.SS3) | 全宽共享专家、潜在路由路径、聚合后 RMSNorm、2 个共享专家 | 架构 §5.1；预训练 §5.1 |
| §2.3.2 / Eq. 12 / Appendix B | [SiTU-GLU](https://arxiv.org/html/2607.24653v1#S2.SS3.SSS2) | 两条乘法分支的 smooth cap、4/25 参数、中间激活的界限 | 架构 §5.2；预训练 §8 |
| §2.3.3 / Eq. 13–14 | [Quantile Balancing](https://arxiv.org/html/2607.24653v1#S2.SS3.SSS3) | bias 只用于 Top-K 选择、目标负载、cutoff、下一步更新、全局 histogram | 架构 §5.3；预训练 §5.1 |
| §2.4 | [Native Vision](https://arxiv.org/html/2607.24653v1#S2.SS4) | MoonViT-V2 原生多模态路径与从头训练 | 架构 §1 |
| §2.5 | [Per-Head Muon](https://arxiv.org/html/2607.24653v1#S2.SS5) | Q/K/V momentum 按 head 分块做 Newton–Schulz，不是按 head 自适应学习率 | 架构 §6；预训练 §4 |
| §3.1 | [Pre-Training Data](https://arxiv.org/html/2607.24653v1#S3.SS1) | 四类文本、视觉材料、质量/去重/采样、rephrasing 与保真 | 预训练 §2 |
| §3.2–3.3 / Figure 7 | [Scaling Law](https://arxiv.org/html/2607.24653v1#S3.SS2) · [Training Recipe](https://arxiv.org/html/2607.24653v1#S3.SS3) | 合并配方的约 2.5× scaling efficiency、分别搜索 schedule 超参数、cosine/1% warmup/WD=0.1 | 预训练 §3 |
| §3.4 | [Long-Context Extension](https://arxiv.org/html/2607.24653v1#S3.SS4) | 长数据处理、全局依赖任务、8K/64K/256K/1M 长度课程、NoPE | 预训练 §2.3 |
| §4.1.4 | [MXFP4 Quantization-Aware Post-Training](https://arxiv.org/html/2607.24653v1#S4.SS1.SSS4.Px1) | 路由专家 MXFP4、相应激活 MXFP8，其他模块高精度；SFT/RL QAT 与 rollout 方案一致 | 架构 §7；预训练 §6.2 |
| §5.1.1 | [KDA Kernels across Regimes](https://arxiv.org/html/2607.24653v1#S5.SS1.SSS1) | FlashKDA、chunk 内/间重叠、SM 级 CP；TP 不缩短递推链 | 架构 §2.3；预训练 §7 |
| §5.1.2 / Eq. 17 | [KDA Context Parallelism](https://arxiv.org/html/2607.24653v1#S5.SS1.SSS2) | 累计转移与零初态状态、可结合组合、固定大小片段交换 | 预训练 §5.2 |
| §5.2–5.2.1 | [Pre-training infrastructure](https://arxiv.org/html/2607.24653v1#S5.SS2) · [Balanced EP](https://arxiv.org/html/2607.24653v1#S5.SS2.SSS1) | PP/VP/EP/DP/CP、MoonEP 冗余专家、静态 shape、零拷贝、expert GEMM 调度 | 预训练 §5 |
| §5.2.2 | [Memory-Efficient Training](https://arxiv.org/html/2607.24653v1#S5.SS2.SSS2) | activation 存储策略、FP8/offload/recompute、PP 峰值不均、梯度分片、Muon P2P | 架构 §6–7；预训练 §4、§6 |
| §5.2.3 | [Multimodal Encoder Optimization](https://arxiv.org/html/2607.24653v1#S5.SS2.SSS3) | 大图/长视频的动态 CP 与 pipeline bubble 中的 encoder 计算 | 预训练 §5.3 |
| §5.4.1–5.4.2 | [Prefix Cache](https://arxiv.org/html/2607.24653v1#S5.SS4.SSS1) · [Kernels](https://arxiv.org/html/2607.24653v1#S5.SS4.SSS2) | KDA/MLA 混合状态管理、decode 状态流量、小 batch MoE 的 memory-bound 特性 | 预训练 §6.3、§7.2 |

## 3. 本轮订正与独立推导

| 原先容易误读的说法 | 当前写法 | 根据 |
| --- | --- | --- |
| K3 的 KV cache 减少 75%，decode 提速 6.3× | 这些数字归于 Kimi Linear 的对应配置 | Kimi Linear 官方 README 的模型表与性能图说明 |
| K3 完整主干严格 3:1 | 周期是 3:1；额外末层使总数为 69:24 | K3 §2.1、Table 1 |
| 全注意力生成一个 token 需要二次复杂度 | 单步 decode 随历史长度增长；整段 prefill 有二次项 | 由注意力张量形状区分整段与单步计算 |
| 没被当前 token 选择的专家不耗任何资源 | 该 token 不执行其 FFN，参数状态、路由和 batch 内其他计算仍存在 | 稀疏条件计算与存储账目的分开推导 |
| latent 维度减半，所以整层速度翻倍 | 路由表示 payload 缩小，公共投影、共享路径与系统成本另计 | K3 Eq. 11；预训练讲义中的独立 payload 算例 |
| Per-Head Muon 为每头分配自适应学习率 | 按 head 分块正交化 Q/K/V momentum | K3 §2.5 |
| 全模型从预训练起使用 4 bit 权重 | MXFP4/MXFP8 是明确到模块的后训练 QAT；预训练另有 activation FP8 存储策略 | K3 §4.1.4、§5.2.2 |
| 2.78T × 4 bit 就是部署显存 | 1.39 TB 是全参数统一 4 bit 的理想数值 payload，实际加混合精度、scale、布局、副本与运行状态 | 字节单位换算与内存分项 |

二维 KDA 更新、两 rank 的 token 加权、状态转移组合、逻辑通信 payload、Adam 式状态量级和故障诊断流程，均为本文的教学构造或一般系统推导。它们没有被标成 K3 实测；建议实验也没有被写成已完成实验。正文没有引用公司内部文档、私人图表或内部实现信息。

[架构讲义](../architecture.md) · [预训练与系统讲义](04-pretraining-systems.md) · [返回仓库](../../README.md)
