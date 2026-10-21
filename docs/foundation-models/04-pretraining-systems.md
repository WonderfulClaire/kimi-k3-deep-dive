# 预训练与系统：把数据、更新和设备连成一条计算链

基模预训练的基本单位是一次有效的参数更新：读取一批数据，得到有效 token，计算损失和梯度，跨设备合并，再由优化器更新模型。架构决定每次更新有哪些算子，数据决定梯度来自什么分布，系统决定固定时间内能完成多少有效更新。

本讲义围绕 Kimi K3 官方技术报告 **v1（2026-07-27）** 的 §2、§3、§5 展开，核对日期为 **2026-10-21**。K3 的具体配置均附原文小节；下面的预算公式、数值算例与排障流程是独立的教学分析。先读[架构拆解](../architecture.md)，资料入口见[来源表](sources.md)。

## 1. 一步训练到底消耗了什么

### 1.1 从“多少条数据”换成“多少有效 token”

假设两张卡各有一条样本。卡 0 有 2 个参与预测的 token，平均损失为 3；卡 1 有 6 个有效 token，平均损失为 1。两卡损失直接平均得到 2，按 token 加权则是

$$
L=\frac{2\times3+6\times1}{2+6}=1.5.
$$

这两个值对应不同目标。训练语言模型时，padding、忽略区间与 packing 会让各 rank 的有效 token 数不同。若希望优化全局 token 平均损失，目标应为

$$
L(\theta)=-\frac{1}{N_{\mathrm{valid}}}
\sum_{r}\sum_{t}m_{r,t}\log p_\theta(x_{r,t}\mid x_{r,<t}),
\qquad N_{\mathrm{valid}}=\sum_{r,t}m_{r,t},
$$

其中 $r$ 是 rank，$m_{r,t}\in\{0,1\}$ 是 loss mask。DP 的梯度平均必须与这个分母一致。若框架最终对 $R$ 个 rank 的梯度取平均，局部 loss sum 需要配上 $R/N_{\mathrm{valid}}$ 的缩放；gradient accumulation 的额外除数也必须一起计入。比较实现时，应把同一小批数据放到单卡，直接核对完整梯度。

这说明一个常见问题：相同的 global batch size，以“样本数”对齐，未必以有效 token 或梯度目标对齐。长文本、图像 patch 数、视频帧数的变化还会让相同 token 数对应不同的计算量。

### 1.2 一步时间是一条关键路径

一次更新可拆成：数据准备 → 前向 → 反向 → 梯度通信 → optimizer → checkpoint。通信、offload 与计算可以部分重叠，因此有用的表达是

$$
t_{\mathrm{step}}\approx t_{\mathrm{compute}}
+t_{\mathrm{comm,exposed}}+t_{\mathrm{input,exposed}}+t_{\mathrm{idle}}.
$$

这里的 exposed 是未被其他工作遮住的部分。把所有通信耗时与 GEMM 耗时直接相加，会重复计算已经重叠的时间；只看最快的 GEMM 又会漏掉空闲与数据等待。记录吞吐时，用整个稳定训练区间的有效 token 数除以墙钟时间，同时记录重计算、恢复与 checkpoint 的成本。

K3 的系统章节沿这条思路安排 KDA 状态传播、专家 dispatch/combine、梯度 offload 与 vision encoder；这些优化共同决定训练进度。[原文：§5 Introduction、§5.2](https://arxiv.org/html/2607.24653v1#S5)

## 2. 数据：token 数、去重和质量如何改变同一笔预算

### 2.1 一份数据清单应该能回答什么

对一批公开材料，可以按以下顺序建立自己的小规模预训练流程：

| 阶段 | 输入与操作 | 保存的产物 | 最容易被忽略的影响 |
| --- | --- | --- | --- |
| 获取与解析 | 原始文件、许可信息、格式解析 | 原始内容 hash、来源、解析版本 | 损坏文件、模板、二进制片段会消耗预算 |
| 去重与分组 | exact hash、近重复簇、版本关系 | canonical 文档与 duplicate cluster | 同一内容跨训练/验证出现会改变评价 |
| 划分与质量处理 | 按文档簇/来源划分，再过滤、重写或合成 | split manifest、过滤与生成版本 | 先改写再查重可能隐藏近重复 |
| 分词与混合 | 固定 tokenizer，对各域采样 | 每域 token 数、采样率、重复次数 | 字节、文档和 token 是不同统计单位 |
| Packing 与 masking | 合并短样本、插入边界、构造 loss mask | token stream、文档边界、有效 token 数 | 跨文档的注意力与 recurrent state 必须按目标处理 |
| 训练交付 | 确定性数据顺序、断点游标 | shard 顺序、seed、消费位置 | 恢复后重复或跳过数据会改变训练轨迹 |

这份表是可执行的数据设计，不是 K3 内部清单。K3 公开 §3.1 披露了 Web、Code、Mathematics、Knowledge 四类文本，以及视觉材料；它使用规则、质量分类器、去重和小模型 ablation 决定采样，知识与数学的 rephrasing 还包含源文保真检查。报告没有在这些小节给出足以重建完整数据集的全部 token 总量和混合比例，所以实验中应报告自己真正使用的清单。[原文：§3.1](https://arxiv.org/html/2607.24653v1#S3.SS1)

### 2.2 重复数据为什么可能“看起来训得很快”

设有 100 个独立文档，复制每个文档十次，可以得到 1,000 条记录。模型仍然会对这些重复 token 做前向和反向；loss 也可能更快下降。但“被读取的 token 数”增加，不等于独立信息增加了十倍。

去重的目标也不应简化为删除所有相似内容：软件的不同版本、同题的多种解法可能有学习价值。要区分重复模板、抄转载、近重复变体与有新增信息的版本，并用按来源或重复簇隔离的验证集判断取舍。rephrasing 和合成数据还需要在生成前后保留 source ID，才能看出新样本是否只换了表面表达。

### 2.3 长上下文数据需要什么额外结构

把很多无关段落拼到 1M token，并不会自动创造长距离学习信号。任务必须让后面的预测依赖远处的信息；同时要记录模型是否可以只看局部就完成。

K3 的长数据流程包含 exact/fuzzy dedup、视频帧 perceptual hash、结构检查与质量过滤；在 cooldown 中上采样长材料，并构造信息分散在长上下文中的任务。长度课程为 **8K → 64K → 256K → 1M**，前两阶段在预训练，后两阶段在 cooldown。这样把较昂贵的长序列阶段放到训练后部。[原文：§3.4](https://arxiv.org/html/2607.24653v1#S3.SS4)

读这个安排时，要同时问三个问题：真正长且连贯的数据占多少；哪些预测需要远距离信息；相同 FLOPs 下，增加长序列训练会挤掉多少短数据更新。

## 3. Scaling：先定义预算，再选择模型与 token

### 3.1 一个起点公式及其适用范围

对主要由稠密矩阵乘法构成的 Transformer，常见的教学估算是

$$
C_{\mathrm{train}}\approx6ND,
$$

其中 $N$ 是参与主要计算的参数规模，$D$ 是训练 token 数；常数 6 粗略合并前向和反向。它省略了注意力的序列长度项、路由、重计算、视觉编码器与硬件利用率。用于 MoE 时，以 $N_{\mathrm{active}}$ 估算条件 FFN 的量级比用全部专家参数合理，但仍应逐算子补齐公共投影与其他计算。

这个公式不提供墙钟时间。两个方案有相似 FLOPs，如果其中一个产生大量小 GEMM、跨机 all-to-all 或更高的 activation 峰值，实际更新速度仍可能差很多。

一个纯教学例子：若可分配的矩阵计算预算固定，方案 A 的 $N$ 是方案 B 的两倍，按上式可训练的 $D$ 约为一半。这不告诉我们谁的验证 loss 更好；必须拟合或实测该模型族在不同 $N,D$ 下的表现。MoE 的总容量与激活量又是两个自变量，单一的“tokens / 总参数”不足以概括最优训练点。

### 3.2 一次可信的小规模搜索怎样做

先固定验证分布与总预算口径，再比较模型宽深、专家数量与 Top-K、token 数、batch、学习率和长度课程。先用较小规模排除明显失败的配置，再检查相邻规模是否遵循同一趋势。记录有效 token、实际 FLOPs 估算、墙钟与验证 loss，避免只优化其中一个量。

K3 §3.2 把 model shape、TPP、batch size 与 learning rate 重新纳入 scaling 搜索，并在 held-out OOD validation 上报告整体约 **2.5× scaling efficiency**。这是架构、数据与训练配方的合并比较，不能说成单独 KDA 的速度或某个 benchmark 的 2.5× 提升。[原文：§3.2、Figure 7](https://arxiv.org/html/2607.24653v1#S3.SS2)

同节还比较 cosine 与 WSD：两种 schedule 分别搜索各自最优的峰值学习率与 batch，再比较最终 loss。这给出一个很具体的方法论：当改动会改变最优超参数时，把两边绑在同一套超参数上，测到的可能是调参匹配程度。K3 的实际 recipe 使用 cosine、1% linear warmup、weight decay 0.1；本文不补写原文未给出的完整逐阶段学习率表。[原文：§3.2–3.3](https://arxiv.org/html/2607.24653v1#S3.SS2)

## 4. Optimizer：更新方向、数值范围与分片必须一致

### 4.1 Per-Head Muon 的含义

先用一个示意更新描述操作顺序：梯度 $G_t$ 更新 momentum $M_t$，然后构造接近正交的矩阵方向 $O(M_t)$，最后按学习率更新参数。若有 $M=U\Sigma V^\top$，理想 polar factor 是 $UV^\top$；实际 Newton–Schulz 迭代用矩阵运算近似相应变换。这个示意省略具体 momentum、尺度修正与 weight decay，不能直接当作 K3 optimizer 源码。

K3 披露的改动是：对 Q/K/V 的 momentum 按 attention head 分块，分别做 Newton–Schulz。这样改变了不同 head 在矩阵变换中的耦合，减少大尺度 head 对共同更新方向的支配；它不等于给每个 head 配一个自适应学习率。[原文：§2.5](https://arxiv.org/html/2607.24653v1#S2.SS5)

检查这种实现，至少要核对切分轴、transpose 后的 head 布局、每个块的矩阵形状与 norm。把同一矩阵任意切成连续的若干 shards，并不等于按 head 正确切块。

### 4.2 为什么优化器会引入另一类通信

DP sharding 追求把状态均匀分散，矩阵正交化却需要同一参数矩阵的完整结构。若每张卡都 all-gather 整个 parameter buffer，会引入临时副本与网络流量。

K3 的 P2P 方案让各 rank 只获取自己负责处理的参数矩阵所需 shards，并按 model chunk 重叠获取与更新。这里需要矩阵级 ownership；仅有“第几字节归哪张卡”的扁平分片映射还不够。[原文：§5.2.2，P2P-based Muon orthogonalization](https://arxiv.org/html/2607.24653v1#S5.SS2.SSS2.Px6)

### 4.3 观察更新，而不只观察 loss

每层可记录 gradient norm、momentum norm、parameter norm、update/weight ratio，并分模块查看非有限值。对 head 分块更新，观察不同 head 的更新尺度；对 MoE，观察长期未使用专家及过热专家；对门控，观察饱和区比例。出现 loss spike 时，这些量能帮助判断问题是在数据、前向激活、反向梯度还是 optimizer update 首先出现。

记录这些统计量的成本也应受控：全量张量只在触发异常时保存，正常训练保存聚合统计与少量确定性样本。

## 5. 并行：每一种切分都要画出通信对象

| 并行方式 | 切分什么 | 常见通信对象 | 会暴露的代价 |
| --- | --- | --- | --- |
| DP | batch / 数据 | 梯度；分片优化器所需状态 | 梯度同步、optimizer state 与 all-gather |
| TP | 单个大矩阵、head 或通道 | 计算中间结果 | 更频繁的 collective 与小 GEMM |
| PP / virtual stages | 网络层 | 相邻 stage 的激活及其梯度 | bubble、stage 负载差与在途激活 |
| EP | 路由专家 | token 表示与专家输出 | dispatch/combine、负载倾斜 |
| CP | 序列区间 | KV 块或状态转移片段 | 边界状态、长序列通信与重构 |

这是通用切分表。K3 的预训练组合明确包含 PP/VP、EP、ZeRO-1 DP、Pipeline ZeRO-2 与 CP；不要把推理时可能使用的 TP 配置擅自填成它的完整预训练拓扑。[原文：§5.2](https://arxiv.org/html/2607.24653v1#S5.SS2)

### 5.1 EP：负载均衡为什么要分统计层与执行层

一个 micro-batch 即使总 token 数固定，各专家收到的 token 也会变化。全局专家利用率影响学习；某一时刻哪张卡最忙影响 step time。

QB 用全局统计调整 router bias，MoonEP 则按当前 micro-batch/layer 规划冗余专家并迁移所需参数，使 rank 级工作量平衡。报告同时指出：rank 的总工作量相同，rank 内不同专家的 token 数仍可能倾斜，因而还需要 workload-aware grouped GEMM scheduling。[原文：§2.3.3](https://arxiv.org/html/2607.24653v1#S2.SS3.SSS3)、[§5.2.1](https://arxiv.org/html/2607.24653v1#S5.SS2.SSS1)

先算一笔 payload。教学例子取一张卡的 4,096 个 token、Top-K=16、每个元素 2 bytes：若每次路由复制全宽 7,168 维表示，单向逻辑 payload 为

$$
4096\times16\times7168\times2=896\ \mathrm{MiB}.
$$

路由表示宽度降为 3,584 时变成 448 MiB。这里只算 dispatch 一次的逻辑数据量；combine、反向、padding、远程比例、冗余专家迁移与通信重叠都另计。这个例子支持“latent 路由缩小了该 payload”，不能推出端到端延迟减半。

### 5.2 CP：KDA 的局部状态不能直接相加

考虑某一段 token 对输入状态的作用：

$$
S_{\mathrm{out}}=A S_{\mathrm{in}}+B.
$$

$A$ 是该段累计的状态转移，$B$ 是该段从零状态出发生成的状态。连续两段的组合是

$$
(A_2,B_2)\circ(A_1,B_1)
=(A_2A_1,\ A_2B_1+B_2).
$$

这个组合可结合，所以能构造 prefix scan。只把各段的 $B$ 相加会漏掉后续段的遗忘与 delta-rule 变换。例如标量教学例子中，第一段为 $(A_1,B_1)=(0.5,2)$，第二段为 $(0.25,3)$，零初态的最终状态是 $0.25\times2+3=3.5$，直接加会得到 5。

K3 KCP 交换的是累计转移与局部零初态状态，随后按同一文档的顺序恢复边界状态；固定 head/state 维度下，交换片段的大小不随序列长度增长。全局 MLA 层仍有自己的长序列通信，不能由 KCP 的特性概括整个混合模型。[原文：§5.1.2，Eq. 17](https://arxiv.org/html/2607.24653v1#S5.SS1.SSS2)

### 5.3 PP：activation 峰值不在所有 stage 上同时相同

pipeline warmup 和在途 micro-batch 数会让早期 stage 保存更多 activations；只把层数平均分配，不保证显存或时间均衡。K3 报告使用 remote offload，把部分 activation 转到其他 PP ranks；它还把视觉编码器的大部分计算放入 pipeline bubbles，并对大图/长视频做动态 CP。[原文：§5.2.2–5.2.3](https://arxiv.org/html/2607.24653v1#S5.SS2.SSS2)

评估这种优化应测量整个 pipeline 的 makespan、每 stage 的高水位显存、P2P 暴露时间与 bubble。单独测 vision encoder 的 kernel 更快，并不能决定它是否仍在关键路径上。

## 6. 把内存账算到字节，再谈能放下

### 6.1 参数、梯度、优化器与激活分开列

训练峰值显存可先写成

$$
M_{\mathrm{peak}}=M_{\mathrm{weights}}+M_{\mathrm{grads}}
+M_{\mathrm{optimizer}}+M_{\mathrm{activations}}
+M_{\mathrm{workspace}}+M_{\mathrm{buffers}}+M_{\mathrm{fragmentation}}.
$$

这是账目分类；有些实现会复用 buffer，因此精确峰值要按张量的存活区间计算，不能把各类独立峰值无条件相加。下面是教学用的未分片 Adam 式配置：

| 状态 | 每参数 bytes | 说明 |
| --- | --- | --- |
| BF16 权重 | 2 | 前后向使用的模型副本 |
| BF16 梯度 | 2 | 实现也可能采用其他梯度格式 |
| FP32 master weights | 4 | 若实现保留 |
| FP32 一阶与二阶矩 | 8 | 两份各 4 bytes |
| 合计 | 16 | 尚未加入激活与临时 workspace |

套入 2.78T 得到 44.48 TB，是该教学配置的全部模型状态量级。**这不是 K3 的实测内存，也不是 Muon 状态的准确布局**；Muon 的状态、混合 optimizer 与临时正交化空间要按实际实现另列。

ZeRO-1 主要切 optimizer states，ZeRO-2 再切梯度；EP 切路由专家，PP 切层，TP 切矩阵。共享参数、副本、通信 buffer 和 activation 的分片方式各不相同，不能把总显存统一除以“总卡数”。K3 实际还组合 activation FP8、offload、recompute、PP 梯度分片与 CPU 累积。[原文：§5.2.2](https://arxiv.org/html/2607.24653v1#S5.SS2.SSS2)

### 6.2 4 bit 权重不等于整个模型只占 4 bit/parameter

先算理想 payload：全部 2.78T 参数用 4 bit 表示需要 1.39 TB；用 BF16 则是 5.56 TB。这是十进制 TB，分别约为 1.26 TiB 与 5.06 TiB。

实际采用混合格式时，可写为

$$
M_W\approx P_q\,q/8+P_h\,h/8
+M_{\mathrm{scales}}+M_{\mathrm{padding/layout}}+M_{\mathrm{replicas}},
$$

其中 $P_q$ 是量化参数数，$P_h$ 是高精度参数数，$q,h$ 为对应 bit 数。K3 的后训练配置只把路由专家权重设为 MXFP4，非路由专家部分与共享专家保留高精度，相应量化激活采用 MXFP8。因此全参数统一乘 0.5 bytes 只能作为理想量级。[原文：§4.1.4](https://arxiv.org/html/2607.24653v1#S4.SS1.SSS4.Px1)

checkpoint 文件大小、加载后的 resident weights、推理峰值显存是三个观测值：文件可能分片或压缩；加载可能发生转置、重排和复制；推理还需要 KV/state cache、激活、通信和内核 workspace。部署时应分别测量这三者。

### 6.3 Cache 与 recurrent state 分别随什么增长

通用 MHA 的理想 KV 数值 payload 为

$$
M_{\mathrm{KV}}=2BTH_{\mathrm{KV}}d_hs,
$$

其中 $B$ 是并发序列数，$T$ 为历史长度，$H_{\mathrm{KV}}$ 是 KV 头数，$d_h$ 为头维度，$s$ 是元素 bytes。MLA 改为存 latent 时，应换成实际 latent cache 的宽度，不能继续套 MHA 头数。

KDA 的主状态量级为

$$
M_{\mathrm{KDA}}\approx B L_{\mathrm{KDA}}H d_kd_vs,
$$

还要加 ShortConv 状态、快照、对齐和调度元数据。它与每条序列的历史长度无关，但会随 batch、头数、状态维度和层数增长；“固定状态”不等于状态免费。K3 的完整缓存还要加 24 层 MLA 的序列相关部分。[原文：§5.1](https://arxiv.org/html/2607.24653v1#S5.SS1)、[§5.4.1](https://arxiv.org/html/2607.24653v1#S5.SS4.SSS1)

### 6.4 Recompute 与 offload 如何选择

重计算以更多 FLOPs 换更少保存的 activation；CPU/remote offload 以传输和调度换显存。选择取决于有无可重叠的计算窗口以及链路带宽。

K3 的 unified activation manager 把 recompute、quantize、offload 作为 tensor 级存储策略；MoE 还通过重新 dispatch 恢复部分 group-GEMM 输入，减少前向保存量。这些选择的目标是降低峰值且尽量隐藏额外工作。[原文：§5.2.2，Unified activation manager、Memory-efficient MoE](https://arxiv.org/html/2607.24653v1#S5.SS2.SSS2)

## 7. Chunkwise 为什么适合 GEMM，什么时候仍然不快

### 7.1 从小向量更新变成矩阵乘法

逐 token 的 KDA 更新反复读写一个 $d_k\times d_v$ 状态。如果每次只做少量乘加，数据移动和 launch 的成本可能盖过计算。把 $C$ 个 token 组织成矩阵后，可以一次执行 $C\times d_k$ 与 $d_k\times d_v$ 的乘法，复用状态和权重，提高并行度。

但 $C$ 越大，块内交互、中间矩阵和 shared memory 压力也越大；$C$ 太小又会增加递推边界和调度成本。它是有限资源下的取舍，不能简单地让 chunk 无限大。K3 的 lower-bounded decay 还改变了可用内核：有限范围使对角 tile 能走 dense Tensor Core GEMM，从算法参数化上消除了特定的 position-pair 路径。[原文：§2.1.1，Lower-bounded decay](https://arxiv.org/html/2607.24653v1#S2.SS1.SSS1.Px2)

### 7.2 同样是矩阵乘法，瓶颈也可能不同

对某个 kernel，若计算量为 $F$ FLOPs、搬运量为 $B_m$ bytes，可用一个粗略的下界思考时间：

$$
t\gtrsim\max\left(\frac{F}{\mathrm{effective\ compute}},
\frac{B_m}{\mathrm{effective\ bandwidth}}\right).
$$

这不是实际时延预测，还省略 launch、依赖和同步。它帮助区分两类优化：算术密度高的训练/prefill 更可能从大 GEMM 与 Tensor Core 获益；小 batch decode 可能主要在搬运权重与状态，减少访存和融合更重要。

K3 报告将 prefill 的 FlashKDA 与 decode 的状态管理分开优化；路由专家 decode 在小 batch 时也可能退化为 memory-bound 的权重流。把一个训练 GEMM 的峰值吞吐直接拿来估算在线 token/s，容易错过主瓶颈。[原文：§5.1.1](https://arxiv.org/html/2607.24653v1#S5.SS1.SSS1)、[§5.4.2](https://arxiv.org/html/2607.24653v1#S5.SS4.SSS2)

## 8. 低精度与故障诊断：先定位第一处偏差

### 8.1 动态范围与有效精度是两件事

教学量化器可写成

$$
q=\operatorname{clip}(\operatorname{round}(x/s),q_{\min},q_{\max}),
\qquad \hat x=sq.
$$

scale 太大，小值容易被舍入到零；太小，大值容易被截断。block-wise scale 能让不同区域使用不同尺度，但需额外存 scale，并关注跨块分布差异。BF16 的指数范围较大，不能据此推断乘加结果与 FP32 等精度。

K3 的三种数值处理分别落在不同位置：lower-bounded KDA decay 控制特定中间量范围；SiTU-GLU 抑制专家乘法分支的大值；MLA 训练输出保留 FP32 以处理 rounding bias。它们不能互相代替。[原文：§2.1.1](https://arxiv.org/html/2607.24653v1#S2.SS1.SSS1.Px2)、[§2.3.2](https://arxiv.org/html/2607.24653v1#S2.SS3.SSS2)、[§2.1.2](https://arxiv.org/html/2607.24653v1#S2.SS1.SSS2)

### 8.2 让排障实验每次只改变一个关键因素

| 观察到的现象 | 先记录什么 | 最小对照 | 可能定位到的问题 |
| --- | --- | --- | --- |
| 某个 batch 后 loss 突升 | 数据 ID、有效 token、各模块首次非有限值 | 同一 batch、较高精度、更新前 checkpoint 重放 | 坏样本、激活异常或数值路径 |
| chunkwise 与 recurrence 不一致 | 初态、边界、Q/K/V、decay、误差随长度曲线 | 单头、短序列、FP64/FP32 参考 | mask 对角线、状态顺序、累计 decay |
| 多卡梯度与单卡不一致 | 各 rank 有效 token、loss 分母、grad reduction | 不等长 batch 的中心化梯度 | 归一化、padding、重复平均 |
| MoE 某几张卡长期变慢 | 每专家/每 rank token 数、通信与 GEMM 时间 | 固定路由与固定 shape | 统计倾斜、执行负载或调度 |
| loss 平稳但验证恶化 | 按域验证、duplicate cluster、采样次数 | 同预算调整数据混合 | 数据重复、分布偏移或过拟合 |
| 只在恢复后出现跳变 | optimizer、RNG、数据游标、router bias | 相同断点连续跑与恢复跑 | 状态遗漏或样本重复/跳过 |
| 随长度增长出现 NaN | state norm、log-decay、tile 内 rescale | recurrence 与分块的长序列对照 | 溢出、下溢或累计误差 |
| 某 stage OOM，其他卡空闲 | 每 stage activation 峰值与在途 micro-batch | 减 micro-batch / 改重算策略 | PP 激活不均、buffer 生命周期 |
| 通信 hang | 最后成功的 collective、各 rank shape 与序号 | 单机同布局、固定 shape | collective 顺序或 shape 不一致 |

表中的定位流程是自己的实验方法，不宣称为 K3 的线上故障记录。处理数值错误时，先找到第一处偏差，再决定提高精度、改归一化、修边界或调整调度；直接换 optimizer 往往会遮住根因。

### 8.3 一份可恢复的 checkpoint 需要哪些状态

除权重之外，至少还应保存 optimizer 与 scheduler 状态、RNG、已消费 token/数据位置、模型与 tokenizer 配置、分片映射及影响训练的 router 状态。若改变并行拓扑，还应验证 reshard 后矩阵与 head 布局、momentum、共享专家副本的一致性。

最有效的恢复测试是固定小数据流：连续训练若干步，与中途保存/恢复后的若干步比较 loss、参数和后续数据 ID。误差容限要匹配实际精度与非确定性算子，不能只检查文件能否加载。

## 9. 把阅读变成三项可追问的验证

这些是建议开展的小实验，不是本仓库已经完成的测量结果。

1. **状态转移的可组合性。** 用二维状态与固定 Q/K/V/decay，实现逐 token 参考；再按不同长度分段，通过 $(A,B)$ 组合恢复状态，检查每个边界与输出。加入两个独立文档，验证 state reset。
2. **有效 token 的梯度目标。** 用一份小模型和不等长样本，对照单卡全局 token 平均、正确加权的多 rank 版本、局部均值直接平均。比较梯度，说明目标变化发生在哪里。
3. **精度与内存的分账。** 对一个公开小矩阵分别记录数值 payload、scale、布局 padding 和计算误差；在固定输入上比较不同块尺度。把公式估算与实际分配的内存并排展示。

向基模团队讲解时，可以沿一条明确链路组织答案：**训练目标 → 数据与 token → 模型条件计算 → 优化器更新 → 通信与显存 → 数值及故障定位**。任何性能数字都应跟着这条链路说明模型版本、长度、batch、精度、硬件与计时范围。

[返回架构](../architecture.md) · [来源表](sources.md) · [返回仓库](../../README.md)
