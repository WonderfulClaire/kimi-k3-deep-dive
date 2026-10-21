# 后训练：从哪些 token 学，到怎样沿自己的轨迹改进

一次 Agent 后训练可以沿着这条链检查：模型看到了什么上下文，哪些输出 token 产生梯度，环境如何判定最终状态，采样策略和更新策略是否一致，以及部署时使用的模型是否与训练前向相同。把这些问题连起来，SFT、RL、蒸馏和推理优化才成为同一个过程。

本讲义基于 [K3 §4](https://arxiv.org/html/2607.24653v1#S4)，公开原文核对日期为 2026-10-21。官方方法的简要关系是 SFT 冷启动 → 分领域和 effort 的 RL 专家 → MOPD 合并；下面的小算例和公式分析由本仓库独立编写。现有代码对应 [SFT 包装器](../../src/k3lab/train/sft.py)、[GRPO 包装器](../../src/k3lab/train/grpo.py)、[环境](../../src/k3lab/train/grpo_env.py)、[交互循环](../../src/k3lab/harness/agent.py) 与 [评测](../../src/k3lab/eval.py)。

## 1. SFT：先画出真正参与预测的两个 token

考虑一个简化的 token 序列：

```text
位置        0       1       2       3       4       5
token      BOS     用户1   用户2   答案     EOT     PAD
loss mask   0       0       0       1       1       0
```

这里要教模型生成“答案”和结束标记 EOT，用户内容提供条件。mask 为零不表示模型看不到该 token：用户 token 通常仍被 attention 读取，只是不作为这次损失的预测目标。若只屏蔽用户的 attention，就连问题本身也从条件里删掉了。

设 batch 为 B、补齐长度为 L、词表大小为 \(|\mathcal V|\)。模型 logits 为 `Z[B,L,|V|]`，token ID 为 `x[B,L]`，目标 mask 为 `m[B,L]`。causal LM 的第 t−1 个 logit 预测第 t 个 token，因此损失为

\[
L_{\mathrm{SFT}}=
-\frac{\sum_{b=1}^{B}\sum_{t=1}^{L-1}
m_{b,t}\log\operatorname{Softmax}(Z_{b,t-1})_{x_{b,t}}}
{\sum_{b,t\ge1}m_{b,t}}.
\]

库通常用 `labels=-100` 表示忽略，再在模型或 collator 内完成 shift。自行预先 shift 一次、模型内部又 shift 一次，会把目标错开一个 token。用上面六个位置逐一打印“哪一行 logits 预测哪个 token”，比只看训练 loss 更容易发现问题。

### 多轮工具交互的 mask 应该怎样画

一条轨迹可能是 `system → user → assistant(tool call) → tool(observation) → assistant(final)`。assistant-only 目标通常监督工具名、参数及最终回答等 assistant 生成区间；tool 返回的观察是环境提供的条件，不应被误当成策略自己要生成的答案。具体特殊 token 是否进入生成区间，由选定 chat template 决定。

[export_sft.py](../../src/k3lab/export_sft.py) 从 verifier 成功的记录重建这些角色消息；它导出 messages，而不构造 token labels。[sft.py](../../src/k3lab/train/sft.py) 再把 `assistant_only_loss` 交给 TRL。这个开关默认关闭。启用时，需要 template 提供正确的 generation 区域；当前 [TRL SFT 文档](https://huggingface.co/docs/trl/sft_trainer#train-on-assistant-messages-only) 说明了相关 template 条件和自动适配范围。

**先排查的失败现象：** loss 很快变小，模型却只会复述工具观察，或者不会停。先查看实际 `input_ids/labels`：用户、工具观察是否被监督；工具调用是否全部被 mask；结束标记是否被截断。成功轨迹经过序列化后，仍可能变成错误的监督目标。

### Packing：EOS 不会自动隔开两份记忆

把两条短样本 A、B 拼进一个训练块，可以减少 padding。若希望它们保持独立，B 的 attention 不应读取 A；对于 [KDA](01-kda.md) 这样的状态层，还需要在样本边界重置或分段传递 recurrent state。仅插入 EOS，或仅把 position ID 从零重启，都不自动保证这两件事。

把每个 token 的样本编号记为 \(s_t\)，独立样本的 causal 可见关系可以写为

\[
M^{\mathrm{attn}}_{ij}=\mathbf1[j\le i]\,\mathbf1[s_i=s_j].
\]

实际 backend 可以用块状 mask、varlen 边界等方式实现。loss mask 则必须在拼接、截断和 shift 后仍对应原来的 assistant 区域。若有意把 A 当作 B 的上下文，也可以使用连续流目标，但此时训练对象已经改变，不能把差异全归给“packing 更省显存”。

本包装器默认 `packing=False`、`max_length=2048`，没有自定义 collator、EOS 设置或边界检查。启动前应取两条样本，用实际 tokenizer 和 trainer collator 检查拼接结果，尤其检查最终回答和 EOT 是否还在。多轮模板还可能区分 end-of-turn 与 end-of-sequence，不能看到一个字符串叫 EOS 就认为停止行为一致。

[K3 §4.1.1](https://arxiv.org/html/2607.24653v1#S4.SS1.SSS1) 披露了经验证的 Agent 轨迹和 XTML 序列化。本仓库的通用 messages 导出没有实现 XTML，也没有从该小节推断官方所有 token mask 或 packing 细节。

## 2. GRPO：四条轨迹怎样给某个 token 一次更新

同一问题采四条回答，环境奖励为 \((1,0,1,0)\)。若教学算例采用总体标准差，均值为 0.5、标准差为 0.5，组内优势是 \((1,-1,1,-1)\)。终局奖励因此能给每条轨迹上的策略 token 分配同号学习信号；这并没有定位到底哪一步修复或破坏了任务。

设 G 是同一 prompt 的采样数，\(T_i\) 是第 i 条轨迹的有效策略 token 数。采样时策略叫 \(\pi_{\mathrm{old}}\)，正在更新的策略叫 \(\pi_\theta\)，另一个可选的长期锚点叫 \(\pi_{\mathrm{ref}}\)。三者的用途不同。

组相对优势的一种基础形式是

\[
\hat A_i=\frac{R_i-\bar R}{\sigma_R+\varepsilon_R}.
\]

实际库的标准差定义、缩放开关和全相同奖励处理必须按配置检查。若四条全为零，中心化后的优势也是零；继续增加相同失败轨迹，不会凭空得到能区分好坏的 policy-gradient 信号。参数是否完全不变，还受到 KL、其他 loss、weight decay 和 optimizer 状态影响。

### Ratio 与 clipping：算出 1.4 之后发生什么

某个已采样 token 在 old 策略下概率为 0.5，在当前策略下为 0.7，ratio 是 1.4。令 clip 区间为 [0.8,1.2]。若优势为 +1，更新方向已经过强，clipped surrogate 取 1.2；若优势为 −1，则取 −1.4，仍推动概率下降。

记当前已观察历史为 \(h_{i,t}\)，token 为 \(y_{i,t}\)：

\[
\rho_{i,t}=\exp\{\log\pi_\theta(y_{i,t}\mid h_{i,t})
-\log\pi_{\mathrm{old}}(y_{i,t}\mid h_{i,t})\},
\]

\[
u_{i,t}=\min\{\rho_{i,t}\hat A_i,
\operatorname{clip}(\rho_{i,t},1-\epsilon,1+\epsilon)\hat A_i\}.
\]

clipping 限制的是部分梯度方向，不是把整个模型的概率比强行锁在区间里。old logprob 应来自真实行为策略；把它用当前模型重新算一遍，会把 ratio 人为变成 1，掩盖实际偏移。工具返回不是策略采样的 token，RL loss 也应按正确的策略生成 mask 排除它。

### KL：对谁求，按什么分布估计

若当前策略为 \((0.8,0.2)\)，reference 为 \((0.5,0.5)\)，当前到 reference 的 KL 约为 0.192745。即使当前策略与 old 完全相同、所有 ratio 都等于 1，它仍可能已经偏离 reference。

固定历史 h 时，完整词表上的惩罚为

\[
K(h)=\sum_{a\in\mathcal V}\pi_\theta(a\mid h)
\log\frac{\pi_\theta(a\mid h)}{\pi_{\mathrm{ref}}(a\mid h)}.
\]

它需要沿词表聚合。常见的 sampled-token 非负估计写作

\[
\hat K=\exp(d)-d-1,\qquad
d=\log\pi_{\mathrm{ref}}(y\mid h)-\log\pi_\theta(y\mid h).
\]

仅当 y 从当前 \(\pi_\theta(\cdot\mid h)\) 采样时，这个量的期望等于上面的条件 KL。旧策略采样会改变测度；即使 on-policy，固定采样结果后直接对这个数值式反传，也不能仅凭“期望值正确”就宣布其梯度无偏。具体实现是否加入可微 importance ratio 等修正，需要一起读 trainer。当前 [TRL GRPO 文档](https://huggingface.co/docs/trl/grpo_trainer) 专门说明了 `use_bias_correction_kl` 的梯度作用。

用 \(\kappa\) 表示 KL 系数，以免与 KDA 的 beta 混淆，一个教学目标是最大化有效 token 上的 \(u_{i,t}-\kappa\hat K_{i,t}\)。reference KL、old/current ratio 和后面 OPD 的 teacher/student logratio 各有用途，不能交换分母。

### Sequence 平均与 token 平均会改变谁更有分量

两条轨迹分别有 2 和 8 个 token，逐 token surrogate 分别都为 +1 和 −1。先在各序列内平均、再平均两条，得到 0；把所有 token 合起来平均，得到 −0.6：

\[
J_{\mathrm{seq}}=\frac1G\sum_i\frac1{T_i}\sum_t u_{i,t},
\qquad
J_{\mathrm{tok}}=\frac{\sum_i\sum_t u_{i,t}}{\sum_iT_i}.
\]

前者让每条序列具有相同的总权重，短序列中的单个 token 权重更大；后者让每个有效 token 权重相同，长序列具有更多总权重。另一种做法使用固定长度分母。它们对应不同训练目标，不能只换一个名称却省略聚合规则。reward normalization、token/sequence importance weighting、loss 聚合是三个独立选择。[TRL loss types](https://huggingface.co/docs/trl/grpo_trainer#loss-types)

[grpo.py](../../src/k3lab/train/grpo.py) 配置 G、最大生成长度与工具迭代数，但没有自行实现上述 loss，也没有显式设定 `loss_type`、KL `beta`、`epsilon` 或 reward scaling。与依赖最低版本相对应的 [TRL v1.13.0 文档](https://huggingface.co/docs/trl/v1.13.0/grpo_trainer) 列出默认 `loss_type="dapo"`、`beta=0.0`；本仓库依赖是下界约束，并未锁定实际版本。本轮也未安装训练依赖，应在具体训练环境保存安装版本和展开后的 `GRPOConfig`，再解释实际优化目标。

## 3. K3 官方 RL 与这个 GRPO lab 怎样对应

先比较一个具体 token：ratio 为 1.4、优势为 −1。PPO 式 surrogate 在这个方向仍有梯度；一个按 logratio 区间直接屏蔽 policy-gradient 项的算法，可能把它屏蔽。这说明“都有 group reward 和 clip”不足以证明两个算法相同。

[K3 §4.1.2](https://arxiv.org/html/2607.24653v1#S4.SS1.SSS2) 写明沿用 K2.5 的策略优化，并使用逐 token 正则处理陈旧轨迹。[K2.5 §4.4.2](https://arxiv.org/html/2602.02276v1#S4.SS4.SSS2.Px1) 说明了按 logratio 范围进行梯度屏蔽、且不依赖优势符号的机制，还包含 current/old logratio 的平方正则。这条 Kimi 系列 RL 规则与 TRL GRPO 应分别解释。

对这种屏蔽的概念性理解，可以先定义 \(\ell=\log\pi_\theta-\log\pi_{\mathrm{old}}\)。奖励相关的 policy-gradient 项只在 \(\ell\) 落入指定区间时保留；正则项仍可以产生梯度。它与前节的 \(\min(\rho A,\operatorname{clip}(\rho)A)\) 在负优势等情形有不同的行为。

本 lab 选择 TRL GRPO，是为了给小模型提供可执行的环境奖励训练入口。不能把这份包装器的参数或教学 loss 反推为 K3 的完整算法配置。

## 4. MOPD：老师评价学生真正走到的分支

把一个历史 h 下的词表缩成两个 token。teacher 概率为 \(T=(0.8,0.2)\)，student 概率为 \(p=(0.5,0.5)\)。学生采到第一个 token 时，老师给出的 logratio 是 \(\log1.6\approx0.470004\)；采到第二个时是 \(\log0.4\approx-0.916291\)。这是每个已采样 token 的相对分布信号，既不需要老师写一条新轨迹，也不等于最终任务是否成功。

一个 batch 中已采样 token 的 student logprob、teacher logprob、reward 都可保存为 `[B,T]`；完整两模型 logits 则是 `[B,T,|V|]`。只 gather 已采样 token，可以避免保存整个词表的 dense distillation target，但 teacher 仍要对相应前缀求值，是否节省多少成本需要实际测量。

[K3 §4.1.3，Eq. (15)](https://arxiv.org/html/2607.24653v1#S4.SS1.SSS3) 按领域 d 和 effort e 选择对应 teacher，在 student 轨迹上使用

\[
r_t=\operatorname{clip}\left(
\operatorname{sg}\left[
\log T^{(d,e)}(y_t\mid x,y_{<t})
-\log p_\theta(y_t\mid e,x,y_{<t})
\right],-R_{\max},R_{\max}\right).
\]

其中 `sg` 表示这份 reward 在 policy update 中被视为固定信号。teacher 不更新，reward 中 student 分母也要 stop-gradient；当前策略用于 policy loss 的 logprob 则保留梯度。直接把同一个 `logp` 同时放进 reward 和乘在 reward 外面而不 detach，会多出一条通过“奖励计算公式”反传的路径。

例如在上面的两 token 例子中，把 `Rmax` 设为 0.5，会把第二个 reward 从 −0.916291 改为 −0.5；这已经改变了学习信号。报告还记录了更细粒度 top-k distillation 在其设置下未显示明确收益；这条观察应连同具体目标与资源条件保留，不能转述成“top-k 蒸馏普遍没有用”。

### Reverse KL 的等价式，先固定一个历史再推

对固定 h、固定 teacher 且不 clip 的情况，由学生自己采样，立即有

\[
\mathbb E_{a\sim p_\theta}[\log T(a)-\log p_\theta(a)]
=-D_{\mathrm{KL}}(p_\theta\Vert T).
\]

前面的例子中期望 reward 是 −0.223144。平均 reward 为负在这里完全正常：没有 clipping 时它对应负 KL，匹配 teacher 时才达到零。

对这份固定历史的目标求导，\(\mathbb E_p[\nabla\log p]=0\) 消掉了对 \(-\log p\) 直接求导产生的常数项，得到

\[
\nabla_\theta[-D_{\mathrm{KL}}(p_\theta\Vert T)]
=\mathbb E_{a\sim p_\theta}
[(\log T(a)-\log p_\theta(a))\nabla_\theta\log p_\theta(a)].
\]

这解释了 stop-gradient reward 配合 score-function 更新的关系。[Thinking Machines 的 OPD 原始说明](https://thinkingmachines.ai/blog/on-policy-distillation/)也从 student 采样和 reverse KL 讨论这一思路。

把这个等价式推广到完整 Agent 训练时，需要继续问：前缀由哪一版策略产生，奖励怎样做 return 或 credit assignment，是否 clipping，是否减去/缩放 baseline。逐 token 条件 KL 在采样前缀上的优化，不自动等于完整轨迹 KL 的无偏梯度；完整轨迹目标还涉及早期动作如何改变后续前缀分布。K3 公布的 reward 公式本身没有替我们补齐这些实现细节。

### Off-policy 校正能补回什么

如果实际 token 来自行为策略 \(\mu\)，在同一个固定历史 h 上，要估计 student 分布下的期望，需要

\[
\mathbb E_{a\sim p_\theta}[f(a)]
=\mathbb E_{a\sim\mu}\left[
\frac{p_\theta(a\mid h)}{\mu(a\mid h)}f(a)\right].
\]

前提是需要的动作具有行为分布支持。top-k/top-p 截断可能删去这部分支持；ratio clipping 或截断能降低方差，但会引入偏差。并且这里修正的只是给定 h 的下一 token 测度：若 h 本身也由旧策略产生，还存在 prefix/state occupancy 的变化。全轨迹校正涉及沿路径的概率比乘积，长轨迹上通常方差很大。

因此，看到一项 token importance ratio，不能就宣布所有旧轨迹都等同于当前策略新采样。应保存行为策略版本、采样设置、每个策略 token 的 behavior logprob，以及 reward 是在哪一版 student/teacher 下计算的。[现有 schema](../../src/k3lab/schema.py) 保存交互记录，但没有提供完整的 OPD 训练记录或上述逐 token 概率链。

## 5. Partial rollout：环境继续活着，策略已经换了一版

设两道题各采两条轨迹。第一组长度为 2、3，第二组为 20、100。完成比例达到一半时，第一组已可形成完整 group；第二组暂停。参数更新后，再恢复第二组，它的前缀和新后缀就可能来自不同策略版本。

[K3 §4.1.2](https://arxiv.org/html/2607.24653v1#S4.SS1.SSS2) 描述了按完成比例暂停生成、优先恢复未完成轨迹，以及某 prompt 的整组回答完成后再送入优化。它利用 sandbox 保留环境状态，逐 token 正则处理由此引入的 staleness。

这里至少有三种时间要分清：环境动作执行顺序、轨迹 token 的行为策略版本、optimizer update 版本。设第 t 个 token 由版本 \(v_t\) 产生，则 ratio 分母应对应

\[
\mu_t(y_t\mid h_t)=\pi_{v_t}(y_t\mid h_t),
\]

而不必是整条轨迹一个统一 old policy。把尚未完成的轨迹当作 reward=0，也会人为偏向短任务；暂停不是失败标签。

排查时先画时间线：哪些 group 完成、哪些样本参与更新、恢复后采用何种权重、旧 prefix 的 KV/循环状态是否与这份策略语义一致。环境 snapshot 保证文件与工具状态延续，不能自动证明旧模型缓存等价于新权重重新计算的前缀。

[MiniRepoGRPOEnv](../../src/k3lab/train/grpo_env.py) 每次 reset 创建独立的小环境，并没有实现跨 optimizer iteration 的暂停/恢复队列、版本化 KV 或混合策略 logprob。`--use-vllm` 只是把生成后端选择交给 trainer，不等于本仓库已经实现 K3 的 partial rollout 系统。

## 6. QAT：同一组参数，两个前向也可能给出不同概率

先用均匀量化做手算。权重 \(w=(0.7,0.3)\)，输入 \(x=(1,-1)\)，浮点点积为 0.4。若以步长 0.5 取整，量化权重为 \((0.5,0.5)\)，点积变为 0。把它视为二分类 logits 的第一项，概率分别约为 0.598688 和 0.5。

同一 token 在 rollout 端按量化模型采样、在 training 端按高精度模型求值，即使权重“没有更新”，ratio 也可能不是 1。这个算例隔离了量化前向差异；MXFP4 还有自己的块级缩放与数值编码，不能用上述均匀格点模拟全部行为。

对矩阵 \(W\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}\)，一个概念性 fake-quant 前向是

\[
Q_\Delta(W)=\Delta\operatorname{round}(W/\Delta),
\qquad y=Q_\Delta(W)x.
\]

round 的普通导数几乎处处为零，训练通常需要 straight-through 一类的梯度近似；例如在选定范围内把 \(\partial Q/\partial W\) 近似为 1。这个近似怎样处理饱和、scale 与精度，是具体 QAT 实现的一部分。

[K3 §4.1.4](https://arxiv.org/html/2607.24653v1#S4.SS1.SSS4) 披露了覆盖 SFT/RL 的 QAT，路由 MoE 专家权重用 MXFP4、输入激活用 MXFP8，注意力、潜在投影、共享专家和 router 等保持较高精度；rollout 与 training 使用同一量化方案。由此核算部署内存时，也不能把整模所有模块都简单乘 4 bit。

本仓库使用 PEFT LoRA 不等于做了这种 QAT，`--bf16` 也只是 trainer 的精度配置。包装器没有 MXFP4 fake-quant 或量化部署对齐实现。若要开展相关实验，应先固定量化前向，在同一输入上比 logits、chosen-token logprob 和采样设置，再比较训练与生成端的差异来源。

## 7. Speculation：接受草稿之后为什么仍然服从 target

target 分布为 \(p=(0.6,0.3,0.1)\)，draft 为 \(q=(0.4,0.5,0.1)\)。draft 采到三个 token 时，接受概率依次为 \((1,0.6,1)\)。总接受质量为

\[
\min(p,q)=(0.4,0.3,0.1),\qquad a=0.8.
\]

剩下 0.2 的拒绝质量来自第二个 token；拒绝后按归一化的正差 \([p-q]_+=(0.2,0,0)\) 采样，于是补回第一个 token。最终分布恢复为 \((0.6,0.3,0.1)\)。如果拒绝后直接从 p 重新抽，补偿质量就会放错位置。

固定上下文时，\(p,q\in\mathbb R^{|\mathcal V|}\)，标准无损 speculative sampling 的单 token 规则是

\[
y\sim q,\qquad
P(\mathrm{accept}\mid y)=\min\{1,p(y)/q(y)\},
\]

\[
p_{\mathrm{residual}}(y)=
\frac{[p(y)-q(y)]_+}{\sum_a[p(a)-q(a)]_+}.
\]

当两分布完全相同，拒绝概率为零，不需要构造分母为零的 residual。分布保持的证明把两部分质量相加即可：\(\min(p,q)+[p-q]_+=p\)。多 token 草稿逐步在已接受前缀上应用同样的条件分布关系。[原始算法 §2.3、附录 A.1](https://arxiv.org/html/2211.17192v2#S2.SS3)

另由 \(\min(p,q)=(p+q-|p-q|)/2\) 可得

\[
a=\sum_y\min(p(y),q(y))=1-\tfrac12\|p-q\|_1.
\]

接受率直接对应两分布的重叠质量；capacity 有限的 draft 去最小化 KL，未必就能最大化这个量。K3 的 draft 微调采用 [§4.1.4 Eq. (16)](https://arxiv.org/html/2607.24653v1#S4.SS1.SSS4) 的 \(L_{\mathrm{LK}}=-\log a\)，在温度 1 下计算，未加 ground-truth CE。本例的 LK loss 为 0.223144。

这里的 p 应当是你最终希望保留的实际 target 采样分布，包括温度和其他采样变换；不能一边校验 temperature=1 的概率，一边宣称保持了另一份采样分布。接受率还要与 draft 耗时、target 批量校验、缓存和调度成本一起解释，不能直接写成吞吐加速倍数。

本 lab 没有 draft 模型、接受/拒绝 sampler 或 LK 训练器；上述小词表例子可以在 CPU 检查概率守恒，不能代替完整模型的部署测量。

## 8. 回到仓库：一次奖励究竟来自哪里

以 [repo-abs-002](../../experiments/tasks/repo_patch.jsonl) 为例，初始实现 `x + 3` 在公开的非负输入上全对，但在负数 held-out 输入上失败。这个具体失败说明，公开测试通过可能只反映测试覆盖范围，而非实现满足 `abs(x)+3` 的规格。

[MiniRepoEnv.verify](../../src/k3lab/envs/minirepo.py) 重新跑创建环境时保留的 public tests、环境侧 hidden tests，并检查 protected files 的内容完整性。[GRPO env](../../src/k3lab/train/grpo_env.py) 返回这三项共同决定的二元 success。policy 可以调用可修改的公开测试；删空公开测试会制造表面的 PASS，却无法通过完整 verifier。

[eval.py](../../src/k3lab/eval.py) 在相同 success 上另外调用 [CompositeReward](../../src/k3lab/rewards/core.py)，默认再扣非法调用和工具次数。**评测输出的 composite reward 与当前 GRPO 训练的环境 reward 不相同。**比较时分别报告 secure success、调用成本和奖励定义，先问目标有没有变，再谈奖励是否上涨。

| 当前代码 | 已表达的机制 | 还需按具体运行核查的部分 |
|---|---|---|
| `export_sft.py` | 按 verifier success 筛选并重建消息 | 是否为完整结束轨迹；输入记录是否可信；实际 tokenizer 是否需要额外 tools schema |
| `train/sft.py` | LoRA、assistant mask/packing 开关 | labels、结束标记、截断、packing 边界；默认开关与实际 config |
| `train/grpo.py` | TRL GRPO 与每条 rollout 的独立环境 | 安装版本、有效 group/batch、loss 聚合、KL、生成 token mask、工具模板 |
| `train/grpo_env.py` | 原始/hidden 测试加完整性决定 0/1 reward | 每次 reset 数据是否隔离；全部失败 group 如何产生学习信号 |
| `harness/agent.py` | assistant/tool 历史与逐步记录 | final text 出现不等于环境正确；max_steps 后记录也可能缺最终回答 |
| `eval.py` | 任务 hash、harness 配置与环境评分 | 训练/验证任务隔离；held-out harness 没有混入选参；实际采样设置是否完整留存 |

现有真实 CPU 结果在 [机制实验报告](../../reports/mechanisms/REPORT.md)，涉及 KDA/AttnRes/MoE 数值检查；Agent 训练增益需要另行采样、训练与封存评价。

训练前最有价值的下一步，是取一条成功轨迹检查真实 labels，再用极小 group 跑一次可追踪更新：保存生成版本与 behavior logprob、有效 token、环境奖励、advantage、loss 分项和参数变化。这样才能把“数据格式正确”“trainer 能更新”“held-out success 提升”接成三项分别可检查的结果。

上一讲：[预训练与系统](04-pretraining-systems.md) · 下一讲：[十分钟讲述与练习](06-interview-walkthrough.md) · 原有入口：[后训练主线](../post-training.md)、[训练命令](../training.md)
