# KDA：从关联记忆到下三角联立求解

KDA 把历史写入一个固定大小的矩阵；每个新 token 先让旧记忆按通道衰减，再根据“旧记忆已经能预测多少”修正写入量。读懂它的关键，是沿着同一个状态矩阵，把单步学习、序列递推和 chunk 并行连起来。

本页以 [Kimi Linear §3](https://arxiv.org/html/2510.26692v1) 的 KDA 状态式为起点，给出本仓库独立推导的单头 NumPy 参考；K3 采用的门控变化另在文末说明，核对 [Kimi K3 §2.1.1](https://arxiv.org/html/2607.24653v1)。参考实现是 [kda.py](../../src/k3lab/mechanisms/kda.py)，检查在 [test_mechanisms_kda.py](../../tests/test_mechanisms_kda.py)。

## 1. 先手算两次写入

让记忆矩阵有两个 key 方向，每个方向存两维 value：

\[
S_0=\begin{bmatrix}0&0\\0&0\end{bmatrix}.
\]

第一条信息的 key 是 \(k_1=(1,0)^\top\)，value 是 \(v_1=(2,1)^\top\)。保留率为 \(\alpha_1=(0.5,0.25)^\top\)，写入强度为 \(\beta_1=0.5\)。旧状态为零，所以衰减后依然为零；沿这个 key 读出的预测也是 \((0,0)^\top\)。只写一半残差：

\[
S_1=0+k_1\{0.5[v_1-S_0^\top k_1]\}^\top
=\begin{bmatrix}1&0.5\\0&0\end{bmatrix}.
\]

若立即用 \(q_1=k_1\) 查询，读出 \(o_1=S_1^\top q_1=(1,0.5)^\top\)。当前 token 已经写入，读出的不是更新前状态。

第二条信息使用另一个方向 \(k_2=q_2=(0,1)^\top\)，value 是 \(v_2=(4,-2)^\top\)，\(\alpha_2=(0.5,1)^\top\)，\(\beta_2=1\)。先让第一行减半，第二行保持：

\[
\bar S_2=\operatorname{Diag}(\alpha_2)S_1
=\begin{bmatrix}0.5&0.25\\0&0\end{bmatrix}.
\]

第二行原先没有信息，预测为零，于是

\[
S_2=\bar S_2+k_2v_2^\top
=\begin{bmatrix}0.5&0.25\\4&-2\end{bmatrix},
\qquad o_2=(4,-2)^\top.
\]

这两步分别呈现了三个控制量：\(\alpha\) 决定旧状态每一行保留多少，\(\beta\) 决定本次修正写多少，\(q\) 决定从更新后的记忆读哪个方向。后面即使反复写同一个 key，也只需补上预测误差。

## 2. 固定维度，再解释每个乘法

| 对象 | 单 token 形状 | 含义 |
|---|---|---|
| \(q_t,k_t\) | \(K\) | 查询方向、写入与预测方向 |
| \(v_t,o_t\) | \(V\) | 要记住的值、读出的值 |
| \(S_t\) | \(K\times V\) | key 行、value 列的关联记忆 |
| \(\alpha_t\) | \(K\) | 每个 key 通道的保留率 |
| \(\beta_t\) | 标量 | 当前 token 的 delta 写入强度 |

用 \(D_t=\operatorname{Diag}(\alpha_t)\in\mathbb R^{K\times K}\) 表示门控矩阵。实际更新可以按四步执行：

\[
\begin{aligned}
\bar S_t&=D_tS_{t-1} &&\text{逐 key 行衰减},\\
\hat v_t&=\bar S_t^\top k_t &&\text{预测当前 value},\\
e_t&=v_t-\hat v_t &&\text{得到待修正残差},\\
S_t&=\bar S_t+\beta_t k_te_t^\top,\qquad
o_t=S_t^\top q_t &&\text{写入后读出}.
\end{aligned}
\]

\(k_te_t^\top\) 是 \(K\times V\) 的外积。把残差展开就得到常见状态式：

\[
S_t=(I-\beta_tk_tk_t^\top)D_tS_{t-1}
+\beta_tk_tv_t^\top.
\]

代码中数组按 token 排行，因此 `k[t] @ state` 返回 \(V\) 维预测，与列向量写法 \(S^\top k_t\) 相同。`alpha[t, :, None] * state` 恰好乘在 \(K\) 这一轴。使用 \(V\times K\) 状态的其他实现需要整体转置公式；仅把两个维度设成相同会掩盖轴错误。

## 3. Delta rule 为什么是一次梯度下降

把矩阵 \(S\) 看作在线学习的线性映射：输入 key，预测 value。当前 token 对应的平方损失为

\[
\ell_t(S)=\frac12\|S^\top k_t-v_t\|_2^2.
\]

对 \(S\) 求导：

\[
\nabla_S\ell_t(S)=k_t(S^\top k_t-v_t)^\top.
\]

以已经衰减的 \(\bar S_t\) 为起点，做学习率为 \(\beta_t\) 的一步梯度下降：

\[
\bar S_t-\beta_t\nabla_S\ell_t(\bar S_t)
=\bar S_t+\beta_t k_t(v_t-\bar S_t^\top k_t)^\top.
\]

这正是上面的写入规则。若直接累加 \(k_tv_t^\top\)，同一条信息重复出现会反复堆高状态；delta 写入则在预测已经正确时把本次更新降到零。

当 \(\|k_t\|_2=1\) 时，本次写入后沿同一个 key 回读，有

\[
S_t^\top k_t=(1-\beta_t)\hat v_t+\beta_tv_t.
\]

所以 \(\beta=0\) 表示“保留衰减后的预测”，\(\beta=1\) 表示“把当前 key 方向改写成目标 value”。它不清空整个矩阵；与 key 正交的分量仍保持衰减后的值。若禁用归一化，右式会出现 \(\|k_t\|_2^2\)，\(\beta=1\) 便不再保证精确改写。

还有一个有用的稳定性观察：单位 key 下，\(I-\beta kk^\top\) 在 key 方向的特征值是 \(1-\beta\)，在其余方向是 1。若 \(\alpha,\beta\in[0,1]\)，齐次状态转移的谱范数至多为 1。这个结论只控制旧状态的传播；不断写入的新 value 仍会影响总状态大小。

## 4. 门控次序为什么不能随便换

正确顺序是先计算 \(D_tS_{t-1}\)，再根据这份衰减后的记忆预测和纠错。因此转移是

\[
(I-\beta kk^\top)D,
\]

而把门控放到纠错后，会得到 \(D(I-\beta kk^\top)\)。两者一般不同：逐通道门控 \(D\) 与 key 的秩一投影 \(kk^\top\) 不交换。只有门控退化为同一个标量，或 key 与门控具有特殊对齐关系时，次序才可能相同。

另一处容易错的是给新写入的 \(kv^\top\) 也乘当前 \(D_t\)。当前 token 的 value 在本步衰减之后才写入，应该从下一步开始被衰减。这也决定了下面 chunk 公式的乘积起点。

## 5. 把一个 chunk 的 token 联立起来

单步递推的依赖在于“这一步的预测要读取前一步状态”。换一个未知量，就能把这个依赖改写成下三角系统。

设一个 chunk 有 \(C\) 个 token，入口状态是 \(S_{\mathrm{in}}\)。定义第 \(i\) 次写入的创新量

\[
z_i=\beta_i(v_i-\bar S_i^\top k_i)\in\mathbb R^V.
\]

于是状态只需写成 \(S_i=D_iS_{i-1}+k_iz_i^\top\)。再定义两个逐通道乘积：

\[
\gamma_i=\prod_{r=1}^{i}\alpha_r\in\mathbb R^K,
\qquad
\delta_{i,j}=\prod_{r=j+1}^{i}\alpha_r\in\mathbb R^K
\quad(j\le i).
\]

空乘积 \(\delta_{i,i}\) 为全 1 向量。它表示“第 \(j\) 步写进去的东西，到第 \(i\) 步还保留多少”；第 \(j\) 步自己的门控不作用于本步新写入。

展开到第 \(i\) 次写入前的状态：

\[
\bar S_i=\operatorname{Diag}(\gamma_i)S_{\mathrm{in}}
+\sum_{j<i}\operatorname{Diag}(\delta_{i,j})k_jz_j^\top.
\]

代回创新量定义，并把依赖前面创新量的项移到左边：

\[
z_i+\sum_{j<i}\underbrace{\beta_i k_i^\top
\operatorname{Diag}(\delta_{i,j})k_j}_{L_{ij}}z_j
=\beta_i\{v_i-S_{\mathrm{in}}^\top
\operatorname{Diag}(\gamma_i)k_i\}.
\]

把 \(z_i^\top\) 按行堆成 \(Z\in\mathbb R^{C\times V}\)，把 \(k_i^\top\)、\(v_i^\top\)、\(\gamma_i^\top\) 分别堆成 \(K_c,V_c,\Gamma\)。这里的 \(K_c\) 是 chunk key 矩阵，不是 key 维数。整个 chunk 满足

\[
\boxed{(I_C+L)Z=
\operatorname{Diag}(\beta)\{V_c-(\Gamma\odot K_c)S_{\mathrm{in}}\}.}
\]

\(L\) 严格下三角，\(I_C+L\) 是对角线全 1 的可逆下三角矩阵。实现实际调用 `np.linalg.solve(system, rhs)` 求这组联立方程；没有在 chunk 内偷偷调用逐 token 状态递推，也没有构造矩阵逆。

### 求出创新量后，怎样一次读完整个 chunk

输出相关系数为

\[
A_{ij}=q_i^\top\operatorname{Diag}(\delta_{i,j})k_j
\quad(j\le i),\qquad A_{ij}=0\quad(j>i).
\]

因此

\[
O=(\Gamma\odot Q_c)S_{\mathrm{in}}+AZ,
\qquad O\in\mathbb R^{C\times V}.
\]

前一项读取前面 chunks 留下的状态，后一项读取当前 chunk 的创新。这里的 \(A\) 必须保留对角线，因为第 \(i\) 个输出已经可以读取第 \(i\) 次写入。构造 \(L\) 时则必须去掉对角线，因为创新量只依赖之前的写入。这两个三角掩码差一条对角线，意义完全不同。

最后把状态传给下一个 chunk。设 \(K_{\mathrm{tail}}\) 的第 \(j\) 行是 \((\delta_{C,j}\odot k_j)^\top\)，则

\[
S_{\mathrm{out}}=\operatorname{Diag}(\gamma_C)S_{\mathrm{in}}
+K_{\mathrm{tail}}^\top Z
\in\mathbb R^{K\times V}.
\]

跨 chunk 仍然递推，只是递推次数从 token 数降为 chunk 数。改变 chunk 大小或把输入分成多次调用，只要传递同一个状态，就应得到一致的输出。

## 6. 这和论文中的 WY / UT 是什么关系

WY 的思路，是把一连串秩一状态修改压成低秩因子的组合，避免逐 token 搬运整个状态。UT 则把因子之间的因果依赖收进一个下三角系统。可以直接用上面的系统看出对应关系。

令 \(B=\operatorname{Diag}(\beta)\)，\(R=\Gamma\odot K_c\)，在数学记号里写

\[
M=(I_C+L)^{-1}B,\qquad U=MV_c,\qquad W=MR.
\]

那么创新量就是

\[
Z=U-WS_{\mathrm{in}}.
\]

它对应论文使用的 pseudo-value 项。分开求 \(U,W\)，可以把 chunk 自身的变换与入口状态分开；本参考直接为 \(Z\) 求解一次联立系统，便于从残差定义理解。上式的逆号只用于表达等价关系，代码使用 solve。

当所有 \(\alpha>0\) 时，\(\delta_{i,j}=\gamma_i/\gamma_j\)。这就能把相关系数写成诸如 \((\Gamma\odot Q_c)(K_c/\Gamma)^\top\) 的矩阵乘积。直接除以很小的累计保留率会放大数值范围；本参考逐区间累乘 \(\delta\)，省去这个除法，也允许用零门控做干预。其代价是显式存储 \(C\times C\times K\) 的中间量。

## 7. NumPy 参考与 GPU 内核各自解决什么

逐 token 版本每步进行若干 \(K\times V\) 乘加，时间随序列长度 \(T\) 为 \(O(TKV)\)，循环状态占 \(O(KV)\)。小矩阵反复更新在 GPU 上容易受调度、数据搬运与低并行度约束。chunk 化把大量计算变成矩阵乘法，再让 chunk 间传递状态。

本参考显式构造相对衰减张量，并调用 NumPy 的通用线性求解器：它适合检查公式和小尺寸数值，不是 GPU 内核的实现路线。通用 solve 不利用下三角结构取得专用求解的计算量；因此这里的 CPU wall time 不能拿来证明 FlashKDA 的加速。实际 [官方 FlashKDA](https://github.com/MoonshotAI/FlashKDA) 采用 CUTLASS，并与 FLA 的 chunk 接口集成。

工程上需要同时考虑：chunk 太小会让矩阵乘法难以充分利用硬件，太大会扩大块内相关矩阵与门控数值范围；低精度需要控制缩放中间量；多条序列的边界必须隔开；训练反向要保存或重算合适的中间量。单头数学等价，并不会自动解决这些资源与调度问题。

KDA 单层的循环状态大小与历史长度无关；包含全注意力层的混合模型，仍需要为那些层保存随长度增长的缓存。应按层类型分别计算内存。

## 8. K3 具体改了哪里

[K3 §2.1.1](https://arxiv.org/html/2607.24653v1) 继续使用上述状态递推，并明确 q/k 做 L2 归一化。它把 log-decay 改为有下界的映射，并把输出门改为 full-rank 投影；本节对应的是报告中的门参数化变化。

\[
g=g_{\min}\,\sigma(e^A z),\qquad
\alpha=e^g,\qquad g_{\min}=-5.
\]

由此可直接算出 \(g\in(-5,0)\)、\(\alpha\in(e^{-5},1)\)。对 16 个 token 相乘，累计 log-decay 大于 \(-80\)，其倒数缩放小于 \(e^{80}\)。这个范围约束让内核更容易安排低精度矩阵乘法。代码中的 `k3_retention` 只实现这个映射；实际浮点 sigmoid 饱和时可以达到区间端点。

Kimi Linear 使用的是另一种 log-decay 映射：\(-e^A\operatorname{Softplus}(z)\)，没有相同的固定下界。[原论文 §4](https://arxiv.org/html/2510.26692v1) 和 [官方仓库](https://github.com/MoonshotAI/Kimi-Linear) 的模型与速度实验属于 Kimi Linear；本页没有把这些结果挪作 K3 的实测，也没有运行完整模型。

## 9. 本参考的接口与选择

本节把 A-003 的数值约定，以及补充选择 A-007 固定下来，便于复查每个实验究竟计算了什么：

| 项目 | 本仓库约定 |
|---|---|
| 输入 | `q/k[T,K]`、`v[T,V]`、`alpha[T,K]`、`beta[T]`，禁止隐式广播门控维度 |
| 状态 | `initial_state[K,V]`，默认全零；返回最后状态；不修改调用方数组 |
| 归一化 | 默认逐 token 把非零 q/k 归一化为单位向量，零向量保留为零；`normalize_qk=False` 接收调用方定义的原始特征 |
| 精度 | 所有状态、输出与中间量使用 NumPy `float64` |
| 读出缩放 | 默认 `scale=1.0` 对齐展示的状态式；只缩放 q 的读出，不改变状态 |
| 门控端点 | 为 beta0/1 与清空干预接收 \([0,1]\)；实际 sigmoid 产生的开放区间是其中子集 |
| chunk | 默认 64，尾块允许不足 64；相对衰减直接累乘，支持 alpha 为零；跨 chunk 延续状态 |
| 一致性标准 | 随机非方阵、非零初态等检查使用 `rtol=atol=2e-12`，精确手算另外逐项比较 |

[FLA 的公开参考](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/kda/naive.py) 接收 log-decay `g`，默认还乘 \(K^{-1/2}\) 的 query scale；与本接口对照时，要用 `alpha=exp(g)` 并显式设置相同 scale。FLA 参考函数本身接收已准备的特征，本接口把可选 q/k 归一化放在入口。两者的默认参数不同，直接把同一组原始数组交进去比较，会混入预处理差异。

这里的输入已经是单头投影特征。q/k/v 神经投影、短卷积、Swish、输出 RMSNorm、输出门和多头拼接未包含在状态核里。`k3_retention` 也不生成那些投影；这样每个干预都能只改变一个机制。

```python
import numpy as np
from k3lab.mechanisms.kda import kda_chunkwise, kda_recurrent

rng = np.random.default_rng(7)
q, k = rng.normal(size=(2, 13, 3))
v = rng.normal(size=(13, 5))
alpha = rng.uniform(0.2, 1.0, size=(13, 3))
beta = rng.uniform(0.0, 1.0, size=13)
state0 = rng.normal(size=(3, 5))

recurrent, sr = kda_recurrent(q, k, v, alpha, beta, initial_state=state0)
chunkwise, sc = kda_chunkwise(q, k, v, alpha, beta, initial_state=state0, chunk_size=4)
np.testing.assert_allclose(chunkwise, recurrent, rtol=2e-12, atol=2e-12)
np.testing.assert_allclose(sc, sr, rtol=2e-12, atol=2e-12)
```

验证命令：

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_mechanisms_kda.py -q
```

测试额外用稠密仿射转移的显式乘积展开作为独立参考；也扰动未来 token 检查因果性，检查 beta0/1 的已知行为、门控次序、当前写入的可读性、跨调用续接、极小/零门控与非方阵状态。它们分别约束不同的机制错误，而不只比较两个函数是否恰好一致。

## 10. 面试里值得追问的五件事

**为什么 fixed-size state 仍会遗忘？** 多个 key 方向共用有限的 \(K\times V\) 状态；相关 key 的写入会相互影响，\(\alpha\) 又会主动衰减。内存固定给出的是存储上界，记忆质量还要由特征、门控和任务决定。

**为什么 beta=0 时状态还在变化？** 因为 beta 控制纠错写入，alpha 控制旧状态衰减。beta0 的正确干预是纯衰减；要保持整个状态不变，还需要所有 alpha=1。

**为什么 chunk 能并行，却仍然是因果的？** 当前 chunk 的未知创新量满足下三角联立方程，第 i 行只依赖 j<i；读出矩阵只保留 j≤i。并行改变的是计算组织，允许使用的信息范围由三角结构限定。

**UT 为什么比直接展开所有状态更有利？** 它把许多次秩一更新中的依赖集中到小的三角系统，剩下大量工作成为矩阵乘法；不需要为每个 token 显式保存一个完整状态矩阵。真实加速幅度仍取决于块大小、精度和内核实现。

**如果要求证明自己的实现可信，先展示什么？** 先用独立的稠密仿射展开核对非方阵和非零初态，再展示 beta0/1、未来扰动、错误门控次序的反例和跨 chunk 续接。之后再讨论更大模型与硬件性能，问题和证据就能一一对应。

公开资料核对日期：2026-10-21。本文推导、例子和 NumPy 代码为本仓库独立编写。
