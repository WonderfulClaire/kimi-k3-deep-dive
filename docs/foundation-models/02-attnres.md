# AttnRes：把深度方向的信息流写成一次可检查的检索

AttnRes 让当前子层为历史表示分配不同权重。它的注意力轴是网络深度，每个 token 只混合自己在不同深度的表示；序列中的 token 交互仍由 KDA、MLA 等子层完成。K3 的公开定义见 [§2.2](https://arxiv.org/html/2607.24653v1#S2.SS2)，独立的 [Attention Residuals 官方仓库](https://github.com/MoonshotAI/Attention-Residuals#overview)给出了 Block 版本伪代码。

## 先手算一次：同一个 token 的三份历史表示

假设当前有三个来源，特征维度都是 2：

\[
v_0=(1,0),\quad v_1=(0,2),\quad v_2=(-1,0).
\]

这三个向量可以分别来自 embedding、较浅子层输出和较深子层输出。当前层持有可学习向量 \(q=(1,0)\)，称为 pseudo-query。为了方便手算，这个例子把 RMSNorm 的 \(\epsilon\) 设为 0，默认增益设为 1。

第一步，对每个来源单独按特征维归一化：

\[
k_0=(\sqrt2,0),\quad k_1=(0,\sqrt2),\quad k_2=(-\sqrt2,0).
\]

第二步，用同一个 query 分别点乘三个 key，得到 logits：

\[
s=(\sqrt2,0,-\sqrt2).
\]

第三步，在三个来源之间做 softmax，得到约 \((0.767918,0.186694,0.045388)\)。最后用这些权重加权**原始** value：

\[
h=0.767918(1,0)+0.186694(0,2)+0.045388(-1,0)
\approx(0.722530,0.373387).
\]

第二个来源虽然在归一化 key 中变为 \((0,\sqrt2)\)，参与最终混合的仍是 \((0,2)\)。因此，key 的幅度控制和 value 的信息传递是两件独立的操作。

可运行对应的最小例子：

```python
import numpy as np
from k3lab.mechanisms.attnres import depth_attention

values = np.array([[1., 0.], [0., 2.], [-1., 0.]])
mixed, weights = depth_attention(values, np.array([1., 0.]), eps=0)
print(weights)
print(mixed)
```

## 从算例到完整的维度约定

对当前深度 \(l\)，把 embedding 和此前的原始子层输出记为 \(v_i\)。这里的输出是 \(f_i(h_i)\)，并非把全部历史已经加在一起的 residual stream。若后者被再次放进来源列表，浅层信息会被重复计数。

| 对象 | 本实现的形状 | 含义 |
| :--- | :--- | :--- |
| `sources` / \(V\) | `[S, B, T, D]` | S 个深度来源，每个来源有 B 个样本、T 个 token |
| `pseudo_query` / \(q_l\) | `[D]` | 当前子层的可学习向量，沿样本和 token 位置共享 |
| `keys` / \(K\) | `[S, B, T, D]` | 对每个来源、样本、token 独立做特征 RMSNorm |
| `logits` / \(s\) | `[S, B, T]` | 每个 token 对各个深度来源的分数 |
| `weights` / \(\alpha\) | `[S, B, T]` | 沿 S 归一化，固定 B、T 后权重和为 1 |
| `mixed` / \(h\) | `[B, T, D]` | 供当前计算子层继续使用的输入 |

代码也接受省略 B 或 T 的情形，只要求第 0 维是来源，最后一维是特征。

要计算一个来源的检索分数，先定义：

\[
r(v)=\sqrt{\frac1D\sum_{a=1}^{D}v_a^2+\epsilon},\qquad
k(v)=v/r(v),\qquad s_i=q_l^\top k(v_i).
\]

然后计算：

\[
\alpha_i=\frac{\exp(s_i-\max_j s_j)}{\sum_j\exp(s_j-\max_r s_r)},
\qquad h=\sum_i\alpha_i v_i.
\]

减去最大值只改善数值稳定性，因为对所有来源加同一个常数不改变 softmax。公开定义的点积没有额外的 \(1/\sqrt D\) 缩放，本参考也遵循这一约定；默认 \(\epsilon=10^{-8}\)，统一用 float64。RMSNorm 的通道增益固定为 1；若需要同一归一化器的可学习增益，可以将它逐通道吸收到 query 中。[公开定义 Eq. (8–9)](https://arxiv.org/html/2607.24653v1#S2.SS2)

query 在输入之间共享，权重仍然依赖输入：token 不同，历史 key 不同，点积也不同。把它理解为“每层一个固定的检索方向”比“每个 token 都生成一个新 query”更准确。

## Full 和 Block 的差别发生在来源集合

Full 为每个历史子层输出保留独立的检索位置。Block 把连续子层输出先相加，同一个块里的细节被合并为一个来源；当前尚未完成的块使用已有输出的 partial sum。embedding 单独保留，空 partial 不应作为一份零向量加入 softmax。K3 的块聚合定义见 [§2.2 的 Eq. (10)](https://arxiv.org/html/2607.24653v1#S2.SS2)，其模型设置包含 8 个块以及单独的 embedding 来源。

用零 query 观察这一差别最直观。设 embedding 为 \((0,0)\)，四个固定子层输出为：

\[
(2,0),\ (2,0),\ (0,6),\ (0,6).
\]

| 形式 | 实际参与 softmax 的来源 | 零 query 时的输出 |
| :--- | :--- | :--- |
| Full | embedding 和四个原始输出，共 5 项 | \((0.8,2.4)\) |
| Block，块大小 2 | embedding、\((4,0)\)、\((0,12)\)，共 3 项 | \((4/3,4)\) |

即使所有 logits 都相同，两者也会不同，因为 Block 在块内做加法，在块间做平均。Block 不能通过只改变温度、屏蔽部分 Full 权重或改一个标签来实现。块大小为 1 时，本聚合参考才与 Full 完全一致。

接着多加一个输出 \((8,9)\)，块大小仍为 2。来源列表变为 embedding、前两项的和、中间两项的和、最后一项。最后一项就是当前 partial；当输出数恰好是 4 时，列表只有前三项，额外零 partial 会错误地分走注意力概率。

```python
import numpy as np
from k3lab.mechanisms.attnres import full_attnres, block_attnres, block_sources

embedding = np.zeros(2)
outputs = np.array([[2., 0.], [2., 0.], [0., 6.], [0., 6.]])
print(full_attnres(embedding, outputs, np.zeros(2)))
print(block_attnres(embedding, outputs, np.zeros(2), block_size=2))
print(block_sources(embedding, outputs, block_size=2))
```

本仓库选择的是**固定子层输出的聚合参考**：先给出 `layer_outputs`，再比较如何混合这些值。在真正的 Transformer 中，混合结果会进入下一次 attention/MLP，从而改变之后的子层输出；上面的固定值算例用于隔离聚合行为，没有模拟这个闭环。接口的 `block_size` 以一次 \(f_i\) 输出为单位，attention 和 MLP 输出各占一个位置。官方不同实现中的“层”与“块大小”要先按它们自己的计数单位换算。

## 为什么归一化 key，而保留原始 value

令 \(v_0=(1,0),v_1=(10,0)\)，仍取 \(q=(1,0)\)。在 \(\epsilon=0\) 时，两个 key 完全相同，权重都为 \(1/2\)，输出为 \((5.5,0)\)。大幅值来源不会仅凭放大 10 倍就获得更高检索分数，但它被选中后的幅度仍会传到下游。

因此，“权重归一化”不等于“最终状态具有固定范数”。输出落在给定 value 集合的凸包内，其范数可由来源的最大范数控制；来源本身若持续增长，这个性质无法自动保证全网络训练稳定。零 query 同样产生的是历史值的均值，传统残差则是逐项求和，初始化时需要留意这一尺度差别。

从梯度看，给定一个来源 \(v_j\)，它影响输出的两条路径分别是 value 和 key：

\[
\frac{\partial h}{\partial v_j}
=\alpha_j I+\alpha_j(v_j-h)\bigl(\nabla_{v_j}s_j\bigr)^\top.
\]

第一项是直接传递被选中的信息；第二项反映改变该来源还会改变检索权重。对于单位增益 RMSNorm，\(\nabla_v s=q/r-v(v^\top q)/(D r^3)\)。这个式子说明它提供了可学习的深度路径，具体优化收益仍取决于训练后的 query、来源表示及损失。

## 计算、存储和工程实现要分开看

若有 L 个历史输出、每个来源占 BTD 个元素，Full 最后一层要读取约 \((L+1)BTD\) 个元素，整个深度的混合计算量约为 \(O(L^2BTD)\)。Block 大小为 S，块数 \(N=\lceil L/S\rceil\)，最终保留 \(N+1\) 个来源，整个深度的检索算术约为 \(O(LNBTD)\)，外加块内相加。

这里统计的是为跨深度检索保留的来源，不包括每个子层反向传播需要的其他激活，也不直接等于训练峰值显存。`attnres_storage_counts` 输出上述来源元素数和字节数；当前 Python 参考仍接收完整输出历史，且临时构造 key 和堆叠数组，因此实际进程内存不会自动下降到这个理论值。

线上实现可以持有已完成的块和当前 partial sum，而不是重新遍历全部历史。每个来源都需要 RMSNorm、打分和加权读出；源数减少还会影响流水线并行中的表示传输。优化时可把块间的稳定结果与块内新增项用在线 softmax 合并，重点测量 kernel 启动、读写和通信开销。[官方 Block 伪代码](https://github.com/MoonshotAI/Attention-Residuals#block-attnres)

## 可以当场回答的追问

**它和 self-attention 的区别？** 两者都用归一化权重做检索，但这里固定 token，沿历史深度选来源；query 是当前子层学习的向量。self-attention 通常沿序列选 token，并从当前输入生成 query。

**RMSNorm 为什么不能放到混合之后替代 key norm？** 混合之后归一化只能调整输出尺度，无法阻止某个来源在检索打分时凭幅度占优势；二者作用位置不同。

**块内求和丢了什么？** 块内各输出失去独立的检索权重，抵消后的细节无法在块间检索时重新拆出。换来的是更少的存储来源与跨阶段传输。

**如何定位 softmax 轴写错？** 构造 `[S,B,T,D]` 输入，检查每个固定 B、T 的 S 维权重和为 1，再只改变一个 token 的来源，确认其他 token 的输出不变。

**最小实现怎样验证？** 看 `tests/test_mechanisms_depth_moe.py`：零 query、key 与 value 的尺度分离、公共 logit 平移、逐 token 独立循环、Full/Block 不同结果、块边界及来源计数均有对应行为检查。

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_mechanisms_depth_moe.py -q
```

[代码](../../src/k3lab/mechanisms/attnres.py) · [MoE 讲义](03-moe.md) · [架构总览](../architecture.md)
