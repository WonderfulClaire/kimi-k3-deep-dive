# Stable LatentMoE：选择、混合、潜在宽度与负载均衡

MoE 把“模型存有多少专家”与“一个 token 计算多少专家”分开。LatentMoE 再把主干宽度与路由专家宽度分开。K3 公开设置是 896 个路由专家、每 token 选择 16 个，另有 2 个全宽共享专家；主干维度 7168，路由潜在维度 3584。[K3 §2.3](https://arxiv.org/html/2607.24653v1#S2.SS3)、[架构表 1](https://arxiv.org/html/2607.24653v1#S3.T1)

下面先用三个专家做一次路由，再把 dispatch、combine、负载偏斜和潜在空间的成本逐项展开。代码采用 float64；线性专家用于检查路由，参数核算单独按 gated FFN 定义计算。

## 一个 token，两个被选专家，两种不同的分数

设 router 的 sigmoid 分数为：

\[
s=(0.9,0.8,0.2),\qquad k=2.
\]

不加 bias 时会选择专家 0、1。如果当前专家 1 过载、专家 2 长期闲置，可以为选择过程加偏置 \(b=(0,-0.8,1)\)，得到 \(s+b=(0.9,0,1.2)\)，于是选择专家 2、0。

混合这两个专家时，使用的仍是原始分数：

\[
p_2=\frac{0.2}{0.2+0.9}=\frac2{11},\qquad
p_0=\frac{0.9}{0.2+0.9}=\frac9{11}.
\]

选择分数中出现 1.2 完全可行，它没有被当作概率。若误用加偏置分数计算混合权重，专家 2 将得到过大的份额，负载调节就会同时改变输出融合的含义。公开路由定义见 [§2.3.3，Eq. (13)](https://arxiv.org/html/2607.24653v1#S2.SS3.SSS3)。

```python
import numpy as np
from k3lab.mechanisms.moe import topk_route

plan = topk_route(
    np.array([[0.9, 0.8, 0.2]]),
    k=2,
    bias=np.array([0., -0.8, 1.]),
)
print(plan.indices)  # [[2, 0]]
print(plan.weights)  # [[2/11, 9/11]]
print(plan.loads)    # [1, 0, 1]
```

## 形状和操作顺序

约定 T 是 token 数，E 是专家数，k 是每 token 选择数，d 是主干宽度，\(\ell\) 是路由潜在宽度，h 是专家 FFN 的中间维度。

| 对象 | 形状 | 操作 |
| :--- | :--- | :--- |
| 主干输入 X | `[T,d]` | 供 router、共享专家和降维投影使用 |
| 原始分数 S | `[T,E]` | 对 router logits 逐元素 sigmoid |
| 独立 bias b | `[E]` | 只加到专家选择分数中 |
| `indices` | `[T,k]` | 每行的 Top-k 专家编号 |
| `weights` | `[T,k]` | 选中原始分数在每行内归一化 |
| `loads` | `[E]` | 每个专家收到多少 token 分派，和为 Tk |
| 潜在输入 Z | `[T,ell]` | X 经降维投影得到 |
| 聚合输出 U | `[T,ell]` | 专家结果按原 token 顺序加权回收 |

`topk_route` 只接收 `[0,1]` 的原始 sigmoid 分数；若从任意 logits 开始，调用 `route_logits`。选择中的并列按较小专家编号打破，以保证可复现。若选中分数总和为 0，接口报错，避免生成没有定义的混合权重。

固定被选集合后，对选中分数 \(s_j\)，权重的导数仍是归一化比分：\(\partial p_j/\partial s_j=(S-s_j)/S^2\)，其中 \(S\) 是该 token 被选分数之和。bias 不直接进入这个式子。离散 Top-k 集合会随分数和 bias 改变，因此在整个训练中，两条路径仍通过选择行为发生联系。

## Dispatch 和 combine 要检查 token 身份

考虑两个 token，输入分别为 \((1,2)\)、\((3,4)\)。三个线性专家输出一个标量：

\[
E_0(x)=x_0,\quad E_1(x)=2x_1,\quad E_2(x)=x_0+x_1.
\]

第一个 token 的分数为 \((0.6,0.3,0.1)\)，选专家 0、1，输出：

\[
\frac{0.6\times1+0.3\times4}{0.9}=2.
\]

第二个 token 的分数为 \((0.1,0.5,0.4)\)，选专家 1、2，输出：

\[
\frac{0.5\times8+0.4\times7}{0.9}=\frac{68}{9}.
\]

dispatch 按专家把 token 分组，所以专家 1 同时接收两个 token；combine 必须根据原 token 编号，把每个专家结果放回正确位置并累加 k 份贡献。只比较输出形状、分派总数或专家均值，很难发现 token 排序被打乱的问题。

`moe_linear_dispatch` 按专家分组并使用 `np.add.at` 回收；`moe_linear_reference` 按 token、选中专家、输入与输出坐标逐项累加。两者在随机路由、多个贡献以及空专家的场景下比较，另用上面的手算值确认输出。这些小专家用于隔离 dispatch/combine 的语义；实际 K3 专家是 gated FFN。

## 先把启发式 bias 和 Quantile Balancing 分清

若 8 个 token 全部选择同一个专家，四专家的负载是 \((8,0,0,0)\)，平均负载为 2，最大值/平均值为 4。一个教学基线可以执行：

\[
b'_j=b_j+\eta\,\operatorname{sign}(\bar n-n_j),\qquad
b'\leftarrow b'-\operatorname{mean}(b').
\]

这样过载专家的 bias 降低，闲置专家的 bias 提高。步长 \(\eta\) 小可能反应慢，大可能震荡。代码把它明确命名为 `heuristic_balance_bias`；它是用于解释独立 bias 的 sign 更新基线。

K3 的 Quantile Balancing 使用每个专家相对 token 选择边界的分位阈值，推导与实际 histogram 估计见 [§2.3.3](https://arxiv.org/html/2607.24653v1#S2.SS3.SSS3) 和[附录 C、D](https://arxiv.org/html/2607.24653v1#A3)。下面给出独立的顺序统计量参考。

### 从目标负载推导阈值

有 T 个 token、E 个专家、每 token 选 k 个专家，总分派数为 Tk，均匀目标是 \(q=Tk/E\)。先用**旧 bias** 给每个 token 排序，把第 \(k+1\) 大的加偏置分数记为 \(\alpha_i\)。它是一个专家要进入该 token 的 Top-k 所需跨过的边界。

固定这些边界后，若下一步为专家 j 选择候选 bias \(\widehat b_j\)，希望满足：

\[
\sum_{i=1}^{T}\mathbf 1[s_{ij}+\widehat b_j>\alpha_i]=q.
\]

定义每个 token 对该专家的 margin 为 \(m_{ij}=s_{ij}-\alpha_i\)。上式等价于希望有 q 个 margin 超过阈值 \(-\widehat b_j\)。当 q 为整数且边界附近没有并列时，取该列第 \(q+1\) 大的 margin 作为阈值即可：

\[
\widehat b_j=-m_{(q+1),j},\qquad
b^{next}=\widehat b-\operatorname{mean}(\widehat b).
\]

中心化不会改变同一个 token 的 Top-k，因为所有专家选择分数都减去同一个常数。它有助于把自由漂移的公共偏移去掉；阈值的逐列计数解释应使用未中心化 bias 和原固定边界，或同时平移边界，避免把坐标系混用。

`quantile_balance_bias` 的选择范围明确限定为 `1 <= k < E`、整数目标 q，采用精确顺序统计量。它没有使用 `np.quantile` 默认线性插值；插值分位数的边界约定可能与“第 q+1 大”不同。新的 bias 应在**下一批**生效；当前批先记录旧路由，再生成更新，避免用本批统计重新定义本批训练路由。[K3 Eq. (14)](https://arxiv.org/html/2607.24653v1#S2.SS3.SSS3)

### 为什么不能承诺下一批严格均衡

上面的列计数推导固定了旧 \(\alpha_i\)，真实下一步会同时更改多个专家的 bias，也会遇到新 token。专家之间的排序边界随之改变，因此单次估计与全局平衡分派的完整求解仍有区别。

还有一个容易漏掉的并列来源：即使原始分数连续随机，当专家 j 在多个 token 上恰好提供第 k+1 名边界时，\(s_{ij}-\alpha_i=-b_j\) 会重复出现。此时严格大于阈值的数量可能小于 q。完全相同分数配合固定 tie-break 时，偏置校准后也可能仍然集中到一个专家。测试显式覆盖了这种情况。

真实规模下还需要跨设备和梯度累积步统计全局 margin。公开方法通过每专家 histogram 的计数 all-reduce 估计分位数；本代码只实现小规模 exact 校准，没有模拟分布式直方图、量化 bin 误差或专家并行调度。[公开 histogram 实现说明](https://arxiv.org/html/2607.24653v1#S2.SS3.SSS3)

## LatentMoE 把哪一部分变窄

用行向量约定，降维矩阵为 \(W_{down}\in\mathbb R^{d\times\ell}\)，升维矩阵为 \(W_{up}\in\mathbb R^{\ell\times d}\)。先算 \(z=xW_{down}\)，路由专家在 \(\ell\) 维上工作：

\[
u=\sum_{j\in\mathcal T(x)}p_j E_j(z),\qquad
y=\sum_{r=1}^{N_s}E_r^{shared}(x)+\operatorname{RMSNorm}(u)W_{up}.
\]

共享专家接收 d 维主干输入；路由专家接收 \(\ell\) 维潜在输入。RMSNorm 位于专家加权聚合之后、升维之前，避免不同专家和权重带来的幅度变化直接进入升维投影。[K3 §2.3.1](https://arxiv.org/html/2607.24653v1#S2.SS3.SSS1)

SiTU-GLU 对 gate 的线性因子和 up 分支分别施加平滑 soft cap。若 \(a=zW_g,c=zW_u\)，则门控中间值为：

\[
g(z)=\left[\beta_1\tanh(a/\beta_1)\odot\sigma(a)\right]
\odot\left[\beta_2\tanh(c/\beta_2)\right].
\]

每个中间坐标的绝对值不超过 \(\beta_1\beta_2\)，但后续输出矩阵仍能改变幅度。K3 设置 \(\beta_1=4,\beta_2=25\)；这个边界作用于门控中间值，不能直接当作完整 MoE 输出的逐坐标上界。[§2.3.2](https://arxiv.org/html/2607.24653v1#S2.SS3.SSS2)

## 参数、矩阵运算和通信各算一遍

假设每个 gated FFN 有 gate、up、down 三个矩阵，FFN 中间维度固定为 h。路由专家走 \(\ell\to h\to\ell\)，共享专家走 \(d\to h\to d\)；两者只有中间维度 h 在本公式中取相同值。所有线性层 bias 与独立路由均衡 bias 状态不计入，潜在 RMSNorm 的通道增益计入。

一个潜在专家参数量是 \(3\ell h\)，E 个路由专家为 \(3E\ell h\)，每 token 选中专家对应的参数为 \(3k\ell h\)。当比较全宽路由专家时，保持 E、k、h 相同，再把 \(\ell\) 换成 d。共享专家始终保留全宽，不随 \(\ell\) 改变。

此外还有降维和升维共 \(2d\ell\) 个参数、router 的 dE 个参数、潜在 RMSNorm 的 \(\ell\) 个参数，以及 \(N_s\) 个全宽共享专家的 \(3N_sdh\) 个参数。因此路由专家的变窄比例不会直接等于整个 MoE 模块的总参数或总耗时比例。

取一个独立核算例子：d=8、\(\ell=4\)、h=3、E=6、k=2、共享专家数 1、T=5。

| 项目 | 独立计算 | 计数 |
| :--- | :--- | ---: |
| 全部路由专家 | \(6\times(4\times3+4\times3+3\times4)\) | 216 参数 |
| 共享专家 | \(8\times3+8\times3+3\times8\) | 72 参数 |
| 降维与升维 | \(8\times4+4\times8\) | 64 参数 |
| Router | \(8\times6\) | 48 参数 |
| 潜在 RMSNorm | 4 个通道增益 | 4 参数 |
| 模块总量 | \(216+72+64+48+4\) | 404 参数 |
| 每 token 激活量 | \(2\times36+72+64+48+4\) | 260 参数 |

如果仅统计矩阵乘加对，一次乘加记为 1 MAdd，5 个 token 是 \(5\times(72+72+64+48)=1280\) MAdds。换成常见 FLOP 计数时，一次乘法加一次加法约为 2 FLOPs；这里没有把 RMSNorm、sigmoid、Top-k、激活函数、权重加权和与调度算进矩阵 MAdds。

再独立看逻辑通信量。若所有 Tk 个分派都发送到远程专家，dispatch 发送 \(Tk\ell\) 个元素，combine 回收相同数量，按每元素 b 字节：

\[
V_{logical}=2Tk\ell b.
\]

本例若 b=2，则是 \(2\times5\times2\times4\times2=160\) 字节。保持其余条件、使用 d=8 的全宽专家则是 320 字节。这是激活数据的逻辑 payload，实际网卡流量还受本地专家比例、协议、metadata、padding、拓扑、replication、collective 实现和回传聚合位置影响。

`latent_moe_cost` 按上述式子返回计数，不从本机 NumPy 耗时推断大模型吞吐。它的 FFN 参数模型包含三矩阵门控结构，和前面用于检查 dispatch 的单矩阵线性专家分开定义。

## 面试时继续追问什么

**更多专家为什么总参数变大、每 token 算力未必同比变大？** E 控制存有多少 FFN；k 控制一个 token 使用多少 FFN。但 router 仍需给 E 个专家打分，参数驻留和通信也会变化，不能只看 k。

**为什么负载不均影响吞吐？** 专家并行的一轮通常要等较慢设备完成。平均分派数相同，热点专家所在设备可能拥有更长 GEMM 和队列；空闲设备没有自动补上那部分计算。要看设备聚合负载与关键路径，不能只看专家平均值。

**bias 没进权重，为什么输出仍会变？** 被选专家集合改变了。bias 调节选择路径；固定同一选择集合时，融合权重继续只由原始分数决定。

**潜在维度减半是不是整体速度翻倍？** 本计数中路由专家矩阵和逻辑激活 payload 随 \(\ell\) 线性变化，但全宽共享路径保持不变，router、投影、排序、通信启动与设备利用率还有各自成本，需要实测才能得到速度。

**怎样证明路由实现没有打乱 token？** 同时检查手算输出、专家分组实现与 token 顺序独立参考；加入空专家和 k 大于 1 的情形，再确认 loads 的和是 Tk、每 token 权重和为 1。

**怎样评价均衡更新？** 用旧 bias 路由当前批，生成新 bias，再在下一批统计负载、质量和时间。校准批上回放可以观察机制，跨批测量才回答适应能力。还应报告 ties 和输入分布变化。

```bash
PYTHONPATH=src .venv/bin/python -m pytest tests/test_mechanisms_depth_moe.py -q
```

[代码](../../src/k3lab/mechanisms/moe.py) · [AttnRes 讲义](02-attnres.md) · [架构总览](../architecture.md)
