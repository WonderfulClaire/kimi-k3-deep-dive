# 基模机制学习路线

这组讲义从一条 token 的计算路径出发，把 Kimi K3 的架构、预训练、后训练和系统实现接起来。每个主题都尽量回答四件事：什么问题促成这个设计，实际算什么，怎样用小实验核对，以及规模扩大后成本在哪里。

## 先跑一遍，再读公式

在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,mechanisms]"
PYTHONPATH=src python scripts/run_mechanism_lab.py --out reports/mechanisms
pytest -q
```

这条 CPU 路线不需要模型权重或 API Key。生成的 [REPORT.md](../../reports/mechanisms/REPORT.md) 说明每个数值怎样得到，[results.json](../../reports/mechanisms/results.json) 保存配置和中间结果。模型端点与 GPU 后训练沿用已有 [Agent Lab](../experiments.md) 和 [训练入口](../training.md)。

## 按计算链阅读

| 主题 | 读完能做什么 | 对应实现 |
|---|---|---|
| [01 / KDA：记忆、遗忘与分块求解](01-kda.md) | 从在线平方损失推导 delta rule，手算门控顺序，解释递推为何可以变成块内三角求解 | [kda.py](../../src/k3lab/mechanisms/kda.py) |
| [02 / AttnRes：沿深度检索](02-attnres.md) | 区分 token attention 与 depth attention，说明 key RMSNorm、原 value 和 Full/Block 取舍 | [attnres.py](../../src/k3lab/mechanisms/attnres.py) |
| [03 / MoE：路由、负载和潜在空间](03-moe.md) | 手算 Top-k 与 combine，解释 selection bias，复核 QB 阈值与通信账本 | [moe.py](../../src/k3lab/mechanisms/moe.py) |
| [04 / 预训练与系统](04-pretraining-systems.md) | 从有效 token 的 loss 走到 optimizer、并行通信、混合精度、状态内存与数值排障 | [架构总览](../architecture.md) |
| [05 / 后训练与策略蒸馏](05-posttraining.md) | 解释 SFT mask、policy ratio、奖励与 KL、MOPD 采样与长轨迹版本问题 | [train/](../../src/k3lab/train/) |
| [06 / 面试 walkthrough 与动手题](06-interview-walkthrough.md) | 用一条完整计算链讲项目，并接受公式、代码、成本和失败分析的追问 | [实验报告](../../reports/mechanisms/REPORT.md) |

## 怎样把阅读变成自己的分析

先预测一个干预会怎样改变输出，再改代码验证。比如固定 KDA 的 key，连续写两个互相冲突的 value；将 beta 置零；只清除一个 alpha 通道；将 AttnRes query 置零；把 MoE 的 bias 放进选集但移出 combine。能解释预测与结果为什么一致，才算理解这条计算链。

接着改一个实验条件，并保留失败：对 QB 换一批分布完全不同的 token，比较校准批和新批的专家负载；对 AttnRes 固定子层输出，再思考完整网络中输出如何随混合改变；对 KDA 同时比较块输出和最终状态，防止当下看似正确但续接出错。

最后给每个模块写下一个实际判断：什么条件下值得用，它把成本移到了哪里，下一项需要测什么。这里的数值检验是机制级实验；GPU 核函数速度和训练收益需要各自的工作负载、实现与测量。

[官方来源与版本定位](sources.md) · [返回仓库](../../README.md)
