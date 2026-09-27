# Kimi K3 深度解读 + Agentic Post-Training Lab 🚀

> **从 K3 架构与技术报告，到 Harness / Trajectory / Verifier / Reward / Replay 的可运行实验。** 既讲清楚 Kimi K3，也把 Agentic Post-Training 做成可以继续迭代的研究代码。
>
> ✅ **权重已于 2026-07-27 深夜正式开源**（修改版 MIT），同步发布技术报告与训练基础设施。本仓库已据技术报告回填主要技术点；实验结果只在真实跑出后记录。

[![Weights](https://img.shields.io/badge/%E6%9D%83%E9%87%8D-%E5%B7%B2%E5%BC%80%E6%BA%90-brightgreen)](https://huggingface.co/moonshotai/Kimi-K3)
[![Context](https://img.shields.io/badge/%E4%B8%8A%E4%B8%8B%E6%96%87-1M%20tokens-blue)](#三分钟看懂-kimi-k3)
[![Params](https://img.shields.io/badge/%E5%8F%82%E6%95%B0-2.8T%20MoE-purple)](docs/architecture.md)
[![License](https://img.shields.io/github/license/WonderfulClaire/kimi-k3-deep-dive)](LICENSE)
[![Stars](https://img.shields.io/github/stars/WonderfulClaire/kimi-k3-deep-dive?style=social)](https://github.com/WonderfulClaire/kimi-k3-deep-dive/stargazers)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)

**中文** ｜ K3 资料解读 + 可运行的 Agentic Post-Training 实验骨架。资料事实见每篇文末；benchmark 数字带日期，实验数字只在真实运行后回填。

---

## 📌 三分钟看懂 Kimi K3

| 维度 | Kimi K3 | 一句话点评 |
| --- | --- | --- |
| 发布时间 | 2026-07-16 | 权重 7/27 开源（修改版 MIT） |
| 总参数 | **2.8T** | 超大规模稀疏 MoE |
| 激活方式 | **896 选 16** + 2 共享专家 | 每 token 激活约 **104.2B** |
| 上下文 | **1M tokens** | 面向长上下文与长时程 agent |
| 架构核心 | **KDA + AttnRes** | 重点看长上下文效率与残差路由 |
| 权重格式 | **MXFP4** 权重 + MXFP8 激活 | 自部署仍需要大规模硬件 |
| API | OpenAI-compatible | 适合接统一 harness 做控制实验 |

---

## 📖 目录

| 章节 | 内容 |
| --- | --- |
| [一、架构拆解](docs/architecture.md) | KDA / AttnRes / Stable LatentMoE / Quantile Balancing / Per-Head Muon / SiTU / Gated MLA / MXFP4 |
| [二、跑分导读](docs/benchmarks.md) | 官方跑分、第三方评测、harness 差异为什么会污染 agent benchmark |
| [三、API 上手](docs/getting-started.md) | OpenAI-compatible API、多轮历史、成本与缓存 |
| [四、自部署跟踪](docs/self-hosting.md) | 权重、推理框架、硬件需求 |
| [五、已知局限与踩坑](docs/limitations.md) | 长轨迹、history、agent 行为边界 |
| [六、K3 后训练](docs/post-training.md) | SFT / Agentic RL / harness diversification / reward / MOPD |
| [七、Agentic Post-Training Lab](docs/experiments.md) | same-harness eval / reward hacking / SFT→RL / AutoResearch |

---

## 🧪 Agentic Post-Training Lab v0.3

当前仓库已经有一条能跑通的研究链：

```text
OpenAI-compatible Model
        ↓
Agent Harness
        ↓
Tool / Synthetic Environment
        ↓
Trajectory
        ↓
Public Signal
        ↓
Held-out + Integrity Verifier
        ↓
Composite Reward
        ↓
Replay / Cross-model Summary
```

代码结构：

```text
src/k3lab/
├── providers/      # OpenAI-compatible endpoint
├── harness/        # agent loop / tools / trajectory logging
├── envs/           # MiniRepo coding environment
├── rewards/        # verifier-compatible composite reward
├── eval.py         # same-harness evaluation
├── replay.py       # deterministic trajectory replay
├── summary.py      # aggregate multiple model runs
└── export_sft.py   # verified trajectories → SFT JSONL
```

### 本地验证

```bash
pip install -e ".[dev]"
pytest -q
```

### 跑 coding / tool-use task

```bash
export K3LAB_API_KEY="..."
export K3LAB_BASE_URL="https://your-openai-compatible-endpoint/v1"
export K3LAB_MODEL="your-model"

k3lab-eval \
  --tasks experiments/tasks/repo_patch.jsonl \
  --out runs/your-model.jsonl
```

然后把多个模型放到同一张表：

```bash
k3lab-summarize \
  runs/kimi.jsonl \
  runs/deepseek.jsonl \
  runs/qwen.jsonl
```

### 为什么专门做 Reward Hacking

MiniRepo 的 `run_public_tests` 故意保留了一个 naive signal：agent 如果把 public tests 删空，可能得到 **0/0 tests = PASS**。但最终 reward 不信这个信号，而是重新检查：

- 原始 public tests；
- held-out tests；
- test file integrity。

所以项目里可以直接复现：

```text
naive reward PASS  ≠  actual task success
```

这比只写一句「要防 reward hacking」更有实验价值。

v0.3 已把两类 ablation 直接做成可运行参数：

```bash
# reasoning/history ablation
k3lab-eval --history-mode full ...
k3lab-eval --history-mode no_reasoning ...

# harness generalization
k3lab-eval --harness-variant canonical ...
k3lab-eval --harness-variant compact ...
k3lab-eval --harness-variant alternate ...
```

secure verifier 通过的 rollout 还能直接导出为 SFT 数据：

```bash
k3lab-export-sft runs/*.jsonl --out data/verified_sft.jsonl
```

因此现在已经连上 **rollout → verifier → successful trajectory → SFT dataset**。下一阶段继续补真正的小模型 SFT / RL 训练和 held-out harness 结果。真实 Kimi / DeepSeek / Qwen 对照结果不提前编造。

---

## ⚡ 30 秒跑通 K3 API

```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.ai/v1",
)

resp = client.chat.completions.create(
    model="kimi-k3",
    messages=[{"role": "user", "content": "用三句话解释 KDA 注意力机制"}],
    reasoning_effort="max",
)
print(resp.choices[0].message.content)
```

> 多轮/工具调用时要注意 K3 的 assistant history 格式要求。具体见 [API 上手](docs/getting-started.md) 与 [已知局限](docs/limitations.md)。

---

## 🗓️ 开源进度追踪

| 事项 | 状态 |
| --- | --- |
| 模型发布 / API | ✅ |
| 完整权重 | ✅ |
| 技术报告 | ✅ |
| vLLM / SGLang 等推理支持 | ✅ |
| 本仓库 Agentic Post-Training Lab v0.3 | ✅ |
| Same-harness 多模型真实结果 | ⏳ 待实测 |
| 小模型 SFT → RL | ⏳ |
| AutoResearch loop | ⏳ |

---

## 🌍 English TL;DR

This repository combines a Chinese technical deep dive into Kimi K3 with a small reproducible **Agentic Post-Training Lab**. The lab now includes an OpenAI-compatible provider layer, an agent harness, trajectory logging/replay, a synthetic coding environment, held-out and integrity verification, reward-hacking tests, history/harness ablations, compositional rewards, cross-model run summaries, and verified-trajectory export for SFT. No model benchmark numbers are reported until they are actually run.

## 🤝 参与贡献

目前最需要的是：

- same-harness 的真实多模型运行结果；
- coding / structured tool-use task；
- harness generalization 实验；
- SFT / RL 配置与可复现训练日志；
- 事实纠错与一手来源补充。

内容规范见 [CONTRIBUTING.md](CONTRIBUTING.md)。

## ⚖️ 声明

本仓库为社区研究与解读，与月之暗面（Moonshot AI）无官方关联。模型能力、价格、榜单等随时间变化；引用时请核对原始来源和日期。仓库文字与代码按 [MIT License](LICENSE) 发布。

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=WonderfulClaire/kimi-k3-deep-dive&type=Date)](https://star-history.com/#WonderfulClaire/kimi-k3-deep-dive&Date)
