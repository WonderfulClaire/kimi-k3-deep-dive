# Kimi K3 深度解读 + Agentic Post-Training Lab 🚀

> **从 K3 架构与技术报告，到 Harness / Trajectory / Verifier / Reward / Replay 的可运行实验。** 既讲清楚 Kimi K3，也把 Agentic Post-Training 做成可以继续迭代的研究代码。
>
> ✅ **权重已于 2026-07-27 深夜正式开源**（修改版 MIT，30 分钟 4000 赞登顶 HF Trending），同步发布技术报告与训练基础设施。本仓库已据技术报告全面回填更新，后续持续追社区实测与量化动态，Watch 本仓库第一时间获取更新。

[![Weights](https://img.shields.io/badge/%E6%9D%83%E9%87%8D-%E5%B7%B2%E5%BC%80%E6%BA%90%20(7%2F27)-brightgreen)](https://huggingface.co/moonshotai/Kimi-K3)
[![Context](https://img.shields.io/badge/%E4%B8%8A%E4%B8%8B%E6%96%87-1M%20tokens-blue)](#三分钟看懂-kimi-k3)
[![Params](https://img.shields.io/badge/%E5%8F%82%E6%95%B0-2.8T%20(MoE%2016%2F896)-purple)](docs/architecture.md)
[![Release](https://img.shields.io/github/v/release/WonderfulClaire/kimi-k3-deep-dive?label=release)](https://github.com/WonderfulClaire/kimi-k3-deep-dive/releases)
[![License](https://img.shields.io/github/license/WonderfulClaire/kimi-k3-deep-dive)](LICENSE)
[![Stars](https://img.shields.io/github/stars/WonderfulClaire/kimi-k3-deep-dive?style=social)](https://github.com/WonderfulClaire/kimi-k3-deep-dive/stargazers)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)

**中文** ｜ K3 资料解读 + 可运行的 Agentic Post-Training 实验骨架。资料事实见每篇文末；实验结果只在真实跑出后回填。

---

## 📌 三分钟看懂 Kimi K3

| 维度 | Kimi K3 | 一句话点评 |
| --- | --- | --- |
| 发布时间 | 2026-07-16（WAIC 前夕） | 权重**已于 7/27 开源**（修改版 MIT） |
| 总参数 | **2.8T**（全球最大开源模型） | 比 DeepSeek V4 Pro 大约 75% |
| 激活方式 | MoE 稀疏激活：**896 选 16** + 2 共享专家 | 每 token 激活 **104.2B**，约 K2 的 3 倍 |
| 上下文 | **1M tokens**（1,048,576） | 约 K2.6（256K）的 4 倍 |
| 架构核心 | **KDA**（Kimi Delta Attention）+ **AttnRes**（Attention Residuals） | 官方称综合 scaling 效率较 K2 提升约 **2.5×** |
| 多模态 | 原生视觉（图像确认可用，产品内演示视频理解） | 不是外挂 encoder，是同一个模型 |
| 权重格式 | **MXFP4** 权重 + MXFP8 激活（SFT 起量化感知训练） | 裸权重估算约 **1.4 TB** |
| API 定价 | 输入 $3/M（缓存命中 $0.30/M）、输出 $15/M | 编程负载官方缓存命中率 >90% |
| 部署建议 | 官方建议 **64+ 加速卡**超节点 | 个人电脑跑完整版：死心 😂 |
| 官方自评 | 整体仍落后 Claude Fable 5 / GPT-5.6 Sol | 但开源阵营里目前无出其右 |

**第三方独立评测速览**（2026-07-17 数据）：

- **Artificial Analysis 智能指数 v4.1**：57 分，189 个模型中排第 4
- **Vals Index**：74.70%，38 个模型中排第 2（卡在 Claude Fable 5 和 GPT-5.6 Sol 中间）
- **Arena WebDev**：初步榜单**第 1**（1679 分，1757 票，仍在变动）
- **Terminal-Bench 2.1（Vals 复测）**：80.90%，排第 2

---

## 📖 目录

| 章节 | 内容 | 适合谁 |
| --- | --- | --- |
| [一、架构拆解](docs/architecture.md) | KDA / AttnRes / Stable LatentMoE / Quantile Balancing / Per-Head Muon / SiTU / Gated MLA / MXFP4，逐个讲人话 | 想搞懂"为什么能到 2.8T"的人 |
| [二、跑分导读](docs/benchmarks.md) | 官方跑分怎么读（harness 差异陷阱）+ 第三方独立评测汇总 + K3 vs K2.6 vs 闭源旗舰 | 想知道"到底强不强"的人 |
| [三、API 上手](docs/getting-started.md) | 5 分钟跑通 kimi-k3 API：OpenAI 兼容接口、思维链保留的坑、成本估算表 | 想马上用起来的开发者 |
| [四、自部署跟踪](docs/self-hosting.md) | 权重已落地：修改版 MIT、vLLM/SGLang 首日支持、国产卡 Day 0 适配、硬件需求测算 | 想私有化部署的团队 |
| [五、已知局限与踩坑](docs/limitations.md) | 官方自曝的 3 大局限 + 思维链历史丢失导致输出不稳的实际影响 | 所有准备上生产的人 |
| [六、后训练拆解](docs/post-training.md) | SFT / Agentic RL / Harness diversification / Reward / MOPD | 想研究后训练和 Agent RL 的人 |
| [七、Agentic Post-Training Lab](docs/experiments.md) | same-harness evaluation、trajectory replay、reward ablation、SFT→RL、AutoResearch 路线 | 想真正跑实验的人 |
| [六、K3 后训练](docs/post-training.md) | SFT / Agentic RL / Harness diversification / GRM / MOPD | 想做后训练与 Agent RL 的人 |
| [七、Agentic Post-Training Lab](docs/experiments.md) | 统一 harness、trajectory、replay、verifiable reward、SFT→RL 实验协议 | 想真正跑实验的人 |

---

## 🧪 Agentic Post-Training Lab（新增）

这个仓库不再只做技术解读。现在加入了一个最小、可运行的实验骨架，用来把 **Agent Harness → Trajectory → Verifier → Reward → Replay** 这条链真正跑通。

```text
src/k3lab/
├── providers/      # OpenAI-compatible endpoint
├── harness/        # agent loop / tools / trajectory logging
├── rewards/        # deterministic verifier + composite reward
├── eval.py         # same-harness evaluation
└── replay.py       # deterministic trajectory replay
```

本地安装与 smoke test：

```bash
pip install -e ".[dev]"
pytest -q
```

接任意 OpenAI-compatible endpoint：

```bash
export K3LAB_API_KEY="..."
export K3LAB_BASE_URL="https://your-endpoint/v1"
export K3LAB_MODEL="your-model"

k3lab-eval \
  --tasks experiments/tasks/math_smoke.jsonl \
  --out runs/your-model.jsonl
```

当前 v0.1 先解决基础设施问题，不提前伪造 benchmark 数字。后续会逐步加入：

- same-harness Kimi / DeepSeek / Qwen 对照；
- reasoning history preservation ablation；
- reward ablation 与 reward hacking case study；
- fixed vs randomized harness generalization；
- 小模型 SFT → RL；
- AutoResearch 自动提出配置、运行实验、分析失败并生成下一轮假设。

详见 [K3 后训练](docs/post-training.md) 和 [实验设计](docs/experiments.md)。

---

## 🧪 Agentic Post-Training Lab

仓库现在不只做技术解读，还加入了一个最小可运行的实验骨架：

```text
Model Provider
      ↓
Agent Harness
      ↓
Tool / Environment
      ↓
Trajectory
      ↓
Verifier
      ↓
Composite Reward
      ↓
Replay / Evaluation
```

当前代码支持：

- **OpenAI-compatible provider**：同一套 harness 可切换不同模型 endpoint；
- **完整 assistant message round-trip**：方便做 reasoning/history preservation ablation；
- **trajectory logging**：记录工具调用、参数、observation、最终答案和 reward；
- **deterministic replay**：重新执行可确定复现的工具调用；
- **verifiable/composite reward**：先从透明规则奖励开始，再扩展到 judge / GRM；
- **same-harness evaluation**：避免把 harness 差异误当成模型差异。

快速跑 smoke test：

```bash
pip install -e ".[dev]"
pytest -q

export K3LAB_API_KEY="..."
export K3LAB_BASE_URL="https://your-openai-compatible-endpoint/v1"
export K3LAB_MODEL="your-model"

k3lab-eval \
  --tasks experiments/tasks/math_smoke.jsonl \
  --out runs/your-model.jsonl
```

> 现在的 math task 只是验证整条 pipeline 能跑通，不作为模型能力结论。真正的研究实验会继续补 coding/tool-use/multi-turn task、reward hacking case study、SFT→RL 和 harness generalization。

详见 [K3 后训练](docs/post-training.md) 和 [实验设计](docs/experiments.md)。

---

## ⚡ 30 秒跑通 API（不想自部署也能玩）

```python
# pip install --upgrade openai
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],
    base_url="https://api.moonshot.ai/v1",
)

resp = client.chat.completions.create(
    model="kimi-k3",
    messages=[{"role": "user", "content": "用三句话解释你自己的 KDA 注意力机制"}],
    reasoning_effort="max",   # 目前只支持 max，低/高档位官方说后续更新
)
print(resp.choices[0].message.content)
```

> ⚠️ **多轮对话必看**：K3 按"保留思维链历史"模式训练，下一轮请求必须把上一轮返回的完整 assistant 消息（含 `reasoning_content`）原样传回，否则输出质量会明显不稳定。详见[已知局限](docs/limitations.md)。

---

## 🗓️ 开源进度追踪（持续更新）

| 事项 | 状态 | 更新时间 |
| --- | --- | --- |
| 模型发布（API 可用） | ✅ 2026-07-16 | 07-16 |
| HF 仓库页 + LICENSE 上线 | ✅ [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) | 07-25 核对 |
| 完整权重放出 | ✅ **07-27 深夜上线**，修改版 MIT | 07-28 更新 |
| 技术报告 | ✅ 随权重发布，本仓库已据此回填 | 07-28 更新 |
| 推理框架支持 | ✅ vLLM / SGLang 首日支持，FlashKDA 开源 | 07-28 更新 |
| 国产算力适配 | ✅ 华为昇腾 0day、阿里云真武 M890 Day 0 | 07-28 更新 |
| 社区量化 / 本地推理探索 | ⏳ 权重刚落地，追更中 | 追更中 |
| License 逐条解读 + 首批社区实测 | ⏳ 整理中 | 追更中 |

---

## 🌍 English TL;DR

This is a **Chinese-language deep dive and reproducible agentic post-training lab** for Kimi K3 — the 2.8T-parameter MoE model (16-of-896 experts + 2 shared, 104.2B activated, 1M context) whose weights **went live on Hugging Face on July 27, 2026** under a modified MIT license, together with the technical report ([HF](https://huggingface.co/moonshotai/Kimi-K3)). Covers architecture (KDA / AttnRes / Stable LatentMoE / MXFP4, now updated with tech-report numbers), benchmark caveats, API quickstart (incl. the `reasoning_content` round-trip pitfall), self-hosting reality check (~1.4 TB raw weights, 64+ accelerators, day-0 vLLM/SGLang support), and official limitations. All facts dated and sourced. Star ⭐ to follow community benchmarks and quantization tracking.

## 🤝 参与贡献

- 权重放出后的**实测数据**（吞吐、显存、量化效果）最缺，欢迎 PR
- 发现事实错误直接开 Issue，注明来源链接即可
- 内容规范与流程见 [CONTRIBUTING.md](CONTRIBUTING.md)
- 转载/引用请注明本仓库

## ⚖️ 声明

本仓库为社区解读，与月之暗面（Moonshot AI）无官方关联。所有数据标注来源与核对日期；模型能力数据随官方与第三方更新可能变动，以一手来源为准。文字内容以 [MIT License](LICENSE) 发布。

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=WonderfulClaire/kimi-k3-deep-dive&type=Date)](https://star-history.com/#WonderfulClaire/kimi-k3-deep-dive&Date)

**如果这份解读帮你省了时间，点个 ⭐ Star 支持追更！**
