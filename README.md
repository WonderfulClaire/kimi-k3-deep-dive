# Kimi K3 深度解读 · Deep Dive 🚀

> **2.8 万亿参数、100 万 token 上下文、全球首个开源 3T 级模型** —— 月之暗面 Kimi K3 完全解读。
>
> 📅 **权重开源倒计时：官方已官宣 2026-07-27 当天正式发布完整权重**（HF 仓库页已挂倒计时），本仓库持续追更（权重、License、技术报告、vLLM 支持、社区量化），Watch 本仓库第一时间获取更新。

[![Weights](https://img.shields.io/badge/%E6%9D%83%E9%87%8D-7%2F27%20%E5%AE%98%E5%AE%A3%E5%8F%91%E5%B8%83-orange)](https://huggingface.co/moonshotai/Kimi-K3)
[![Context](https://img.shields.io/badge/%E4%B8%8A%E4%B8%8B%E6%96%87-1M%20tokens-blue)](#三分钟看懂-kimi-k3)
[![Params](https://img.shields.io/badge/%E5%8F%82%E6%95%B0-2.8T%20(MoE%2016%2F896)-purple)](docs/architecture.md)
[![Release](https://img.shields.io/github/v/release/WonderfulClaire/kimi-k3-deep-dive?label=release)](https://github.com/WonderfulClaire/kimi-k3-deep-dive/releases)
[![License](https://img.shields.io/github/license/WonderfulClaire/kimi-k3-deep-dive)](LICENSE)
[![Stars](https://img.shields.io/github/stars/WonderfulClaire/kimi-k3-deep-dive?style=social)](https://github.com/WonderfulClaire/kimi-k3-deep-dive/stargazers)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen)](CONTRIBUTING.md)

**中文** ｜ 信息核对至 2026-07-25（晚间），一手来源见每篇文末（官方博客 / HF / Artificial Analysis / Vals）

---

## 📌 三分钟看懂 Kimi K3

| 维度 | Kimi K3 | 一句话点评 |
| --- | --- | --- |
| 发布时间 | 2026-07-16（WAIC 前夕） | 权重**官宣 7/27 当天**全量开源 |
| 总参数 | **2.8T**（全球最大开源模型） | 比 DeepSeek V4 Pro 大约 75% |
| 激活方式 | MoE 稀疏激活：**896 选 16** 专家 | 每 token 不用全部参数，算力可控 |
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
| [四、自部署跟踪](docs/self-hosting.md) | 权重/License/vLLM(KDA) 状态追踪、硬件需求测算、社区量化动态 —— **7/27 后重点更新** | 想私有化部署的团队 |
| [五、已知局限与踩坑](docs/limitations.md) | 官方自曝的 3 大局限 + 思维链历史丢失导致输出不稳的实际影响 | 所有准备上生产的人 |

---

## ⚡ 30 秒跑通 API（先于权重开源就能玩）

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
| 完整权重放出 | ⏳ **官宣 07-27 当天发布**，HF 页已挂倒计时 | 07-25 更新 |
| 技术报告 | ⏳ 官方确认将发布（架构设计、训练细节、评估结果） | 07-25 更新 |
| vLLM KDA（含 prefill cache）合入 | ⏳ 随权重发布 | 追更中 |
| 社区量化 / 本地推理探索 | ⏳ 等权重 | 追更中 |

---

## 🌍 English TL;DR

This is a **Chinese-language deep dive** into Kimi K3 — the 2.8T-parameter MoE model (16-of-896 experts, 1M context) that Moonshot AI has **confirmed will be open-weighted on July 27, 2026** (countdown live on [HF](https://huggingface.co/moonshotai/Kimi-K3)). Covers architecture (KDA / AttnRes / Stable LatentMoE / MXFP4), benchmark caveats, API quickstart (incl. the `reasoning_content` round-trip pitfall), self-hosting hardware math (~1.4 TB raw weights, 64+ accelerators), and official limitations. All facts dated and sourced. Star ⭐ to follow the weights-release tracking.

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
