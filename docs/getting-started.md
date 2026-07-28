# 三、API 上手：5 分钟跑通 kimi-k3

> 权重还没放出（7/27 前），但 API 现在就能用。本文带你避开 K3 特有的几个坑。核对至 2026-07-25。

## 1. 拿 Key、发第一个请求

K3 的 API 完全兼容 OpenAI Chat Completions 格式，模型名 `kimi-k3`：

```python
# pip install --upgrade openai
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ["MOONSHOT_API_KEY"],   # 在 Kimi 开放平台创建
    base_url="https://api.moonshot.ai/v1",     # 国内平台参见 platform 文档
)

resp = client.chat.completions.create(
    model="kimi-k3",
    messages=[{"role": "user", "content": "介绍一下你的 Attention Residuals 机制"}],
    reasoning_effort="max",   # 当前唯一档位；低/高档位官方说后续开放
)
print(resp.choices[0].message.content)
```

要点：
- **K3 永远在思考**：它是推理模型且默认（目前也只能）max 思考力度，简单问题也会先想再答——延迟和输出 token 都比非推理模型高；
- **流式输出**加 `stream=True` 即可，格式与 OpenAI SSE 完全一致。

## 2. 最大的坑：多轮对话必须回传思维链

K3 按"**保留思维链历史**"模式训练。官方明确要求：多轮对话/工具调用时，要把 API 返回的**完整 assistant 消息（包括 `reasoning_content` 字段）原样放回下一轮的 messages**。

```python
messages = [{"role": "user", "content": "帮我分析这段代码"}]
resp = client.chat.completions.create(model="kimi-k3", messages=messages, reasoning_effort="max")

msg = resp.choices[0].message
# ✅ 正确：整个 message 对象放回去（含 reasoning_content）
messages.append(msg.model_dump(exclude_none=True))
messages.append({"role": "user", "content": "那第二个函数呢？"})

# ❌ 错误：只放 content 字符串——第二轮开始输出质量会明显不稳定
# messages.append({"role": "assistant", "content": msg.content})
```

连带两个推论：
- **别在会话中途把其他模型的对话切到 K3**（历史里没有 K3 格式的思维链，质量会崩）；
- 自研 agent 框架接 K3 前，先确认框架不会裁剪 assistant 消息字段。

## 3. 成本估算：贵，但缓存是命门

| Token 类型 | 价格 |
| --- | --- |
| 输入（缓存未命中） | $3.00 / M |
| 输入（缓存命中） | **$0.30 / M**（省 90%） |
| 输出 | $15.00 / M |

**实际场景估算**（含缓存对比）：

| 请求 | 无缓存 | 输入全命中缓存 |
| --- | --- | --- |
| 10K 入 + 2K 出 | $0.06 | $0.033 |
| 100K 入 + 10K 出 | $0.45 | $0.18 |
| 500K 入 + 20K 出 | $1.80 | $0.45 |

三条省钱铁律：
1. **把稳定不变的内容放前缀**（系统提示、代码库、文档），变化的内容放末尾——缓存按前缀匹配，官方编程负载命中率能到 90%+；
2. **K3 话多**：第三方实测其输出 token 约为同级推理模型的 2 倍，预算按输出 ×2 估；
3. 1M 上下文 ≠ 每次塞满 1M：长 prefill 又慢又贵，检索/摘要该用还得用。

## 4. 多模态：图像直接发

K3 是原生多模态，图像输入走标准 OpenAI 图像格式：

```python
resp = client.chat.completions.create(
    model="kimi-k3",
    messages=[{
        "role": "user",
        "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}},
            {"type": "text", "text": "这张架构图里哪个模块是瓶颈？"},
        ],
    }],
    reasoning_effort="max",
)
```

注意：视频理解目前在 Kimi 产品内演示过，**API 端点是否收视频文件请以官方文档为准**，别默认可用。

## 5. 什么任务值得用 K3（以及什么不值得）

✅ 值得：大仓库代码导航与修改、长时程编码 agent（编译-测试-修复循环）、跨大量文档的研究、报表/PPT/视觉材料分析、带视觉反馈的前端/游戏/CAD 工作流。

❌ 不值得：简单分类、短对话、信息抽取、高并发低延迟场景——K3 的推理开销和单价在这些场景纯属浪费，小模型更香。

---

**来源**：[Kimi K3 Quickstart](https://platform.kimi.ai/docs/guide/kimi-k3-quickstart)、[Kimi API 平台定价页](https://platform.kimi.ai/)、官方博客 Availability 章节。价格可能调整，下单前再核对一次。

⬅️ 上一篇：[跑分导读](benchmarks.md) ｜ ➡️ 下一篇：[自部署跟踪](self-hosting.md)
