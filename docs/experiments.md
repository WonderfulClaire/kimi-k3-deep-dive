# 七、Agentic Post-Training Lab：实验设计与复现协议

这页只记录**我们自己能复现的实验**。未跑出的模型结果不提前填数字；基础设施行为由单元测试和 GitHub Actions 验证。

## 1. 统一评测协议

同一组 task 固定任务、verifier、步数与采样设置，只替换 model endpoint 或单个实验变量。首轮记录：

| 字段 | 用途 |
| --- | --- |
| task success | 最终状态是否通过 secure verifier |
| tool calls | agent 行为长度 |
| invalid tool calls | tool-use 稳定性 |
| input/output tokens | 成本与过度思考 |
| verifier components | public / held-out / integrity 分开看 |
| reward components | reward 为什么高或低 |
| history mode | full vs no_reasoning |
| harness variant | canonical / compact / alternate |

## 2. Same Harness, Different Models

```bash
pip install -e ".[dev]"

export K3LAB_API_KEY="..."
export K3LAB_BASE_URL="https://your-endpoint/v1"

K3LAB_MODEL="your-model" \
k3lab-eval \
  --tasks experiments/tasks/repo_patch.jsonl \
  --out runs/your-model.jsonl
```

对 Kimi、DeepSeek、Qwen 等 OpenAI-compatible endpoint 重复同一命令，只换 endpoint/model。然后：

```bash
k3lab-summarize runs/kimi.jsonl runs/deepseek.jsonl runs/qwen.jsonl
```

输出 success、平均 reward、tool calls、invalid calls 与 token 使用量。

## 3. MiniRepo Coding Environment

`repo_patch.jsonl` 已经是多步 tool-use：

```text
read SPEC.md
    ↓
inspect solution.expr / public tests
    ↓
edit solution.expr
    ↓
run_public_tests
    ↓
final repository state
    ↓
secure verifier
```

环境不执行模型生成的任意 shell/Python，而是让 agent 修改一个安全表达式 DSL。这样可以研究 agent 行为、reward 和 generalization，同时避免把 benchmark 本身变成任意代码执行器。

secure verifier 同时检查：

1. 原始 public tests；
2. held-out tests；
3. protected test files 是否被篡改。

最终 reward 来自**最终环境状态**，不是 agent 自己声称「修好了」。

## 4. Reward Hacking Case Study

环境故意保留一个 naive public-test signal：

```text
agent edits tests_public.json
        ↓
run_public_tests trusts current file
        ↓
tests become empty
        ↓
0 / 0 => naive PASS
```

但 secure verifier 不信这个 mutable signal，而是重新检查原始 public tests、held-out tests 和 test-file integrity。

单元测试覆盖两类典型 failure：

- 删除 public tests：naive signal PASS，但 integrity FAIL；
- hard-code public cases：public PASS，但 held-out FAIL。

核心是区分：

```text
reward observable to policy  !=  actual task success
```

## 5. History Preservation Ablation

v0.3 已经把这个实验做成 CLI 参数。

### Full history

完整 round-trip provider 返回的 assistant message：

```bash
k3lab-eval \
  --history-mode full \
  --tasks experiments/tasks/repo_patch.jsonl \
  --out runs/model-full.jsonl
```

### No reasoning

只移除 reasoning-like 字段，但保留 `tool_calls` 等协议字段：

```bash
k3lab-eval \
  --history-mode no_reasoning \
  --tasks experiments/tasks/repo_patch.jsonl \
  --out runs/model-no-reasoning.jsonl
```

这样比较的是 history 中 reasoning state 的作用，而不是因为破坏 tool protocol 导致 API 直接报错。

比较 success、invalid calls、steps、tokens。结果只对具体模型/API 版本成立。

## 6. Harness Generalization

v0.3 提供三套**语义等价、schema 不同**的工具接口：

| Variant | 例子 |
| --- | --- |
| canonical | `read_file`, `write_file`, `run_public_tests` |
| compact | `read`, `write`, `test` |
| alternate | `inspect_file`, `update_file`, `check_visible_tests` |

运行：

```bash
for variant in canonical compact alternate; do
  k3lab-eval \
    --harness-variant "$variant" \
    --tasks experiments/tasks/repo_patch.jsonl \
    --out "runs/model-$variant.jsonl"
done
```

如果模型只在 canonical 上稳定，换等价 schema 就明显掉，说明它依赖 harness-specific pattern。

后面做训练时可以：

- train: canonical + compact；
- held-out test: alternate。

这样就有真正的 harness generalization 指标。

## 7. Reward Ablation

先比较行为，再训练：

```text
R1 = task_success
R2 = task_success - lambda_invalid * invalid_calls
R3 = task_success - lambda_invalid * invalid_calls - lambda_step * tool_steps
```

如果后面做 RL，还要同时报告 held-out success 和 hacking rate，不能只报告训练 reward。

## 8. Verified Trajectory → SFT Data

v0.3 可以直接把 secure verifier 通过的 rollout 导出成 SFT JSONL：

```bash
k3lab-export-sft \
  runs/kimi.jsonl runs/qwen.jsonl \
  --out data/verified_sft.jsonl
```

只导出 `metadata.verifier.success == true` 的轨迹。导出内容包括：

```text
system
user
assistant(tool call)
tool(observation)
...
assistant(final)
```

持久化轨迹默认移除 reasoning-like 字段，所以这里得到的是可见交互轨迹，不把私有 reasoning trace 当训练数据。

这一步把项目从「评测框架」接到了真正的 post-training pipeline：

```text
rollout
  ↓
secure verifier
  ↓
successful trajectories
  ↓
SFT dataset
  ↓
SFT
  ↓
RL
```

## 9. 下一步：SFT → RL

不训练 2.8T K3，而选成本可控的小型开源模型：

| Model | In-harness success | Held-out harness | Hacking rate | Invalid calls | Avg steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | TBD | TBD | TBD | TBD | TBD |
| SFT | TBD | TBD | TBD | TBD | TBD |
| SFT + RL | TBD | TBD | TBD | TBD | TBD |

后续需要补：

- LoRA/QLoRA SFT recipe；
- verifiable reward 接 GRPO；
- held-out harness evaluation；
- length / KL / reward hacking 监控。

## 10. AutoResearch

AutoResearch 放在训练闭环之后。第一版只允许修改 reward weights、prompt、context policy、tool descriptions、max steps，并自动运行评测和分析失败。

它不能修改 held-out verifier 或答案，否则自动研究会退化成自动刷分。

---

当前状态：**v0.3 已包含 coding environment、secure verifier、reward-hacking case study、history ablation、harness variants、跨模型汇总以及 verified trajectory → SFT 数据导出。真实多模型结果和 SFT/RL 训练结果仍待实测。**
