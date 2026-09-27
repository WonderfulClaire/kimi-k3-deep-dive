# 七、Agentic Post-Training Lab：实验设计与复现协议

这页只记录**我们自己能复现的实验**。未跑出的模型结果不提前填数字；基础设施行为则由单元测试和 CI 验证。

## 1. 统一评测协议

同一组 task 固定 system prompt、tool schema、max steps、verifier、retry policy 和 context policy，只替换 model endpoint。这样才能把「模型差异」和「harness 差异」拆开。

首轮记录：

| 字段 | 用途 |
| --- | --- |
| task success | 最终状态是否通过安全 verifier |
| tool calls | agent 行为长度 |
| invalid tool calls | tool-use 稳定性 |
| input/output tokens | 成本与过度思考 |
| verifier components | public / held-out / integrity 分开看 |
| reward components | reward 为什么高或低 |

## 2. Experiment A — Same Harness, Different Models

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

汇总表会给 success、平均 reward、tool calls、invalid calls 和 token 使用量。

## 3. v0.2 新增：MiniRepo coding environment

相比 calculator smoke test，`repo_patch.jsonl` 已经是真正的多步 tool-use：

```text
read SPEC.md
    ↓
inspect solution.expr / tests
    ↓
edit solution.expr
    ↓
run_public_tests
    ↓
final repository state
    ↓
secure verifier
```

环境不执行模型生成的任意 shell/Python，而是让 agent 修改一个安全表达式 DSL。这样能研究 agent 行为和 reward，又不会把 benchmark 变成任意代码执行器。

secure verifier 同时检查：

1. 原始 public tests；
2. held-out tests；
3. protected test files 是否被篡改。

所以 reward 来自**最终环境状态**，不是 agent 自己声称「修好了」。

## 4. Experiment B — Reward Hacking Case Study

v0.2 故意保留了一个 naive public-test signal：

```text
agent can edit tests_public.json
        ↓
run_public_tests trusts current file
        ↓
delete all tests
        ↓
0 / 0 tests => naive PASS
```

但安全 verifier 不信这个 mutable signal：

```text
final solution
   ├── original public tests
   ├── held-out tests
   └── test-file integrity
             ↓
         secure success
```

单元测试已经覆盖两类典型 hacking：

- **删测试**：naive public signal PASS，但 integrity verifier FAIL；
- **hard-code public cases**：public PASS，但 held-out FAIL。

这正是后训练里需要区分的两件事：`reward observable to the policy` 和 `actual task success` 不一定相同。

## 5. Experiment C — History Preservation Ablation

固定任务与工具，做两组：

- Full history：完整 assistant message round-trip；
- Content only：只回传可见 content。

比较 success、invalid calls、steps、tokens。这个结论只对具体模型/API 版本有效，不从单次实验外推成所有 reasoning model 的普遍规律。

## 6. Experiment D — Reward Ablation

先比较行为，再训练：

```text
R1 = task_success
R2 = task_success - lambda_invalid * invalid_calls
R3 = task_success - lambda_invalid * invalid_calls - lambda_step * tool_steps
```

如果后面做 RL，还要同时报告 held-out success 和 hacking rate，不能只报告训练 reward。

## 7. Experiment E — Harness Generalization

采样/训练时随机化：

- tool name / description；
- system prompt wording；
- tool order；
- error message format；
- context compaction policy。

测试放到未见过的组合。如果只在固定 harness 上涨，换 schema 就掉，说明学到的是 harness-specific pattern。

## 8. Experiment F — SFT → RL

不训练 2.8T K3，而选成本可控的小型开源模型：

```text
base model
   ↓
collect + secure-verify successful trajectories
   ↓
SFT
   ↓
same-harness evaluation
   ↓
RL with verifiable/composite reward
   ↓
held-out harness evaluation
```

最低对照：

| Model | In-harness success | Held-out harness | Hacking rate | Invalid calls | Avg steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | TBD | TBD | TBD | TBD | TBD |
| SFT | TBD | TBD | TBD | TBD | TBD |
| SFT + RL | TBD | TBD | TBD | TBD | TBD |

## 9. AutoResearch 放在最后

第一版 AutoResearch 只允许修改 reward weights、prompt、context policy、tool descriptions、max steps，并自动运行评测和分析失败。

它**不能**修改 held-out verifier 或答案，否则自动研究会退化成自动刷分。

---

当前状态：**v0.2 已包含可运行 coding/tool-use environment、held-out verifier、reward-hacking 单元测试和多 run 汇总工具；真实模型对比结果仍待 endpoint 实测后回填。**
