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

输出 success、平均 reward、tool calls、invalid calls 与 token 使用量。聚合键现在包含 **model + history mode + harness variant**，不会再把同一个模型在不同 harness/history 条件下的结果错误混在一起。

## 3. MiniRepo Coding Environment

`repo_patch.jsonl` 是多步 tool-use：

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

环境不执行模型生成的任意 shell/Python，而是让 agent 修改一个安全表达式 DSL。secure verifier 同时检查原始 public tests、held-out tests 和 protected-file integrity。

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

但 secure verifier 不信这个 mutable signal。

单元测试覆盖两类典型 failure：

- 删除 public tests：naive signal PASS，但 integrity FAIL；
- hard-code public cases：public PASS，但 held-out FAIL。

核心是区分：

```text
reward observable to policy  !=  actual task success
```

## 5. History Preservation Ablation

v0.4 已经把 history ablation 做成 CLI 参数。

完整 round-trip：

```bash
k3lab-eval \
  --history-mode full \
  --tasks experiments/tasks/repo_patch.jsonl \
  --out runs/model-full.jsonl
```

移除 reasoning-like history，但保留 tool-call protocol：

```bash
k3lab-eval \
  --history-mode no_reasoning \
  --tasks experiments/tasks/repo_patch.jsonl \
  --out runs/model-no-reasoning.jsonl
```

这样比较的是 reasoning/history state 的作用，而不是把工具协议破坏后再比较。

## 6. Harness Generalization

提供三套**语义等价、schema 不同**的工具接口：

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

训练时可以只见 canonical + compact，把 alternate 留作 held-out harness。

每次 `k3lab-eval` 还会在 JSONL 旁写一个 `*.manifest.json`，记录模型名、task 文件 SHA-256、K3Lab 版本、Python/平台、max steps、history mode 与 harness variant。API key 不会写入 manifest。这样后续表格里的每个数字都能追溯到具体任务字节与执行设置。

## 7. Reward Ablation

先比较行为，再训练：

```text
R1 = task_success
R2 = task_success - lambda_invalid * invalid_calls
R3 = task_success - lambda_invalid * invalid_calls - lambda_step * tool_steps
```

后面做 RL 时同时报告 held-out success 和 hacking rate，不能只报告训练 reward。

## 8. Verified Trajectory → SFT Data

secure verifier 通过的 rollout 可以直接导出：

```bash
k3lab-export-sft \
  runs/kimi.jsonl runs/qwen.jsonl \
  --out data/verified_sft.jsonl
```

只导出 `metadata.verifier.success == true` 的轨迹，保留 system / user / assistant tool call / tool observation / final answer；持久化轨迹默认不保存 reasoning-like 字段。

## 9. LoRA SFT

```bash
pip install -e ".[train]"

k3lab-train-sft \
  --model Qwen/Qwen3-0.6B \
  --data data/verified_sft.jsonl \
  --output-dir outputs/qwen3-sft \
  --epochs 1 \
  --bf16
```

训练脚本使用 TRL SFTTrainer + PEFT LoRA。

## 10. Agentic GRPO

```bash
k3lab-train-grpo \
  --model Qwen/Qwen3-0.6B \
  --tasks experiments/tasks/repo_patch.jsonl \
  --output-dir outputs/qwen3-grpo \
  --steps 100 \
  --num-generations 4 \
  --max-tool-iterations 8 \
  --bf16
```

这里不是用最终文本 judge，而是用 stateful environment 的 `get_reward()`：

```text
policy
  ↓
tool calls
  ↓
environment state
  ↓
original public + held-out + integrity
  ↓
secure reward
```

因此 public-test signal 即使被 hack，训练 reward 仍然可以为 0。

完整训练说明见 [SFT / GRPO Training](training.md)。

## 11. 最小实验矩阵

| Model | Training | Canonical | Compact | Alternate held-out | Hacking rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | none | TBD | TBD | TBD | TBD |
| Base + SFT | verified trajectories | TBD | TBD | TBD | TBD |
| Base + SFT + GRPO | secure env reward | TBD | TBD | TBD | TBD |

## 12. AutoResearch

AutoResearch 放在训练闭环之后。第一版只允许修改 reward weights、prompt、context policy、tool descriptions、max steps，并自动运行评测和分析失败。

它不能修改 held-out verifier 或答案，否则自动研究会退化成自动刷分。

---

当前状态：**v0.4 已包含 coding/tool-use environment、secure verifier、reward-hacking case study、history/harness ablation、run manifest、verified trajectory export、LoRA SFT 与 agentic GRPO 训练入口。真实多模型结果和训练曲线仍待实际 GPU / endpoint 运行后回填。**
