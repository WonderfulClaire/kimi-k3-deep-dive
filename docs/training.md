# 八、从 Verified Trajectory 到 SFT / GRPO

> 这部分是本仓库真正的「后训练」入口。K3 本身不作为本地训练对象；实验默认用成本可控的小型开源模型。训练脚本基于 Hugging Face TRL + PEFT，训练依赖单独放在 `.[train]`。

## 1. 安装

```bash
pip install -e ".[train]"
```

普通的 harness / verifier / evaluation 不需要安装训练依赖。

## 2. 先采 rollout，再做 SFT

第一步不是直接 train，而是先让不同 endpoint 在同一环境里跑：

```bash
k3lab-eval \
  --tasks experiments/tasks/repo_patch.jsonl \
  --out runs/model.jsonl
```

只有 secure verifier 通过的轨迹进入 SFT 数据：

```bash
k3lab-export-sft \
  runs/model.jsonl \
  --out data/verified_sft.jsonl
```

这样得到的数据结构是 conversational `messages`：

```text
system
user
assistant(tool call)
tool(observation)
...
assistant(final)
```

持久化轨迹默认已经移除 reasoning-like 字段，所以训练的是可见交互轨迹，而不是私有 reasoning trace。

## 3. LoRA SFT

```bash
k3lab-train-sft \
  --model Qwen/Qwen3-0.6B \
  --data data/verified_sft.jsonl \
  --output-dir outputs/qwen3-sft \
  --epochs 1 \
  --learning-rate 2e-4 \
  --bf16
```

脚本使用：

- TRL `SFTTrainer`；
- PEFT LoRA；
- `target_modules="all-linear"`；
- gradient checkpointing；
- conversational `messages` dataset。

如果模型 chat template 支持 assistant mask，可以额外打开：

```bash
--assistant-only-loss
```

不要默认对所有模型打开：TRL 要求对应 chat template 能标出 assistant generation 区域。

## 4. Agentic GRPO：Reward 由环境最终状态给

v0.4 新增 `MiniRepoGRPOEnv`。它不是对最终文本做 judge，而是直接让 environment 拥有 reward：

```text
policy
  ↓
tool calls
  ↓
MiniRepo environment state
  ↓
original public tests
+ held-out tests
+ integrity check
  ↓
get_reward() ∈ {0, 1}
```

训练：

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

显存允许时可加：

```bash
--use-vllm
```

脚本使用 TRL 当前的 `environment_factory` agent-training 接口：每条 rollout 都拿到独立 environment，public methods 暴露成 tools，而 `reset()` / `get_reward()` 是 lifecycle 方法。

## 5. 为什么这个 GRPO reward 比「让 LLM judge 打分」更适合第一阶段

MiniRepo 是可验证任务，所以没必要先引入 judge 噪声。

这里故意把 policy 能看到的 `run_public_tests` 和真正训练 reward 分开：

```text
visible public test signal
          ≠
environment-owned secure reward
```

模型即使删掉 public tests，让 `run_public_tests` 显示 PASS，`get_reward()` 仍会因为 integrity / held-out verifier 返回 0。

这正好可以测：

- reward hacking rate；
- public success vs secure success gap；
- SFT 后是否更少 hack；
- GRPO 后是否提高 secure success；
- reward 是否伴随 tool steps / output length 异常增长。

## 6. 最小实验矩阵

| Model | Training | Canonical | Compact | Alternate (held-out) | Hacking rate |
| --- | --- | ---: | ---: | ---: | ---: |
| Base | none | TBD | TBD | TBD | TBD |
| Base + SFT | verified trajectories | TBD | TBD | TBD | TBD |
| Base + SFT + GRPO | secure env reward | TBD | TBD | TBD | TBD |

最重要的是 **Alternate 不参与训练**。否则所谓 harness generalization 没有意义。

## 7. 当前限制

目前 MiniRepo 是安全、轻量的 synthetic coding environment，目的是先把研究闭环做对，不代表 SWE-bench 级真实软件工程。

后续升级顺序：

1. 增加更多 task family；
2. 加长 multi-turn trajectory；
3. 增加 harness randomization；
4. 接容器化 coding environment；
5. 再比较 rule verifier / environment reward / GRM；
6. 最后做 AutoResearch 自动配置搜索。

---

参考接口：

- TRL SFTTrainer: https://huggingface.co/docs/trl/sft_trainer
- TRL GRPO agent training / environment_factory: https://huggingface.co/docs/trl/grpo_trainer
- PEFT LoRA: https://huggingface.co/docs/peft/en/package_reference/lora
