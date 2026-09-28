# 九、Bounded AutoResearch：让自动实验不能改考卷

K3Lab v0.5 增加了一个受约束的 AutoResearch controller。

它不是“让 Agent 随便改仓库然后挑最高分”，而是把自动实验限制在一个明确的研究协议里：

~~~text
allowlisted search space
        ↓
dev tasks only
        ↓
run candidate configs
        ↓
secure verifier success
        ↓
select one candidate
        ↓
held-out tasks + alternate harness
        ↓
evaluate once
        ↓
ledger.json
~~~

## 1. 为什么要做成 bounded

自动科研最危险的不是搜索慢，而是实验者和考官没有隔离。

如果自动循环能修改：

- held-out task；
- expected answer；
- secure verifier；
- reporting rule；
- alternate harness；

那么“自动提升”很容易退化成自动刷 benchmark。

因此当前 controller 只允许搜索四类变量：

- max_steps
- history_mode
- harness_variant，dev 阶段只能 canonical / compact
- prompt_variant，来自仓库内预定义模板

alternate harness 被保留给最终 held-out evaluation，不能进入 dev search。

## 2. 搜索空间

仓库给了一个默认例子：

experiments/autoresearch_space.json

~~~json
{
  "max_steps": [6, 10],
  "history_mode": ["full", "no_reasoning"],
  "harness_variant": ["canonical", "compact"],
  "prompt_variant": ["baseline", "evidence_first"]
}
~~~

如果 search-space JSON 出现非 allowlist 字段，controller 会直接拒绝。

## 3. 运行

先准备 dev task。真正做研究时，再准备一份训练过程从未使用的 held-out task 文件。

~~~bash
export K3LAB_API_KEY="..."
export K3LAB_BASE_URL="https://your-endpoint/v1"
export K3LAB_MODEL="your-model"

k3lab-autoresearch \
  --dev-tasks experiments/tasks/repo_patch.jsonl \
  --heldout-tasks experiments/tasks/repo_patch_heldout.jsonl \
  --space experiments/autoresearch_space.json \
  --out-dir runs/autoresearch-001
~~~

如果暂时没有独立 held-out task，可以省略 --heldout-tasks。此时 controller 只做 dev search，不会伪造 held-out 结果。

## 4. 怎么选 candidate

selection 的第一目标是 secure verifier task success，而不是训练 reward。

~~~text
1. maximize secure_success_rate
2. fewer invalid tool calls
3. fewer total tool calls
~~~

这是故意的。

这个仓库已经有 reward-hacking case：弱 public signal 可能给 exploit 高 reward。因此 AutoResearch 不能再拿同一个弱 reward 当“研究成功”的最终裁判。

## 5. Held-out evaluation 只在选择后发生

candidate search 完成以后，controller 才拿选中的配置去跑 held-out。

并且 held-out harness 固定为 alternate：

~~~text
dev:
  canonical / compact

held-out:
  alternate
~~~

这样可以把两个问题分开：

- 这个配置是否在 dev task 上更好？
- 它是否只记住了 seen harness？

当前 controller 不会根据 held-out 结果继续改候选参数。

## 6. 输出

每个 candidate 都会保存完整 trajectory JSONL。

最终 runs/autoresearch-001/ledger.json 会记录：

- model
- selection rule
- 完整 search space
- dev task path + SHA-256
- 所有 candidate 配置和指标
- 被选中的 candidate
- held-out task path + SHA-256
- held-out harness
- held-out 是否只在 selection 后评估
- held-out result

API key 不会写入 ledger。

## 7. 这和 RL / Agentic Training 怎么接

第一阶段的 AutoResearch 是外层实验控制器：

~~~text
AutoResearch
    ↓ proposes harness/training setting
Agent rollout
    ↓
secure verifier
    ↓
experiment evidence
    ↓
select next fixed config
~~~

后面如果继续扩展，可以把搜索变量增加到：

- SFT data mix
- GRPO group size
- KL coefficient
- reward component weights
- rollout budget

但前提是这些变量被显式 allowlist，并且 held-out verifier / task / reporting code 仍然不可修改。

## 8. 当前边界

v0.5 实现的是“受约束、可审计的自动实验控制”，不是通用 autonomous scientist。

它目前：

- 不自动写训练代码；
- 不自动修改 verifier；
- 不从 held-out 结果继续调参；
- 不声称已经产生真实模型提升；
- 不替代多 seed 实验。

真正的结果必须在实际 endpoint / GPU 上运行以后写进实验报告。
