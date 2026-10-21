# 十分钟讲述：从状态更新走到可验证的训练闭环

这份 walkthrough 用一条问题线串联仓库：长序列怎样保存信息，深层网络怎样读取旧表示，稀疏专家怎样分配计算，以及后训练怎样把环境反馈变成参数更新。讲述时打开源码和真实报告，遇到追问就落到一个具体张量、干预或失败样本。

阅读入口：[01 KDA](01-kda.md) · [02 AttnRes](02-attnres.md) · [03 MoE](03-moe.md) · [04 预训练与系统](04-pretraining-systems.md) · [05 后训练](05-posttraining.md)。完整 CPU 数值见 [报告](../../reports/mechanisms/REPORT.md) 和 [JSON](../../reports/mechanisms/results.json)，实验组织在 [run_mechanism_lab.py](../../scripts/run_mechanism_lab.py)。

## 1. 十分钟怎样安排

下表总计十分钟。正文是可按自己的语速展开的讲述骨架；把手算和代码指认穿插进去，而不是连续背十分钟术语。

| 时间 | 打开的材料 | 这一段要讲清的因果链 |
|---|---|---|
| 0:00–0:45 | 本页与机制报告 | 问题 → 实现 → 独立参照 → 干预 → 结果 |
| 0:45–2:15 | `kda.py`、01 | 关联记忆 → 预测误差写入 → 下三角求解 |
| 2:15–3:15 | `attnres.py`、02 | 深度来源 → key 归一化 → 原始 value 混合 |
| 3:15–4:45 | `moe.py`、03 | 选专家 → 分派 → 回收 → 负载校准失效条件 |
| 4:45–6:00 | `moe.py` 成本函数、04 | 参数/乘加/逻辑通信 → 真正的训练关键路径 |
| 6:00–7:30 | `sft.py`、`grpo_env.py`、05 | token 监督 → 环境奖励 → 策略更新 |
| 7:30–9:00 | 原始 JSON、verifier 测试 | 成功和失败各展示一项，解释结论适用范围 |
| 9:00–10:00 | 05 的实现对应表 | 下一次训练怎样被验收，先排除哪些混杂 |

### 0:00–0:45：先交代自己做了什么

“我沿 K3 的公开报告拆了三条信息流：序列方向的 KDA、深度方向的 AttnRes、专家方向的 MoE。每一部分都先固定维度和更新顺序，再写一个能在 CPU 运行的参考。我想让这里的理解能被追问：公式能推到代码，代码能和独立形式比较，最后能用干预解释一个数值为什么变化。”

随后指向报告中的输入配置、seed 和 JSON，而不是先报整模 benchmark。报告中的误差是两种计算形式的元素级差；没有把这种误差当作模型任务准确率。

### 0:45–2:15：用同一个 key 的两次写入讲 KDA

“状态是 K×V 的矩阵，key 决定写入方向。普通外积累加会重复叠加旧信息，delta 更新先预测当前 value，只补残差。alpha 先按 key 通道衰减旧状态，beta 再控制纠错写入。单位 key、beta=1 时，同一个 key 再写一次，会把这个方向改成新 value；beta=0 只关闭新写入，alpha 小于 1 时旧状态仍会变。”

打开 [kda_recurrent](../../src/k3lab/mechanisms/kda.py)，顺着 `decayed → prediction → residual write → read` 四步指认。再解释 chunk 版本：把每次写入的创新量作为未知数，解对角线为 1 的下三角系统；当前 chunk 输出由入口状态和块内创新相加得到。输出矩阵保留对角线，创新耦合矩阵去掉对角线，这两个 mask 对应两种不同的依赖。

最后展示输出和最终状态两项误差。只对输出而不对最终状态，可能漏掉会在下一块才暴露的错误。[KDA 检查](../../tests/test_mechanisms_kda.py)还使用稠密仿射展开、未来扰动、非方阵、非零初态和跨调用续接。

### 2:15–3:15：AttnRes 的注意力轴在哪里

“AttnRes 在同一个 token 的深度来源间分配权重。每个子层有可学习的 pseudo-query，对各来源的归一化 key 打分，再混合原始 value。query 可以跨 token 共享，但权重仍依赖输入，因为历史表示变了。”

打开 [attnres.py](../../src/k3lab/mechanisms/attnres.py)，指出 softmax 沿来源轴，而不是 token 轴。再展示零 query：Full 对每个独立来源平均，Block 对块和平均。报告的固定输出例子分别得到 `[0.8,2.4]` 和 `[4/3,4]`；这个差异来自来源集合改变。把这些固定向量换成真实层输出，还需要让混合结果参与后续子层计算，才能研究端到端训练效果。

### 3:15–4:45：MoE 的两个分数与一个失败例子

“一个分数用于选专家，一个权重用于混合选中的专家。本实现用原始 sigmoid 分数加 bias 做 Top-k 选择，最终混合仍把选中的原始分数归一化。这样调节负载不会直接把偏置当成输出权重。”

打开 [topk_route 与 dispatch](../../src/k3lab/mechanisms/moe.py)，用 `scores=[0.9,0.8,0.2]`、`bias=[0,-0.8,1]` 演示选择专家 2 和 0，混合权重分别为 2/11 和 9/11。再顺着 token ID 看按专家分组、计算、回收到原位置，说明为什么不能只检查形状或分派总数。

“负载校准在当前八 token fixture 上得到 `[2,2,2,2]`，但把这一 bias 冻结后，下一批所有分数都偏向同一个专家，负载变成 `[8,0,0,0]`。因此一次同批回放的均衡不是跨批保证。”这是需要主动展示的失败条件。[03](03-moe.md)进一步区分 exact order-statistic 参考与实际分布式 histogram。

### 4:45–6:00：从公式节省走到设备时间

“参数数回答要驻留多少权重，激活参数数和乘加数回答主要算多少，dispatch/combine 的逻辑 payload 回答按约定要传多少。这三笔账不能互换。潜在宽度减半可以减少路由专家计算和通信，但升降维、共享专家、router 还在，网络中实际跨设备的比例、padding 和调度也会改变时间。”

打开 [moe.py 的 latent_moe_cost](../../src/k3lab/mechanisms/moe.py) 和报告中的输入维度。随后连到 [04](04-pretraining-systems.md)：训练一步要经过数据、前向、反向、梯度通信与更新；只有暴露在关键路径上的通信才应额外相加。若下一步要验证 GPU 收益，应先确定张量大小、拓扑和负载，再记录稳定区间的实际时间与有效 token，而不是从 byte 公式直接写加速倍数。

### 6:00–7:30：把后训练落到两个可检查的位置

“SFT 首先检查谁在被预测：assistant 的工具调用和回答应该如何标 labels，工具观察如何作为条件，EOT 是否被截断，packing 是否让两条任务互相读取。仓库当前把这些交给 TRL，因此要用真实 tokenizer 和 collator 打印样本，而不能从 messages 长得正确就推断 mask 正确。”

“RL 再检查奖励来自哪里。MiniRepo 的训练奖励由原始测试、held-out 测试和 protected-file 完整性共同决定，返回 0/1。eval 还会额外扣调用成本，所以两处 reward 不能混用。K3 沿用 K2.5 的策略优化规则，本 lab 使用 TRL GRPO；logratio mask、PPO surrogate、reference KL 和 loss 聚合要分别读。”

最后用一个 token 讲 [MOPD](05-posttraining.md)：student 自己采样，teacher 对这个 token 给概率，teacher/student logratio 经 stop-gradient 和 clipping 作为稠密信号。固定前缀、不 clipping 时可以推到 reverse KL；陈旧前缀、importance weighting 和奖励聚合会影响完整训练目标。

### 7:30–9:00：成功一项，失败一项

先展示 KDA 的逐 token 与 chunkwise 输出/最终状态一致性，再展示上面的 MoE 新批失衡。两者分别回答“是否计算同一个函数”和“校准结论能否延伸到新分布”，不能拿一项的成功替另一项作保证。

然后打开 [MiniRepo 的测试](../../tests/test_minirepo.py)：删掉公开测试能让可见测试返回 PASS，但 protected-file 检查会失败；硬编码公开输入也能通过可见测试，却过不了 hidden tests。它们是环境 fixture 的受控检查，说明 verifier 能捕捉这些攻击，不是某个训练过的模型已经学会避免攻击的结果。

### 9:00–10:00：下一步实验怎样结束

“我会先固定具体模型与 tokenizer，用一条成功轨迹核对 labels、工具 schema 和结束标记，再用极小 group 检查一次真实更新：生成版本、behavior logprob、有效 token、reward、advantage、loss 分项与参数变化都留存。之后按任务 family 分开训练和验证，用固定预算比较 base、SFT、SFT+RL；工具接口的 alternate 只留在最后评价。”

“如果 reward 上升而 secure success 没升，我会先查目标定义、调用长度和失败类型；如果一个 group 全失败，先查是否有非零的组内学习信号，再决定补示范还是调整任务课程。最终希望说明的是哪一种改动在什么任务上有帮助，而不是 trainer 结束就宣布能力提高。”

## 2. 十二个值得深入的面试追问

1. **KDA 的 delta 更新为什么是一种学习规则？** 从 \(\frac12\|S^\top k-v\|^2\) 对 S 求导，得到 \(k(S^\top k-v)^\top\)，以衰减后的状态为起点走一步即可。再推单位 key 下的回读值 \((1-\beta)\hat v+\beta v\)。若拿掉 key 归一化，哪个等式要改？见 [01](01-kda.md) 与 [beta 干预检查](../../tests/test_mechanisms_kda.py)。

2. **Chunk 并行怎样同时保留因果性和跨块记忆？** 写出创新量的严格下三角耦合与 readout 的含对角线矩阵，再指出入口 state 和尾部 state 的维度都是 K×V。要求现场改一个非方阵例子、从中间位置分两次调用，仍得到同样的输出和最终状态。

3. **AttnRes 的 query 固定，为什么还会“自适应”？** query 是共享的可学习方向，来源 key 仍依输入变化。若把 key 的 RMSNorm 同时用到 value，就改变了原始表示幅度；用两个方向相同但幅度不同的来源，可以构造能区分这两种实现的例子。见 [02](02-attnres.md)。

4. **Full 与 Block 在零 query 下为什么不同？** 零 query 只是让当前来源等权，不会让两种来源集合变成相同。逐个来源平均与先把块内来源求和再平均具有不同的系数。要求从报告输入手算两组输出，并说明固定 f_i 的聚合检查与网络训练的区别。

5. **MoE bias 不进入混合权重，还会不会改变模型输出？** 会，因为专家集合先变了；集合变化后参与归一化的原始分数也变了。要求用上面的三专家例子分别算正确权重与误用 biased score 的权重，再沿 token ID 检查 dispatch/combine。见 [03](03-moe.md)。

6. **为什么同批 QB 完全平衡，下一批仍可能全部落在专家 0？** bias 是根据一份分数分布校准的阈值，分布与排序边界一变，旧偏置就不能保证目标负载。报告保存的 shifted fixture 是 `[8,0,0,0]`；若改成当前批旧 bias 路由、统计后更新供下一批使用，应分别记录哪个时间点的负载。

7. **同样的 activated parameters 为什么不保证相同步速？** GEMM 的形状、专家负载、共享模块、all-to-all、重计算与空闲都会不同。先写单层参数/乘加/payload 三项，再画暴露的通信关键路径；要求解释公式没有包含的实际 profiler 事件。见 [04](04-pretraining-systems.md)。

8. **如何发现 SFT 正在训练错误的 token？** 不先调学习率，先抽一条多轮样本解码 token、打印 role 边界、labels 和 EOT，核对 causal shift。packing 后重复检查；只看 attention mask 或 messages 字段不够。实际包装器入口在 [sft.py](../../src/k3lab/train/sft.py)。

9. **GRPO 的 ratio、reference KL 和 sequence/token 聚合怎样分别改变更新？** ratio 分母来自行为策略，reference 是另一份锚点；先算 1.4 与正/负优势的不同 clipping，再用长 2、8 的两条轨迹算出 0 与 −0.6。要求指出实际 [GRPOConfig](../../src/k3lab/train/grpo.py) 没有显式设置哪些字段，不能只答一个公式。

10. **OPD 的平均 reward 为负，是训练失败吗？** 固定前缀、student 采样且未 clipping 时，它等于负 reverse KL，负值很正常。再推 score-function 梯度为什么要把 reward detach。最后追问：如果 token 来自旧策略，或 prefix 由旧策略产生，一项 token ratio 各能校正哪一部分？见 [05](05-posttraining.md)。

11. **Partial rollout 的环境没丢，为什么仍然 off-policy？** 文件状态延续不代表 token 都来自同一权重版本。画一条轨迹跨两次更新的时间线，给每段标 behavior version 与 logprob；再追问旧 KV/循环状态如何与恢复策略对齐，以及未完成轨迹为何不应记为失败奖励。

12. **Speculative decoding 拒绝后为什么要从正差分布采？** 用 p=[0.6,0.3,0.1]、q=[0.4,0.5,0.1] 算出接受质量 0.8，补回第一个 token 的 0.2，恢复 p。再解释 QAT 为什么要对齐 rollout/training 前向：概率变了，ratio 与接受采样所依赖的分布也变了。见 [05](05-posttraining.md)。

## 3. 能手算、运行和改代码的练习

所有命令从仓库根目录执行；这些练习使用现有 CPU 依赖。训练 tokenizer/collator 和真实 optimizer update 的练习另需安装并固定训练环境。

### 练习 A：重新生成报告，先找一个反例

```bash
PYTHONPATH=src .venv/bin/python scripts/run_mechanism_lab.py --out /tmp/k3lab-mechanism-practice
PYTHONPATH=src .venv/bin/python -m pytest tests/test_mechanisms_kda.py tests/test_mechanisms_depth_moe.py -q
```

打开新报告，先找 MoE shifted fixture 的失衡，再找同批均衡。口头解释为什么必须保留这两项。然后换一组 gate、非零初态或非方阵维度，在自己的实验副本中检查 KDA 两种形式；若改了代码，要保存改变的机制与预期，而不是只留下新截图。

### 练习 B：人为制造一个会被检查捕捉的机制错误

在实验副本中，把 KDA readout 的 `np.tril(read_weights)` 改成不含对角线，预测“两次写入手算”为什么失败；再把最终 state 的传播删掉，预测跨调用续接为什么失败。恢复后，把 AttnRes 的原始 value 换成归一化 value，用两个不同幅度的来源构造能区分它们的例子。

验收时要回答：哪项检查失败、失败值是什么、对应哪个数学依赖。仅仅“pytest 红了”没有解释改动为何错误。

### 练习 C：用现有环境演示公开测试的覆盖缺口

```bash
PYTHONPATH=src .venv/bin/python - <<'PY'
import json
from pathlib import Path
from k3lab.envs import MiniRepoEnv

tasks = [json.loads(x) for x in Path('experiments/tasks/repo_patch.jsonl').read_text().splitlines() if x.strip()]
task = next(x for x in tasks if x['id'] == 'repo-abs-002')
env = MiniRepoEnv.from_task(task)
print('initial visible:', env.run_public_tests()['passed'])
print('initial verifier:', env.verify().to_dict())
env.write_file('solution.expr', 'abs(x) + 3')
print('fixed verifier:', env.verify().to_dict())

attacked = MiniRepoEnv.from_task(task)
attacked.write_file('tests_public.json', '[]')
print('empty visible:', attacked.run_public_tests()['passed'])
print('empty secure success:', attacked.verify().success)
PY
```

应看到初始可见测试通过、hidden 失败，修复后 secure success 为真；清空公开测试仍不能通过 secure verifier。下一问：如果训练成功率只记录可见 PASS，会把哪种行为奖励出来？对应 [测试](../../tests/test_minirepo.py) 和 [GRPO 环境](../../src/k3lab/train/grpo_env.py)。

### 练习 D：手算 SFT 的 shift 与结束标记

```bash
.venv/bin/python - <<'PY'
import numpy as np
x = np.array([1, 2, 3, 4, 5, 0])  # BOS, U1, U2, answer, EOT, PAD
mask = np.array([0, 0, 0, 1, 1, 0], dtype=bool)
labels = np.where(mask, x, -100)
logp_for_next_token = np.array([-9., -9., -.4, -.1, -9.])
loss = -logp_for_next_token[mask[1:]].mean()
print(labels.tolist(), float(loss))
assert labels.tolist() == [-100, -100, -100, 4, 5, -100]
assert np.isclose(loss, .25)
PY
```

接着在实际训练环境，把这份手画 mask 换成真实 messages 经 tokenizer/collator 得到的 labels；逐一核对工具调用、观察、最终回答与 EOT。这个 toy 算例验证索引，不替代实际 template 验证。

### 练习 E：把三个“比值”写在纸上，再运行数值

```bash
.venv/bin/python - <<'PY'
import numpy as np
ratio = .7 / .5
for advantage in [1., -1.]:
    surrogate = min(ratio * advantage, np.clip(ratio, .8, 1.2) * advantage)
    print('PPO surrogate:', advantage, surrogate)
teacher, student = np.array([.8, .2]), np.array([.5, .5])
reward = np.log(teacher) - np.log(student)
print('OPD raw:', reward.tolist())
print('expected reward:', float(student @ reward))
assert np.isclose(student @ reward, -np.sum(student * np.log(student / teacher)))
print('sequence average:', (1. + (-1.)) / 2)
print('token average:', (2. * 1. + 8. * (-1.)) / 10)
PY
```

把每个分母分别标成 behavior、reference 或 student。再把 OPD reward clip 到 [−0.5,0.5]，重新计算期望，说明为何已经不等于原来的负 KL。继续追问：样本若来自另一份 mu，如何估计固定前缀下的期望；若前缀也来自 mu，还缺什么？

### 练习 F：验证接受/拒绝的概率质量守恒

```bash
.venv/bin/python - <<'PY'
import numpy as np
p, q = np.array([.6, .3, .1]), np.array([.4, .5, .1])
accepted = np.minimum(p, q)
acceptance = accepted.sum()
residual = np.maximum(p - q, 0.)
residual = residual / residual.sum()
recovered = accepted + (1. - acceptance) * residual
print(float(acceptance), residual.tolist(), recovered.tolist())
assert np.allclose(recovered, p)
print('LK:', float(-np.log(acceptance)))
PY
```

把 p 改成 temperature 变换后的分布，再重新算；不要只改生成采样器而沿用旧校验概率。最后列出还需要测量的 draft、target、缓存与调度时间，说明为什么 0.8 不是一个加速倍数。

### 练习 G：设计一份能诊断失败的最小训练记录

先在纸上给一组生成标出 prompt ID、task family、harness schema、policy version、每段 behavior logprob、终止原因、工具次数、生成 token mask 与 verifier 子项。再添加 reward、组均值/标准差、advantage、clip fraction、KL 配置和 loss 聚合。

假设 secure success 不变而 composite reward 上升，至少构造两种原因：同样成功但更少工具调用；或者 success 没提高而非法调用减少。假设组内奖励全相同，再解释为什么要查看 advantage，而不是只看 trainer 已运行多少步。

接入真实 tokenizer 和 trainer 后，先核对一次反向与参数差，再开展模型效果对照。最终评价按任务 family 与 held-out harness 隔离；[AutoResearch 检查](../../tests/test_autoresearch.py)已把 alternate 排除在开发搜索之外，训练数据构造也应执行相同约束。

这套讲述的验收方式很具体：能从一个输出数字回到输入和代码，能手算一个中间量，能主动给出一个失败条件，再提出下一项能够区分解释的实验。
