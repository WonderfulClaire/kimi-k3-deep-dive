#!/usr/bin/env python3
"""Rebuild deterministic float64 mechanism results and a Chinese walkthrough.

Run from the repository root:
    PYTHONPATH=src .venv/bin/python scripts/run_mechanism_lab.py --out reports/mechanisms

No model weights, paid endpoints, GPU runtime, or benchmark downloads are used.
The JSON intentionally contains neither wall-clock timings nor timestamps.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path
import shlex

import numpy as np

from k3lab.mechanisms.attnres import block_attnres, depth_attention, full_attnres
from k3lab.mechanisms.kda import kda_chunkwise, kda_recurrent
from k3lab.mechanisms.moe import (
    heuristic_balance_bias,
    latent_moe_cost,
    load_statistics,
    moe_linear_dispatch,
    moe_linear_reference,
    quantile_balance_bias,
    topk_route,
)


SCHEMA_VERSION = "1.0.0"
SEEDS = (0, 1, 2)
CHUNK_SIZES = (1, 4, 8, 64)


def max_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)), initial=0.0))


def kda_experiments() -> dict:
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        q = rng.normal(size=(37, 4))
        k = rng.normal(size=(37, 4))
        v = rng.normal(size=(37, 3))
        alpha = rng.uniform(0.7, 1.0, size=(37, 4))
        beta = rng.uniform(0.0, 1.0, size=37)
        initial = rng.normal(size=(4, 3))
        recurrent, final = kda_recurrent(q, k, v, alpha, beta, initial_state=initial)
        comparisons = []
        for chunk_size in CHUNK_SIZES:
            chunked, state = kda_chunkwise(q, k, v, alpha, beta, chunk_size=chunk_size, initial_state=initial)
            comparisons.append({
                "chunk_size": chunk_size,
                "output_max_abs_error": max_error(chunked, recurrent),
                "final_state_max_abs_error": max_error(state, final),
            })

        # Perturb every input strictly after prefix positions 0..18. Earlier
        # output positions must remain identical under both representations.
        qp, kp, vp, ap, bp = [x.copy() for x in (q, k, v, alpha, beta)]
        qp[19:] += rng.normal(size=qp[19:].shape) * 4
        kp[19:] -= rng.normal(size=kp[19:].shape) * 3
        vp[19:] += 20
        ap[19:] = 0.2
        bp[19:] = 0.9
        future_changed, _ = kda_recurrent(qp, kp, vp, ap, bp, initial_state=initial)
        causal_chunks = []
        for chunk_size in CHUNK_SIZES:
            changed, _ = kda_chunkwise(qp, kp, vp, ap, bp, chunk_size=chunk_size, initial_state=initial)
            original, _ = kda_chunkwise(q, k, v, alpha, beta, chunk_size=chunk_size, initial_state=initial)
            causal_chunks.append({
                "chunk_size": chunk_size,
                "prefix_output_max_abs_change": max_error(changed[:19], original[:19]),
            })
        rows.append({
            "seed": seed,
            "comparisons": comparisons,
            "future_perturbation": {
                "first_changed_token_zero_based": 19,
                "recurrent_prefix_output_max_abs_change": max_error(future_changed[:19], recurrent[:19]),
                "recurrent_suffix_output_max_abs_change": max_error(future_changed[19:], recurrent[19:]),
                "chunkwise": causal_chunks,
            },
        })

    keys = np.array([[1., 0.], [1., 0.]])
    values = np.array([[1., 2.], [7., 8.]])
    ones = np.ones((2, 2))
    overwrite, overwrite_state = kda_recurrent(keys, keys, values, ones, np.ones(2))
    no_write_initial = np.array([[2., 3.], [4., 5.]])
    no_write, no_write_state = kda_recurrent(keys[:1], keys[:1], values[:1] * 100, ones[:1], np.zeros(1), initial_state=no_write_initial)
    forget_initial = np.array([[2., 3.], [4., 5.]])
    _, forget_state = kda_recurrent(
        np.array([[1., 0.]]), np.array([[1., 0.]]), np.zeros((1, 2)),
        np.array([[0., 1.]]), np.zeros(1), initial_state=forget_initial,
    )
    return {
        "random_equivalence": rows,
        "interventions": {
            "same_key_beta_one_overwrites": {
                "key": [1., 0.], "values": values.tolist(),
                "outputs": overwrite.tolist(), "final_state": overwrite_state.tolist(),
                "second_output_vs_second_value_max_abs_error": max_error(overwrite[1], values[1]),
            },
            "beta_zero_does_not_write": {
                "alpha": [1., 1.], "initial_state": no_write_initial.tolist(),
                "attempted_value": (values[0] * 100).tolist(),
                "output": no_write.tolist(), "final_state": no_write_state.tolist(),
                "state_vs_initial_max_abs_error": max_error(no_write_state, no_write_initial),
            },
            "alpha_channel_forgetting": {
                "alpha": [0., 1.], "beta": 0., "initial_state": forget_initial.tolist(),
                "final_state": forget_state.tolist(),
                "forgotten_row_max_abs_value": max_error(forget_state[0], np.zeros(2)),
                "preserved_row_max_abs_change": max_error(forget_state[1], forget_initial[1]),
            },
        },
    }


def attnres_experiments() -> dict:
    sources = np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]])
    uniform, uniform_weights = depth_attention(sources, np.zeros(2))
    selected, selected_weights = depth_attention(sources, np.array([8., 0.]))
    embedding = np.zeros(2)
    outputs = np.array([[2., 0.], [2., 0.], [0., 6.], [0., 6.]])
    full, full_weights = full_attnres(embedding, outputs, np.zeros(2))
    block, block_weights = block_attnres(embedding, outputs, np.zeros(2), block_size=2)
    return {
        "query_intervention": {
            "sources": sources.tolist(),
            "zero_query": [0., 0.], "zero_query_weights": uniform_weights.tolist(),
            "zero_query_output": uniform.tolist(),
            "uniform_weight_max_abs_error": max_error(uniform_weights, np.full(4, .25)),
            "target_query": [8., 0.], "target_source_index": 0,
            "target_query_weights": selected_weights.tolist(),
            "target_query_output": selected.tolist(),
            "target_source_weight": float(selected_weights[0]),
        },
        "full_vs_block_fixed_outputs": {
            "semantic": "fixed_f_i_outputs_aggregation_not_full_network_training",
            "embedding": embedding.tolist(), "layer_outputs": outputs.tolist(),
            "query": [0., 0.], "block_size_f_i_outputs": 2,
            "full_source_count": len(full_weights), "block_source_count": len(block_weights),
            "full_output": full.tolist(), "block_output": block.tolist(),
            "output_max_abs_difference": max_error(full, block),
        },
    }


def moe_experiments() -> dict:
    scores = np.array([
        [.9, .8, .1, .05], [.88, .3, .2, .1], [.7, .75, .2, .1],
        [.65, .4, .6, .2], [.8, .2, .1, .7], [.55, .5, .4, .3],
        [.6, .59, .58, .57], [.7, .2, .65, .1],
    ])
    raw = topk_route(scores, 1)
    sign_bias = heuristic_balance_bias(raw.loads, step_size=.2)
    qb_bias = quantile_balance_bias(scores, 1)
    rng = np.random.default_rng(0)
    tokens = rng.normal(size=(8, 3))
    expert_weights = rng.normal(size=(4, 3, 2))
    methods = []
    for name, bias in (("raw", np.zeros(4)), ("sign_bias", sign_bias), ("exact_quantile_bias", qb_bias)):
        plan = topk_route(scores, 1, bias)
        output = moe_linear_dispatch(tokens, expert_weights, plan)
        reference = moe_linear_reference(tokens, expert_weights, plan)
        methods.append({
            "method": name, "bias": bias.tolist(), "loads": plan.loads.tolist(),
            "selected_experts": plan.indices.tolist(),
            "mixture_weights": plan.weights.tolist(),
            "load_statistics": load_statistics(plan),
            "dispatch_vs_token_reference_max_abs_error": max_error(output, reference),
        })
    shifted_scores = np.tile([.9, .1, .05, .02], (8, 1))
    shifted = topk_route(shifted_scores, 1, qb_bias)
    return {
        "scores": scores.tolist(), "top_k": 1, "target_load": 2,
        "evaluation_scope": "same_calibration_fixture_replay_for_intervention_only",
        "sign_step_size": .2, "methods": methods,
        "dispatch_fixture": {"tokens": tokens.tolist(), "linear_expert_weights": expert_weights.tolist()},
        "new_batch_distribution_shift": {
            "score_row_repeated_eight_times": [.9, .1, .05, .02],
            "bias_from": "calibration_fixture_exact_quantile",
            "loads_with_frozen_bias": shifted.loads.tolist(),
            "meaning": "one_calibration_update_does_not_guarantee_perfect_balance_on_new_batches",
        },
    }


def cost_experiments() -> dict:
    k3_config = dict(hidden_dim=7168, latent_dim=3584, ffn_dim=3072, num_experts=896, top_k=16, tokens=1, shared_experts=2, dtype_bytes=2)
    k3_cost = latent_moe_cost(**k3_config)
    b, h, dk, dv, recurrent_layers, global_layers, bytes_per_element = 1, 8, 64, 64, 18, 6, 2
    state = recurrent_layers * b * h * dk * dv * bytes_per_element
    hybrid_rows = []
    for seq_len in (1024, 4096, 16384):
        global_cache = global_layers * b * h * seq_len * (dk + dv) * bytes_per_element
        dense_cache = (recurrent_layers + global_layers) * b * h * seq_len * (dk + dv) * bytes_per_element
        hybrid_rows.append({
            "sequence_length": seq_len,
            "recurrent_state_bytes": state,
            "global_kv_bytes": global_cache,
            "hybrid_state_plus_kv_bytes": state + global_cache,
            "all_global_kv_bytes": dense_cache,
            "hybrid_over_all_global_ratio": (state + global_cache) / dense_cache,
        })
    return {
        "k3_like_single_layer_formula": {
            "scope": "simplified_single_layer_parameter_and_logical_payload_formula_not_full_model_or_measurement",
            "public_source": "https://arxiv.org/html/2607.24653v1#S3.T1",
            "assumptions": {
                "routed_expert_input_output_width": 3584,
                "shared_expert_input_output_width": 7168,
                "routed_ffn_intermediate_width": 3072,
                "shared_ffn_intermediate_width": 3072,
                "linear_bias_parameters_counted": False,
                "independent_routing_control_bias_state_counted": False,
                "latent_rmsnorm_gain_parameters_counted": True,
            },
            "config": k3_config, "counts": k3_cost,
        },
        "illustrative_hybrid_attention": {
            "scope": "illustrative_dimensions_not_k3_measured_memory",
            "config": {"batch_size": b, "heads": h, "key_dim": dk, "value_dim": dv,
                       "layers": 24, "recurrent_layers": recurrent_layers, "global_layers": global_layers,
                       "bytes_per_element": bytes_per_element},
            "rows": hybrid_rows,
        },
    }


def build_results() -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "config": {
            "dtype": "float64", "device": "CPU", "kda_tokens": 37,
            "kda_key_dim": 4, "kda_value_dim": 3, "chunk_sizes": list(CHUNK_SIZES),
            "normalize_qk": True, "readout_scale": 1., "source_rmsnorm_epsilon": 1e-8,
            "timing_measured": False,
        },
        "seed": {"values": list(SEEDS), "generator": "numpy.random.default_rng",
                 "scope": "KDA random experiments; MoE dispatch matrices independently use seed 0; other interventions are fixed"},
        "runtime": {"python": platform.python_version(), "numpy": np.__version__,
                    "pytest": importlib.metadata.version("pytest") if importlib.util.find_spec("pytest") else "not installed"},
        "definitions": {
            "max_abs_error": "max(abs(actual-reference)); float64 arithmetic; each output/state is compared separately",
            "kda_state": "[key_dim,value_dim]; decay rows by alpha, predict with normalized key, delta-write with beta, then read",
            "causal_prefix": "token indices 0..18; all q/k/v/alpha/beta inputs at indices >=19 are changed",
            "attnres": "key-only RMSNorm, depth-axis softmax, raw values; fixed supplied f_i outputs, not end-to-end training",
            "routing": "Top-k on raw sigmoid score+bias; mixture weights normalize selected raw scores without bias",
            "qb": "exact per-expert order-statistic calibration, not distributed histogram; same-batch replay is a diagnostic, next-batch balance not guaranteed",
            "madds": "one scalar multiplication-plus-addition pair counts as one MAdd; nonlinear and scheduling work excluded",
            "moe_parameters": "routed experts ell->h->ell; full-width shared experts d->h->d; common intermediate h only; exclude all linear biases and routing-control bias state; include d<->latent, router and latent RMSNorm gain",
            "moe_payload": "logical activation dispatch+combine, every assignment assumed remote; excludes metadata/padding/topology/replication",
            "hybrid_payload": "recurrent states plus full global-layer K/V logical inference payload; excludes training activations, workspace and allocator overhead",
        },
        "results": {"kda": kda_experiments(), "attnres": attnres_experiments(),
                    "moe": moe_experiments(), "cost_ledger": cost_experiments()},
    }


def render_report(data: dict, command: str) -> str:
    r = data["results"]
    kda, attnres, moe, cost = (r[name] for name in ("kda", "attnres", "moe", "cost_ledger"))
    lines = [
        "# 基模机制实验：KDA、AttnRes 与 MoE", "",
        "本报告由 CPU 上的 NumPy float64 实际执行结果生成。随机检验固定 seed=0、1、2；固定干预单独列出，计数公式另列账本。", "",
        "## 运行与约定", "", "```bash", command, "```", "",
        f"依赖版本：Python {data['runtime']['python']}；NumPy {data['runtime']['numpy']}；pytest {data['runtime']['pytest']}。",
        "脚本只依赖 Python 标准库和 NumPy；pytest 用于自动检查。JSON 不包含时间戳或耗时，重复运行可以直接比较结果文件。", "",
        "KDA 的 q/k 逐行 L2 归一化，读出 scale=1；alpha 是每 key 通道的保留率，beta 是写入强度。AttnRes 的 key RMSNorm 使用 epsilon=1e-8，value 保持原值。", "",
        "## KDA：两种形式是否计算同一个函数", "",
        "随机输入长度 37、key 维度 4、value 维度 3，并使用非零初始状态。chunk=1、4、8、64 分别覆盖单步、分块、不同余块以及整段一次求解。误差是元素级绝对差的最大值。", "",
        "| Seed | Chunk | 输出最大绝对误差 | 最终状态最大绝对误差 |", "| ---: | ---: | ---: | ---: |",
    ]
    for row in kda["random_equivalence"]:
        for comparison in row["comparisons"]:
            lines.append(f"| {row['seed']} | {comparison['chunk_size']} | {comparison['output_max_abs_error']:.3e} | {comparison['final_state_max_abs_error']:.3e} |")
    lines += ["", "两种实现分别按 token 递推和块内下三角线性系统求解。输出与最终状态同时接近，说明分块没有遗漏跨块状态；这个结论验证数值等价性，未测 GPU 速度。", "",
              "### 未来扰动与因果性", "", "把索引 19 及之后的全部 q/k/v/alpha/beta 改动，检查索引 0..18。后缀应改变，前缀应保持不变。", "",
              "| Seed | 递推前缀最大变化 | Chunk 前缀最大变化（1 / 4 / 8 / 64） | 递推后缀最大变化 |", "| ---: | ---: | :--- | ---: |"]
    for row in kda["random_equivalence"]:
        causal = row["future_perturbation"]
        chunks = " / ".join(f"{x['prefix_output_max_abs_change']:.3e}" for x in causal["chunkwise"])
        lines.append(f"| {row['seed']} | {causal['recurrent_prefix_output_max_abs_change']:.3e} | {chunks} | {causal['recurrent_suffix_output_max_abs_change']:.3e} |")
    interventions = kda["interventions"]
    overwrite = interventions["same_key_beta_one_overwrites"]
    no_write = interventions["beta_zero_does_not_write"]
    forget = interventions["alpha_channel_forgetting"]
    lines += ["", "### 写入与遗忘的三个固定干预", "",
        f"1. 同 key、alpha=1、beta=1：先写 [1,2]，再写 [7,8]，两步读出为 `{overwrite['outputs']}`。第二步与新值最大误差 {overwrite['second_output_vs_second_value_max_abs_error']:.3e}；delta 更新覆盖该 key 方向的旧预测。",
        f"2. beta=0、alpha=1：即使尝试写入 `{no_write['attempted_value']}`，最终状态仍为 `{no_write['final_state']}`，相对初态误差 {no_write['state_vs_initial_max_abs_error']:.3e}。这是隔离写入开关的干预；alpha 小于 1 时仍可发生遗忘。",
        f"3. alpha=[0,1]、beta=0：初态 `{forget['initial_state']}` 变为 `{forget['final_state']}`。第一行被清除，第二行变化 {forget['preserved_row_max_abs_change']:.3e}；因此 alpha 沿 key 通道控制状态行。", "",
        "## AttnRes：query 选择和 Full / Block 聚合", "",
        f"四个来源为 `{attnres['query_intervention']['sources']}`。零 pseudo-query 得到 `{attnres['query_intervention']['zero_query_weights']}`，相对 1/4 的误差 {attnres['query_intervention']['uniform_weight_max_abs_error']:.3e}。",
        f"改为 query=[8,0]，第 0 来源权重为 {attnres['query_intervention']['target_source_weight']:.9f}；完整权重为 `{attnres['query_intervention']['target_query_weights']}`。固定来源后只改变 query，能够直接检查深度选择。", "",
        f"在另外一个固定 f_i 输出例子中，Full 有 {attnres['full_vs_block_fixed_outputs']['full_source_count']} 个来源，输出 `{attnres['full_vs_block_fixed_outputs']['full_output']}`；块大小 2 的 Block 有 {attnres['full_vs_block_fixed_outputs']['block_source_count']} 个来源，输出 `{attnres['full_vs_block_fixed_outputs']['block_output']}`，最大差 {attnres['full_vs_block_fixed_outputs']['output_max_abs_difference']:.6f}。",
        "这里先给定 embedding 与 f_i 输出，再比较来源集合。真实网络的混合还会改变后续子层输出，本节结果仅描述固定输出下的聚合行为，不是全网络训练指标。", "",
        "## MoE：同批路由干预与逐 token 参考", "",
        "8 个 token、4 个专家、Top-1、目标负载为每专家 2。原始 sigmoid 分数见 results.json；sign bias 步长为 0.2，exact QB 使用逐专家顺序统计量。bias 只改变选择，混合继续使用原始分数。", "",
        "| 方法 | 同批回放负载 | 最大/平均负载 | 与逐 token 参考最大误差 |", "| :--- | :--- | ---: | ---: |",
    ]
    for method in moe["methods"]:
        lines.append(f"| {method['method']} | `{method['loads']}` | {method['load_statistics']['max_over_mean']:.3f} | {method['dispatch_vs_token_reference_max_abs_error']:.3e} |")
    lines += ["", "expert-major dispatch/combine 与逐 token 标量循环使用同一选中集合比较，检查专家分组和回收的 token 身份。线性专家只用于隔离这部分逻辑。", "",
        f"QB 的同批回放展示校准干预；换一批全部偏向专家 0 的固定分数后，冻结该 bias 的负载为 `{moe['new_batch_distribution_shift']['loads_with_frozen_bias']}`。一次校准不保证新批完全平衡；实际训练应当前批使用旧 bias，新 bias 从下一批生效。该 exact 参考也未模拟分布式 histogram。", "",
        "## 成本账本：单层公式和示意缓存", "",
        "### 采用公开 K3 维度的简化单层 MoE 公式", "",
        "使用 d=7168、ell=3584、h=3072、E=896、k=16、2 个共享专家。路由专家为 ell→h→ell，全宽共享专家为 d→h→d；两者仅假设 FFN 中间维度 h 相同。共享专家参数按 3×d×h×2 计算，并且每个 token 都计算完整共享路径。", "",
        "所有线性层 bias 和独立路由均衡 bias 状态均未计入；计入降维、升维、router 权重和潜在 RMSNorm 增益。维度来源：[公开架构表 1](https://arxiv.org/html/2607.24653v1#S3.T1)，共享全宽路径见 [Eq. (11)](https://arxiv.org/html/2607.24653v1#S2.SS3)。这是简化单层公式，不是整模参数复现或实际带宽测量。", "",
        "| 项目 | 公式计数 |", "| :--- | ---: |",
    ]
    counts = cost["k3_like_single_layer_formula"]["counts"]
    for key in ("routed_expert_parameters", "shared_expert_parameters", "projection_parameters", "router_parameters", "norm_parameters", "total_parameters", "active_parameters_per_token", "routed_matrix_madds", "shared_matrix_madds", "projection_matrix_madds", "router_matrix_madds", "matrix_madds", "logical_payload_bytes", "full_width_logical_payload_bytes"):
        lines.append(f"| {key} | {counts[key]:,} |")
    lines += ["", "参数对应驻留与激活规模；MAdds 只统计矩阵乘加对；payload 按单 token、2 字节元素、所有分派都跨设备的 dispatch+combine 计算，排除 metadata、padding、拓扑与复制。三种数字分别回答存多少、算多少、逻辑传多少。", "",
        "### 24 层混合注意力的示意推理状态 payload", "",
        "使用 B=1、H=8、dk=dv=64、18 层循环状态、6 层全局 K/V、每元素 2 字节。循环状态字节数为 18×B×H×dk×dv×2；全局 K/V 为 6×B×H×T×(dk+dv)×2；对照为全部 24 层存全局 K/V。", "",
        "| 序列长度 | 循环状态 MiB | 全局 K/V MiB | 混合合计 MiB | 全全局 K/V MiB | 比例 |", "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in cost["illustrative_hybrid_attention"]["rows"]:
        mib = [row[key] / 2**20 for key in ("recurrent_state_bytes", "global_kv_bytes", "hybrid_state_plus_kv_bytes", "all_global_kv_bytes")]
        lines.append(f"| {row['sequence_length']} | {mib[0]:.3f} | {mib[1]:.3f} | {mib[2]:.3f} | {mib[3]:.3f} | {row['hybrid_over_all_global_ratio']:.6f} |")
    lines += ["", "循环状态随序列长度保持固定，全局 K/V 随 T 增长，所以长序列时占比逐渐由全局层数决定。这组维度是教学示意，不是 K3 实测；账本不包含训练激活、临时工作区、分配器开销或 MLA 的压缩缓存实现。", "",
        "## 继续追问", "",
        "- KDA：为什么 beta=0 仍可能遗忘？alpha 先衰减旧状态；为什么要同时比输出和最终状态？最终状态会成为下一块的输入。",
        "- AttnRes：为什么零 query 的 Full / Block 结果不同？块内先求和，跨来源再做归一化平均；如何落到真实网络？需要让混合结果进入后续子层。",
        "- MoE：为什么 bias 不进权重也能改变输出？专家集合变了；为什么同批均衡不能预测新批？分布与各专家排序边界都可能变化。",
        "- 系统：为什么公式字节减少不能直接宣称加速？还需要测量计算、通信、调度及利用率，本报告没有测量 GPU 速度。", "",
        "数值、输入、配置和定义保存在同目录 `results.json`。机制代码与单元检查分别位于 `src/k3lab/mechanisms/` 和 `tests/`。", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("reports/mechanisms"), help="output directory for results.json and REPORT.md")
    args = parser.parse_args()
    data = build_results()
    output = json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "results.json").write_text(output, encoding="utf-8")
    command = "PYTHONPATH=src .venv/bin/python scripts/run_mechanism_lab.py --out " + shlex.quote(str(args.out))
    (args.out / "REPORT.md").write_text(render_report(data, command), encoding="utf-8")
    print(f"Wrote {args.out / 'results.json'} and {args.out / 'REPORT.md'}")


if __name__ == "__main__":
    main()
