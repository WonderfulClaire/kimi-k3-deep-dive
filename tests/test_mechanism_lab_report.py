"""Run the public CPU CLI and validate reproducibility and result semantics."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def run_cli(output: Path) -> tuple[dict, str, bytes]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_mechanism_lab.py"), "--out", str(output)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
        timeout=60,
    )
    assert "results.json" in completed.stdout
    raw = (output / "results.json").read_bytes()
    data = json.loads(raw, parse_constant=lambda value: pytest.fail(f"nonfinite JSON: {value}"))
    report = (output / "REPORT.md").read_text(encoding="utf-8")
    return data, report, raw


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    directory = tmp_path_factory.mktemp("mechanism-lab")
    return run_cli(directory / "first"), run_cli(directory / "second")


def assert_finite_json(value):
    if isinstance(value, float):
        assert math.isfinite(value)
    elif isinstance(value, list):
        for item in value:
            assert_finite_json(item)
    elif isinstance(value, dict):
        for item in value.values():
            assert_finite_json(item)


def test_cli_produces_valid_finite_versioned_json_and_repeats_exactly(generated):
    first, second = generated
    data = first[0]
    assert first[2] == second[2]
    assert data == second[0]
    assert data["schema_version"] == "1.0.0"
    assert data["seed"]["values"] == [0, 1, 2]
    assert data["config"]["dtype"] == "float64"
    assert data["config"]["device"] == "CPU"
    assert data["config"]["timing_measured"] is False
    assert data["definitions"]
    assert set(data["results"]) == {"kda", "attnres", "moe", "cost_ledger"}
    assert_finite_json(data)


def test_random_equivalence_causality_and_interventions_are_real_values(generated):
    kda = generated[0][0]["results"]["kda"]
    assert [row["seed"] for row in kda["random_equivalence"]] == [0, 1, 2]
    for row in kda["random_equivalence"]:
        assert [x["chunk_size"] for x in row["comparisons"]] == [1, 4, 8, 64]
        for comparison in row["comparisons"]:
            assert comparison["output_max_abs_error"] < 1e-11
            assert comparison["final_state_max_abs_error"] < 1e-11
        causal = row["future_perturbation"]
        assert causal["first_changed_token_zero_based"] == 19
        assert causal["recurrent_prefix_output_max_abs_change"] < 1e-11
        assert causal["recurrent_suffix_output_max_abs_change"] > 1
        for chunk in causal["chunkwise"]:
            assert chunk["prefix_output_max_abs_change"] < 1e-11
    interventions = kda["interventions"]
    assert interventions["same_key_beta_one_overwrites"]["outputs"] == [[1, 2], [7, 8]]
    assert interventions["beta_zero_does_not_write"]["state_vs_initial_max_abs_error"] == 0
    assert interventions["alpha_channel_forgetting"]["final_state"] == [[0, 0], [4, 5]]


def test_attnres_and_moe_results_keep_intervention_scope(generated):
    results = generated[0][0]["results"]
    query = results["attnres"]["query_intervention"]
    assert query["zero_query_weights"] == [.25] * 4
    assert query["target_source_weight"] > .999
    blocked = results["attnres"]["full_vs_block_fixed_outputs"]
    assert blocked["full_source_count"] == 5 and blocked["block_source_count"] == 3
    assert blocked["output_max_abs_difference"] > 1
    assert "not_full_network_training" in blocked["semantic"]
    moe = results["moe"]
    assert "same_calibration_fixture_replay" in moe["evaluation_scope"]
    methods = {row["method"]: row for row in moe["methods"]}
    assert methods["raw"]["loads"] == [7, 1, 0, 0]
    assert methods["exact_quantile_bias"]["loads"] == [2, 2, 2, 2]
    for row in methods.values():
        assert sum(row["loads"]) == 8
        assert row["dispatch_vs_token_reference_max_abs_error"] < 1e-12
    assert moe["new_batch_distribution_shift"]["loads_with_frozen_bias"] == [8, 0, 0, 0]


def test_formula_ledger_matches_independent_payload_and_parameter_count(generated):
    cost = generated[0][0]["results"]["cost_ledger"]
    single = cost["k3_like_single_layer_formula"]
    assert single["config"] == {
        "hidden_dim": 7168, "latent_dim": 3584, "ffn_dim": 3072,
        "num_experts": 896, "top_k": 16, "tokens": 1, "shared_experts": 2, "dtype_bytes": 2,
    }
    assert single["counts"]["routed_expert_parameters"] == 3 * 896 * 3584 * 3072
    shared_full_width = 2 * (7168 * 3072 + 7168 * 3072 + 3072 * 7168)
    assert single["counts"]["shared_expert_parameters"] == shared_full_width
    assert single["counts"]["shared_matrix_madds"] == shared_full_width  # one token
    assert single["assumptions"]["shared_expert_input_output_width"] == 7168
    assert single["assumptions"]["routed_expert_input_output_width"] == 3584
    assert single["assumptions"]["linear_bias_parameters_counted"] is False
    assert single["assumptions"]["independent_routing_control_bias_state_counted"] is False
    assert single["counts"]["logical_payload_bytes"] == 2 * 16 * 3584 * 2
    for row in cost["illustrative_hybrid_attention"]["rows"]:
        seq_len = row["sequence_length"]
        state = 18 * 1 * 8 * 64 * 64 * 2
        kv = 6 * 1 * 8 * seq_len * (64 + 64) * 2
        assert row["recurrent_state_bytes"] == state
        assert row["global_kv_bytes"] == kv
        assert row["hybrid_state_plus_kv_bytes"] == state + kv
        assert row["all_global_kv_bytes"] == kv * 4


def test_report_labels_measured_numerics_formulas_and_scope_clearly(generated):
    report = generated[0][1]
    for text in (
        "NumPy float64 实际执行", "seed=0、1、2", "--out", "依赖版本：Python",
        "最终状态最大绝对误差", "未来扰动与因果性", "固定 f_i 输出",
        "不是全网络训练指标", "同批回放", "一次校准不保证新批完全平衡",
        "简化单层公式", "教学示意，不是 K3 实测", "未测 GPU 速度",
        "全宽共享专家为 d→h→d", "独立路由均衡 bias 状态均未计入",
    ):
        assert text in report
    assert "TODO" not in report and "待填" not in report
    assert report.count("```") % 2 == 0
