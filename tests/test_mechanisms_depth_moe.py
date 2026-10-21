"""Behavioral invariants and independent small references for depth and MoE."""

import numpy as np
import pytest

from k3lab.mechanisms.attnres import (
    attnres_storage_counts,
    block_attnres,
    block_sources,
    depth_attention,
    full_attnres,
    rms_norm,
)
from k3lab.mechanisms.moe import (
    heuristic_balance_bias,
    latent_moe_cost,
    load_statistics,
    moe_linear_dispatch,
    moe_linear_reference,
    quantile_balance_bias,
    sigmoid,
    topk_route,
)


def test_zero_pseudo_query_gives_uniform_source_weights_and_raw_mean():
    values = np.array([[1.0, 2.0], [10.0, 20.0], [-3.0, 4.0]])
    output, weights = depth_attention(values, np.zeros(2))
    np.testing.assert_allclose(weights, [1 / 3] * 3)
    np.testing.assert_allclose(output, [8 / 3, 26 / 3])


def test_key_rms_normalization_preserves_raw_value_scale():
    values = np.array([[1.0, 0.0], [10.0, 0.0]])
    output, weights = depth_attention(values, np.array([1.0, 0.0]), eps=0)
    np.testing.assert_allclose(weights, [0.5, 0.5])
    np.testing.assert_allclose(output, [5.5, 0.0])


def test_positive_source_scale_changes_values_not_key_scores_when_eps_zero():
    values = np.array([[1.0, 2.0], [2.0, -1.0], [1.0, -2.0]])
    scaled = values * np.array([5.0, 0.5, 3.0])[:, None]
    query = np.array([0.3, -0.2])
    _, weights = depth_attention(values, query, eps=0)
    output2, weights2 = depth_attention(scaled, query, eps=0)
    np.testing.assert_allclose(weights2, weights)
    np.testing.assert_allclose(output2, np.sum(weights[:, None] * scaled, axis=0))


def test_common_depth_logit_shift_does_not_change_weights():
    # All normalized keys have the same first coordinate, so modifying this
    # query coordinate adds exactly one common logit offset over depth.
    values = np.array([[1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, -1.0, 0.0]])
    query = np.array([0.2, 0.7, -0.3])
    output, weights = depth_attention(values, query)
    shifted_out, shifted = depth_attention(values, query + [10000, 0, 0])
    np.testing.assert_allclose(shifted, weights, atol=1e-12, rtol=1e-11)
    np.testing.assert_allclose(shifted_out, output, atol=1e-12, rtol=1e-11)


def test_depth_softmax_is_per_token_and_matches_independent_loop():
    rng = np.random.default_rng(7)
    values = rng.normal(size=(4, 2, 3, 5))
    query = rng.normal(size=5)
    output, weights = depth_attention(values, query, eps=1e-8)
    assert output.shape == (2, 3, 5)
    assert weights.shape == (4, 2, 3)
    np.testing.assert_allclose(weights.sum(axis=0), np.ones((2, 3)))
    reference = np.zeros_like(output)
    for batch in range(2):
        for token in range(3):
            logits = []
            for source in range(4):
                v = values[source, batch, token]
                key = v / np.sqrt(sum(float(x) ** 2 for x in v) / 5 + 1e-8)
                logits.append(sum(float(a) * float(b) for a, b in zip(query, key)))
            probabilities = np.exp(logits - np.max(logits))
            probabilities /= probabilities.sum()
            for source in range(4):
                reference[batch, token] += probabilities[source] * values[source, batch, token]
    np.testing.assert_allclose(output, reference, atol=1e-12, rtol=1e-12)


def test_no_token_mixing_in_depth_attention():
    rng = np.random.default_rng(13)
    values = rng.normal(size=(3, 4, 5))
    query = rng.normal(size=5)
    before, _ = depth_attention(values, query)
    values[:, 2] += 100
    after, _ = depth_attention(values, query)
    np.testing.assert_array_equal(before[[0, 1, 3]], after[[0, 1, 3]])
    assert not np.allclose(before[2], after[2])


def test_full_and_block_use_different_source_sets():
    embedding = np.zeros(2)
    outputs = np.array([[2, 0], [2, 0], [0, 6], [0, 6]], dtype=float)
    full, full_weights = full_attnres(embedding, outputs, np.zeros(2))
    block, block_weights = block_attnres(embedding, outputs, np.zeros(2), block_size=2)
    np.testing.assert_allclose(full, [0.8, 2.4])
    np.testing.assert_allclose(block, [4 / 3, 4.0])
    assert full_weights.shape == (5,)
    assert block_weights.shape == (3,)
    assert not np.allclose(full, block)


def test_block_size_one_equals_full_and_embedding_is_always_its_own_source():
    rng = np.random.default_rng(1)
    embedding = rng.normal(size=(2, 4))
    outputs = rng.normal(size=(5, 2, 4))
    query = rng.normal(size=4)
    full = full_attnres(embedding, outputs, query)
    blocked = block_attnres(embedding, outputs, query, block_size=1)
    for a, b in zip(full, blocked):
        np.testing.assert_allclose(a, b)


def test_partial_block_and_exact_boundary_do_not_insert_zero_sources():
    outputs = np.arange(10, dtype=float).reshape(5, 2)
    embedding = np.array([10.0, 20.0])
    sources = block_sources(embedding, outputs, block_size=2)
    np.testing.assert_array_equal(sources, [[10, 20], [2, 4], [10, 12], [8, 9]])
    boundary = block_sources(embedding, outputs[:4], block_size=2)
    assert boundary.shape == (3, 2)
    np.testing.assert_array_equal(boundary, sources[:3])
    out, weights = block_attnres(embedding, [], np.zeros(2), block_size=2)
    np.testing.assert_array_equal(out, embedding)
    np.testing.assert_array_equal(weights, [1])


def test_attnres_storage_includes_embedding_and_final_partial_block():
    counts = attnres_storage_counts(5, 7, block_size=2, batch_size=3, seq_len=11, dtype_bytes=8)
    assert counts == {
        "full_sources": 6,
        "block_sources": 4,
        "full_elements": 6 * 3 * 11 * 7,
        "block_elements": 4 * 3 * 11 * 7,
        "full_bytes": 6 * 3 * 11 * 7 * 8,
        "block_bytes": 4 * 3 * 11 * 7 * 8,
    }


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_invalid_block_size_is_rejected(bad):
    with pytest.raises(ValueError):
        block_sources(np.ones(2), [], block_size=bad)


def test_rmsnorm_axis_is_features_not_tokens_or_depth():
    x = np.array([[3.0, 4.0], [30.0, 40.0]])
    normalized = rms_norm(x, eps=0)
    np.testing.assert_allclose(np.mean(normalized**2, axis=-1), [1, 1])
    np.testing.assert_allclose(normalized[0], normalized[1])


def test_bias_selects_experts_but_does_not_enter_mixture_weights():
    scores = np.array([[0.9, 0.8, 0.2]])
    plan = topk_route(scores, 2, bias=np.array([0, -0.8, 1]))
    np.testing.assert_array_equal(plan.indices, [[2, 0]])
    np.testing.assert_allclose(plan.weights, [[0.2 / 1.1, 0.9 / 1.1]])
    np.testing.assert_array_equal(plan.loads, [1, 0, 1])
    same_selection = topk_route(scores, 2, bias=np.array([0, -0.9, 2]))
    np.testing.assert_array_equal(same_selection.indices, plan.indices)
    np.testing.assert_array_equal(same_selection.weights, plan.weights)


def test_common_bias_offset_invariant_and_ties_use_expert_id():
    scores = np.array([[0.8, 0.8, 0.1], [0.2, 0.7, 0.7]])
    baseline = topk_route(scores, 2)
    shifted = topk_route(scores, 2, np.full(3, 10.0))
    np.testing.assert_array_equal(baseline.indices, [[0, 1], [1, 2]])
    np.testing.assert_array_equal(baseline.indices, shifted.indices)
    np.testing.assert_allclose(baseline.weights, shifted.weights)
    np.testing.assert_allclose(baseline.weights.sum(axis=-1), [1, 1])


def test_dispatch_combine_matches_hand_computation():
    tokens = np.array([[1.0, 2.0], [3.0, 4.0]])
    experts = np.array([[[1.0], [0.0]], [[0.0], [2.0]], [[1.0], [1.0]]])
    plan = topk_route(np.array([[0.6, 0.3, 0.1], [0.1, 0.5, 0.4]]), 2)
    expected = np.array([[(0.6 * 1 + 0.3 * 4) / 0.9], [(0.5 * 8 + 0.4 * 7) / 0.9]])
    np.testing.assert_allclose(moe_linear_dispatch(tokens, experts, plan), expected)
    np.testing.assert_allclose(moe_linear_reference(tokens, experts, plan), expected)


def test_expert_major_dispatch_matches_independent_token_loop_with_empty_experts():
    rng = np.random.default_rng(4)
    tokens = rng.normal(size=(11, 4))
    experts = rng.normal(size=(7, 4, 3))
    scores = rng.uniform(0.1, 0.9, size=(11, 7))
    scores[:, -1] = 0
    plan = topk_route(scores, 3)
    assert plan.loads[-1] == 0
    assert plan.loads.sum() == 11 * 3
    np.testing.assert_allclose(
        moe_linear_dispatch(tokens, experts, plan),
        moe_linear_reference(tokens, experts, plan),
        atol=1e-12,
        rtol=1e-12,
    )


def test_load_skew_and_independent_heuristic_bias_direction():
    scores = np.tile([0.9, 0.8, 0.1, 0.05], (8, 1))
    plan = topk_route(scores, 1)
    np.testing.assert_array_equal(plan.loads, [8, 0, 0, 0])
    assert load_statistics(plan)["max_over_mean"] == 4
    bias = heuristic_balance_bias(plan.loads, step_size=0.2)
    assert bias[0] < 0 and np.all(bias[1:] > 0)
    assert abs(bias.sum()) < 1e-15
    np.testing.assert_array_equal(scores, np.tile([0.9, 0.8, 0.1, 0.05], (8, 1)))


def test_exact_quantile_matches_independent_order_statistic_and_common_shift():
    rng = np.random.default_rng(123)
    scores = rng.uniform(0.05, 0.95, size=(12, 4))
    old_bias = np.array([0.08, -0.04, 0.02, -0.06])
    k, target = 1, 3
    alpha = np.array([sorted(row + old_bias, reverse=True)[k] for row in scores])
    expected_raw = np.array([-sorted(scores[:, e] - alpha, reverse=True)[target] for e in range(4)])
    actual = quantile_balance_bias(scores, k, old_bias)
    np.testing.assert_allclose(actual, expected_raw - expected_raw.mean())
    np.testing.assert_allclose(actual, quantile_balance_bias(scores, k, old_bias + 5))
    # Even continuous raw scores can have tied margins when an expert itself
    # supplies alpha. Order-statistic inequalities survive those ties.
    for expert in range(4):
        margins = scores[:, expert] - alpha
        threshold = -expected_raw[expert]
        assert np.sum(margins > threshold) <= target
        assert np.sum(margins >= threshold) >= target + 1
        if np.sum(margins == threshold) == 1:
            assert np.sum(margins > threshold) == target


def test_quantile_ties_and_fractional_targets_are_explicit():
    scores = np.full((8, 4), 0.5)
    bias = quantile_balance_bias(scores, 1)
    np.testing.assert_array_equal(bias, np.zeros(4))
    # Identical scores cannot yield equal top-1 loads under deterministic ties.
    np.testing.assert_array_equal(topk_route(scores, 1, bias).loads, [8, 0, 0, 0])
    with pytest.raises(ValueError, match="integer"):
        quantile_balance_bias(np.full((5, 3), 0.5), 1)


def test_latent_parameters_and_payload_counts_checked_independently():
    c = latent_moe_cost(8, 4, 3, 6, 2, tokens=5, shared_experts=1, dtype_bytes=2)
    # Per expert: gate 4x3 + up 4x3 + down 3x4 = 36 parameters.
    assert c["routed_expert_parameters"] == 6 * 36
    assert c["shared_expert_parameters"] == 8 * 3 + 8 * 3 + 3 * 8
    assert c["projection_parameters"] == 8 * 4 + 4 * 8
    assert c["router_parameters"] == 8 * 6
    assert c["norm_parameters"] == 4
    assert c["total_parameters"] == 216 + 72 + 64 + 48 + 4
    assert c["active_parameters_per_token"] == 2 * 36 + 72 + 64 + 48 + 4
    assert c["matrix_madds"] == 5 * (72 + 72 + 64 + 48)
    assert c["shared_matrix_madds"] == 5 * (8 * 3 + 8 * 3 + 3 * 8)
    assert c["matrix_madds"] == sum(c[key] for key in (
        "routed_matrix_madds", "shared_matrix_madds", "projection_matrix_madds", "router_matrix_madds"
    ))
    # 5 tokens x 2 destinations x 4 coordinates x 2 bytes x 2 directions.
    assert c["logical_payload_bytes"] == 160
    assert c["full_width_logical_payload_bytes"] == 320
    assert c["dispatch_elements"] == c["combine_elements"] == 40


def test_width_halving_has_projection_overhead_and_topk_changes_work_not_pool():
    full = latent_moe_cost(16, 16, 7, 8, 2, tokens=3)
    half = latent_moe_cost(16, 8, 7, 8, 2, tokens=3)
    more_active = latent_moe_cost(16, 8, 7, 8, 4, tokens=3)
    assert half["routed_expert_parameters"] * 2 == full["routed_expert_parameters"]
    assert half["logical_payload_bytes"] * 2 == full["logical_payload_bytes"]
    assert half["projection_parameters"] > 0
    assert half["total_parameters"] == more_active["total_parameters"]
    assert more_active["active_parameters_per_token"] > half["active_parameters_per_token"]


def test_shared_experts_remain_full_width_when_routed_latent_width_changes():
    wide = latent_moe_cost(16, 16, 7, 8, 2, tokens=3, shared_experts=2)
    narrow = latent_moe_cost(16, 4, 7, 8, 2, tokens=3, shared_experts=2)
    # Both shared experts have gate/up 16x7 and down 7x16, for every token.
    expected_shared = 2 * (16 * 7 + 16 * 7 + 7 * 16)
    assert wide["shared_expert_parameters"] == narrow["shared_expert_parameters"] == expected_shared
    assert wide["shared_matrix_madds"] == narrow["shared_matrix_madds"] == 3 * expected_shared
    assert wide["routed_expert_parameters"] == 4 * narrow["routed_expert_parameters"]
    assert wide["routed_matrix_madds"] == 4 * narrow["routed_matrix_madds"]


@pytest.mark.parametrize("scores", [np.array([[-0.1, 0.5]]), np.array([[1.1, 0.5]]), np.array([[np.nan, 0.5]])])
def test_bad_router_scores_rejected(scores):
    with pytest.raises(ValueError):
        topk_route(scores, 1)


def test_stable_sigmoid_and_zero_mass_rejection():
    np.testing.assert_allclose(sigmoid(np.array([-1000, 0, 1000])), [0, 0.5, 1])
    with pytest.raises(ValueError, match="mass"):
        topk_route(np.zeros((2, 3)), 1)
