import pytest

from k3lab.autoresearch import generate_candidates, select_best, validate_space


def test_rejects_heldout_harness_in_dev_search():
    with pytest.raises(ValueError, match="alternate is reserved"):
        validate_space({"harness_variant": ["alternate"]})


def test_rejects_non_allowlisted_search_key():
    with pytest.raises(ValueError, match="forbidden keys"):
        validate_space({"private_setting": ["x"]})


def test_generates_bounded_candidates():
    candidates = generate_candidates(
        {
            "max_steps": [4, 8],
            "history_mode": ["full"],
            "harness_variant": ["canonical"],
            "prompt_variant": ["baseline", "evidence_first"],
        },
        max_candidates=3,
    )
    assert len(candidates) == 3
    assert all(c.harness_variant == "canonical" for c in candidates)
    assert {c.max_steps for c in candidates}.issubset({4, 8})


def test_selects_secure_success_before_reward():
    results = [
        {
            "candidate": {"candidate_id": "c001"},
            "secure_success_rate": 0.5,
            "avg_reward": 10.0,
            "avg_invalid_calls": 0.0,
            "avg_tool_calls": 1.0,
        },
        {
            "candidate": {"candidate_id": "c002"},
            "secure_success_rate": 1.0,
            "avg_reward": 0.1,
            "avg_invalid_calls": 0.2,
            "avg_tool_calls": 4.0,
        },
    ]
    assert select_best(results)["candidate"]["candidate_id"] == "c002"


def test_tie_breaks_by_invalid_then_tool_calls():
    results = [
        {
            "candidate": {"candidate_id": "c001"},
            "secure_success_rate": 1.0,
            "avg_reward": 0.1,
            "avg_invalid_calls": 1.0,
            "avg_tool_calls": 2.0,
        },
        {
            "candidate": {"candidate_id": "c002"},
            "secure_success_rate": 1.0,
            "avg_reward": 0.1,
            "avg_invalid_calls": 0.0,
            "avg_tool_calls": 5.0,
        },
        {
            "candidate": {"candidate_id": "c003"},
            "secure_success_rate": 1.0,
            "avg_reward": 0.1,
            "avg_invalid_calls": 0.0,
            "avg_tool_calls": 3.0,
        },
    ]
    assert select_best(results)["candidate"]["candidate_id"] == "c003"
