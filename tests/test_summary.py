from k3lab.summary import summarize


def _record(*, model="m1", history="full", harness="canonical", success=True, reward=0.8):
    return {
        "model": model,
        "tool_events": [{"ok": True}, {"ok": False}],
        "metadata": {
            "history_mode": history,
            "harness_variant": harness,
            "verifier": {"success": success},
            "reward": {"total": reward},
            "usage": [{"prompt_tokens": 10, "completion_tokens": 4}],
        },
    }


def test_summary_groups_same_experimental_setting():
    records = [
        _record(success=True, reward=0.8),
        {
            "model": "m1",
            "tool_events": [],
            "metadata": {
                "history_mode": "full",
                "harness_variant": "canonical",
                "verifier": {"success": False},
                "reward": {"total": 0.0},
                "usage": [{"input_tokens": 5, "output_tokens": 2}],
            },
        },
    ]
    row = summarize(records)[0]
    assert row["model"] == "m1"
    assert row["history_mode"] == "full"
    assert row["harness_variant"] == "canonical"
    assert row["tasks"] == 2
    assert row["success_rate"] == 0.5
    assert row["input_tokens"] == 15
    assert row["output_tokens"] == 6


def test_summary_does_not_mix_harness_or_history_ablations():
    records = [
        _record(history="full", harness="canonical", success=True),
        _record(history="full", harness="alternate", success=False),
        _record(history="no_reasoning", harness="canonical", success=False),
    ]
    rows = summarize(records)
    assert len(rows) == 3
    settings = {
        (row["history_mode"], row["harness_variant"]): row["success_rate"]
        for row in rows
    }
    assert settings[("full", "canonical")] == 1.0
    assert settings[("full", "alternate")] == 0.0
    assert settings[("no_reasoning", "canonical")] == 0.0
