from k3lab.summary import summarize


def test_summary_groups_by_model():
    records = [
        {
            "model": "m1",
            "tool_events": [{"ok": True}, {"ok": False}],
            "metadata": {
                "verifier": {"success": True},
                "reward": {"total": 0.8},
                "usage": [{"prompt_tokens": 10, "completion_tokens": 4}],
            },
        },
        {
            "model": "m1",
            "tool_events": [],
            "metadata": {
                "verifier": {"success": False},
                "reward": {"total": 0.0},
                "usage": [{"input_tokens": 5, "output_tokens": 2}],
            },
        },
    ]
    row = summarize(records)[0]
    assert row["model"] == "m1"
    assert row["tasks"] == 2
    assert row["success_rate"] == 0.5
    assert row["input_tokens"] == 15
    assert row["output_tokens"] == 6
