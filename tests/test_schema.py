from k3lab.schema import Trajectory


def test_reasoning_is_redacted_from_persisted_trace_by_default():
    trajectory = Trajectory(task_id="x", model="m", prompt="p")
    trajectory.assistant_messages.append(
        {
            "role": "assistant",
            "content": "answer",
            "reasoning_content": "private trace",
            "nested": {"thinking": "also private", "value": 3},
        }
    )
    stored = trajectory.to_dict()
    message = stored["assistant_messages"][0]
    assert "reasoning_content" not in message
    assert "thinking" not in message["nested"]
    assert message["nested"]["value"] == 3
