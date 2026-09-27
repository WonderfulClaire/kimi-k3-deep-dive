from k3lab.export_sft import is_verified_success, trajectory_to_sft


def test_verified_trajectory_reconstructs_tool_messages():
    record = {
        "task_id": "t1",
        "model": "m1",
        "prompt": "fix it",
        "assistant_messages": [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {"name": "read", "arguments": "{}"},
                    }
                ],
            },
            {"role": "assistant", "content": "done"},
        ],
        "tool_events": [
            {
                "step": 0,
                "tool_call_id": "c1",
                "name": "read",
                "arguments": {},
                "observation": "file",
                "ok": True,
            }
        ],
        "metadata": {
            "system_prompt": "system",
            "history_mode": "full",
            "harness_variant": "compact",
            "task_type": "repo_patch",
            "verifier": {"success": True},
            "reward": {"total": 0.97},
        },
    }
    assert is_verified_success(record)
    exported = trajectory_to_sft(record)
    roles = [message["role"] for message in exported["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert exported["metadata"]["verified_success"] is True
