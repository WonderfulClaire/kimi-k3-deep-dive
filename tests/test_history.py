from types import SimpleNamespace

from k3lab.harness.agent import AgentHarness
from k3lab.harness.tools import default_registry


class FakeProvider:
    def __init__(self):
        self.calls = []

    def complete(self, messages, tools, **extra):
        self.calls.append(messages)
        if len(self.calls) == 1:
            return SimpleNamespace(
                assistant_message={
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "private reasoning state",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {
                                "name": "calculator",
                                "arguments": '{"expression":"1+1"}',
                            },
                        }
                    ],
                },
                usage={},
            )
        return SimpleNamespace(
            assistant_message={
                "role": "assistant",
                "content": "2",
                "reasoning_content": "second private state",
            },
            usage={},
        )


def _previous_assistant(provider):
    return [
        message
        for message in provider.calls[1]
        if message.get("role") == "assistant"
    ][0]


def test_full_history_round_trips_reasoning_fields():
    provider = FakeProvider()
    AgentHarness(
        provider=provider,
        tools=default_registry(),
        history_mode="full",
    ).run(task_id="x", prompt="1+1", model_name="fake")
    assert _previous_assistant(provider)["reasoning_content"] == "private reasoning state"


def test_no_reasoning_keeps_tool_protocol_but_strips_reasoning():
    provider = FakeProvider()
    AgentHarness(
        provider=provider,
        tools=default_registry(),
        history_mode="no_reasoning",
    ).run(task_id="x", prompt="1+1", model_name="fake")
    previous = _previous_assistant(provider)
    assert "reasoning_content" not in previous
    assert previous["tool_calls"][0]["function"]["name"] == "calculator"
