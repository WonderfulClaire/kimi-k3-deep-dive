from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ToolEvent:
    step: int
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    observation: str
    ok: bool


@dataclass
class Trajectory:
    task_id: str
    model: str
    prompt: str
    final_answer: str = ""
    tool_events: list[ToolEvent] = field(default_factory=list)
    assistant_messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def invalid_tool_calls(self) -> int:
        return sum(not event.ok for event in self.tool_events)

    @property
    def num_tool_calls(self) -> int:
        return len(self.tool_events)

    def to_dict(self, include_reasoning: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        if not include_reasoning:
            payload["assistant_messages"] = [
                _redact_reasoning(message) for message in self.assistant_messages
            ]
        return payload


def _redact_reasoning(value: Any) -> Any:
    """Remove reasoning-like fields from persisted traces by default."""
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            lowered = key.lower()
            if "reasoning" in lowered or lowered in {"thought", "thinking"}:
                continue
            out[key] = _redact_reasoning(item)
        return out
    if isinstance(value, list):
        return [_redact_reasoning(item) for item in value]
    return value
