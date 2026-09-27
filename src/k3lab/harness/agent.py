from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from k3lab.schema import ToolEvent, Trajectory, redact_reasoning_fields

from .tools import ToolRegistry

HistoryMode = Literal["full", "no_reasoning"]


class AgentHarness:
    """Minimal agent loop with explicit trajectory capture.

    `full` history round-trips the provider's complete assistant message.
    `no_reasoning` removes reasoning-like fields but preserves content and
    tool-call protocol fields, enabling a controlled history ablation.
    """

    def __init__(
        self,
        *,
        provider: Any,
        tools: ToolRegistry,
        max_steps: int = 12,
        system_prompt: str = (
            "You are an agent. Use tools when useful. "
            "Return a concise final answer when the task is complete."
        ),
        history_mode: HistoryMode = "full",
    ) -> None:
        if history_mode not in {"full", "no_reasoning"}:
            raise ValueError(f"unsupported history_mode: {history_mode}")
        self.provider = provider
        self.tools = tools
        self.max_steps = max_steps
        self.system_prompt = system_prompt
        self.history_mode = history_mode

    def run(
        self,
        *,
        task_id: str,
        prompt: str,
        model_name: str,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> Trajectory:
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        trajectory = Trajectory(task_id=task_id, model=model_name, prompt=prompt)
        trajectory.metadata["system_prompt"] = self.system_prompt
        trajectory.metadata["history_mode"] = self.history_mode
        generation_kwargs = generation_kwargs or {}
        usage_records: list[dict[str, Any]] = []

        for step_idx in range(self.max_steps):
            response = self.provider.complete(
                messages=messages,
                tools=self.tools.schemas(),
                **generation_kwargs,
            )
            assistant = dict(response.assistant_message)
            assistant.setdefault("role", "assistant")
            trajectory.assistant_messages.append(assistant)
            usage_records.append(response.usage)

            messages.append(self._history_message(assistant))

            tool_calls = assistant.get("tool_calls") or []
            if not tool_calls:
                trajectory.final_answer = assistant.get("content") or ""
                trajectory.metadata["usage"] = usage_records
                trajectory.metadata["stop_reason"] = "final_answer"
                return trajectory

            for tool_call in tool_calls:
                function = tool_call.get("function", {})
                name = function.get("name", "")
                raw_args = function.get("arguments", "{}")
                try:
                    arguments = (
                        raw_args if isinstance(raw_args, dict) else json.loads(raw_args)
                    )
                    if not isinstance(arguments, dict):
                        raise ValueError("tool arguments must decode to an object")
                    ok, observation = self.tools.execute(name, arguments)
                except Exception as exc:
                    arguments = {"_raw": raw_args}
                    ok, observation = False, f"{type(exc).__name__}: {exc}"

                call_id = str(tool_call.get("id") or f"step-{step_idx}-{name}")
                trajectory.tool_events.append(
                    ToolEvent(
                        step=step_idx,
                        tool_call_id=call_id,
                        name=name,
                        arguments=arguments,
                        observation=observation,
                        ok=ok,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": observation,
                    }
                )

        trajectory.metadata["usage"] = usage_records
        trajectory.metadata["stop_reason"] = "max_steps"
        return trajectory

    def _history_message(self, assistant: dict[str, Any]) -> dict[str, Any]:
        if self.history_mode == "full":
            return dict(assistant)
        return redact_reasoning_fields(assistant)


def append_jsonl(
    path: str | Path,
    trajectory: Trajectory,
    *,
    include_reasoning: bool = False,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                trajectory.to_dict(include_reasoning=include_reasoning),
                ensure_ascii=False,
            )
            + "\n"
        )
