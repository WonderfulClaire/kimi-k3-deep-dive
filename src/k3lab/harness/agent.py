from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from k3lab.schema import ToolEvent, Trajectory

from .tools import ToolRegistry


class AgentHarness:
    """Minimal agent loop with explicit trajectory capture.

    Important for Kimi-style multi-turn reasoning APIs: the full assistant
    message returned by the provider is round-tripped in memory rather than
    rebuilding only {"role": "assistant", "content": "..."}.
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
    ) -> None:
        self.provider = provider
        self.tools = tools
        self.max_steps = max_steps
        self.system_prompt = system_prompt

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

            messages.append(assistant)

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
