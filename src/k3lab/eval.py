from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from k3lab.envs import MiniRepoEnv
from k3lab.harness.agent import AgentHarness, append_jsonl
from k3lab.harness.tools import default_registry
from k3lab.providers import OpenAICompatibleProvider
from k3lab.rewards import CompositeReward, ExactAnswerVerifier


def _score_task(
    *,
    task: dict[str, Any],
    provider: OpenAICompatibleProvider,
    model: str,
    max_steps: int,
) -> tuple[Any, float]:
    task_type = task.get("type", "exact_answer")
    rewarder = CompositeReward()

    if task_type == "repo_patch":
        env = MiniRepoEnv.from_task(task)
        harness = AgentHarness(
            provider=provider,
            tools=env.registry(),
            max_steps=max_steps,
            system_prompt=(
                "You are a coding agent in a small synthetic repository. "
                "Inspect files, edit the implementation, and run public tests. "
                "Finish only when you believe the repository is correct."
            ),
        )
        trajectory = harness.run(
            task_id=task["id"],
            prompt=task["prompt"],
            model_name=model,
        )
        verifier = env.verify()
        success = float(verifier.success)
        trajectory.metadata["task_type"] = task_type
        trajectory.metadata["verifier"] = verifier.to_dict()
        trajectory.metadata["environment"] = env.snapshot()
    elif task_type == "exact_answer":
        harness = AgentHarness(
            provider=provider,
            tools=default_registry(),
            max_steps=max_steps,
        )
        trajectory = harness.run(
            task_id=task["id"],
            prompt=task["prompt"],
            model_name=model,
        )
        success = ExactAnswerVerifier(str(task["expected"]))(
            trajectory.final_answer
        )
        trajectory.metadata["task_type"] = task_type
        trajectory.metadata["verifier"] = {"success": bool(success)}
    else:
        raise ValueError(f"unknown task type: {task_type}")

    reward = rewarder.score(trajectory, success)
    trajectory.metadata["reward"] = {
        "total": reward.total,
        "components": reward.components,
    }
    return trajectory, success


def main() -> None:
    parser = argparse.ArgumentParser(description="Run same-harness agent evaluation.")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/trajectories.jsonl"))
    parser.add_argument("--model", default=os.getenv("K3LAB_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("K3LAB_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("K3LAB_API_KEY"))
    parser.add_argument("--max-steps", type=int, default=12)
    args = parser.parse_args()

    if not args.model or not args.api_key:
        raise SystemExit("Set --model/--api-key or K3LAB_MODEL/K3LAB_API_KEY.")

    provider = OpenAICompatibleProvider(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    with args.tasks.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            task = json.loads(line)
            trajectory, success = _score_task(
                task=task,
                provider=provider,
                model=args.model,
                max_steps=args.max_steps,
            )
            append_jsonl(args.out, trajectory)
            print(
                json.dumps(
                    {
                        "task_id": task["id"],
                        "task_type": trajectory.metadata["task_type"],
                        "success": success,
                        "reward": trajectory.metadata["reward"]["total"],
                        "tool_calls": trajectory.num_tool_calls,
                        "invalid_tool_calls": trajectory.invalid_tool_calls,
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
