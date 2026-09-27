from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from k3lab.harness.agent import AgentHarness, append_jsonl
from k3lab.harness.tools import default_registry
from k3lab.providers import OpenAICompatibleProvider
from k3lab.rewards import CompositeReward, ExactAnswerVerifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Run same-harness agent evaluation.")
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("runs/trajectories.jsonl"))
    parser.add_argument("--model", default=os.getenv("K3LAB_MODEL"))
    parser.add_argument("--base-url", default=os.getenv("K3LAB_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("K3LAB_API_KEY"))
    parser.add_argument("--max-steps", type=int, default=8)
    args = parser.parse_args()

    if not args.model or not args.api_key:
        raise SystemExit("Set --model/--api-key or K3LAB_MODEL/K3LAB_API_KEY.")

    provider = OpenAICompatibleProvider(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )
    harness = AgentHarness(
        provider=provider,
        tools=default_registry(),
        max_steps=args.max_steps,
    )
    rewarder = CompositeReward()

    with args.tasks.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            task = json.loads(line)
            trajectory = harness.run(
                task_id=task["id"],
                prompt=task["prompt"],
                model_name=args.model,
            )
            verifier = ExactAnswerVerifier(str(task["expected"]))
            success = verifier(trajectory.final_answer)
            reward = rewarder.score(trajectory, success)
            trajectory.metadata["reward"] = {
                "total": reward.total,
                "components": reward.components,
            }
            append_jsonl(args.out, trajectory)
            print(
                json.dumps(
                    {
                        "task_id": task["id"],
                        "success": success,
                        "reward": reward.total,
                        "tool_calls": trajectory.num_tool_calls,
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
