from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from k3lab import __version__
from k3lab.envs import MiniRepoEnv
from k3lab.harness.agent import AgentHarness
from k3lab.harness.agent import append_jsonl
from k3lab.harness.tools import default_registry
from k3lab.providers import OpenAICompatibleProvider
from k3lab.rewards import CompositeReward, ExactAnswerVerifier


REPO_SYSTEM_PROMPT = (
    "You are a coding agent in a small synthetic repository. "
    "Inspect files, edit the implementation, and run public tests. "
    "Finish only when you believe the repository is correct."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_config(args: argparse.Namespace, task_sha256: str) -> dict[str, Any]:
    return {
        "k3lab_version": __version__,
        "model": args.model,
        "tasks": str(args.tasks),
        "tasks_sha256": task_sha256,
        "max_steps": args.max_steps,
        "history_mode": args.history_mode,
        "harness_variant": args.harness_variant,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
    }


def _write_manifest(out: Path, config: dict[str, Any]) -> Path:
    path = out.with_name(out.name + ".manifest.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **config,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _score_task(
    *,
    task: dict[str, Any],
    provider: OpenAICompatibleProvider,
    model: str,
    max_steps: int,
    history_mode: str,
    harness_variant: str,
    system_prompt: str | None = None,
) -> tuple[Any, float]:
    task_type = task.get("type", "exact_answer")
    rewarder = CompositeReward()

    if task_type == "repo_patch":
        env = MiniRepoEnv.from_task(task)
        harness = AgentHarness(
            provider=provider,
            tools=env.registry(variant=harness_variant),
            max_steps=max_steps,
            history_mode=history_mode,
            system_prompt=system_prompt or REPO_SYSTEM_PROMPT,
        )
        trajectory = harness.run(
            task_id=task["id"],
            prompt=task["prompt"],
            model_name=model,
        )
        verifier = env.verify()
        success = float(verifier.success)
        trajectory.metadata["task_type"] = task_type
        trajectory.metadata["harness_variant"] = harness_variant
        trajectory.metadata["verifier"] = verifier.to_dict()
        trajectory.metadata["environment"] = env.snapshot()
    elif task_type == "exact_answer":
        harness = AgentHarness(
            provider=provider,
            tools=default_registry(),
            max_steps=max_steps,
            history_mode=history_mode,
            **({"system_prompt": system_prompt} if system_prompt else {}),
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
        trajectory.metadata["harness_variant"] = "canonical"
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
    parser.add_argument(
        "--history-mode",
        choices=["full", "no_reasoning"],
        default="full",
    )
    parser.add_argument(
        "--harness-variant",
        choices=["canonical", "compact", "alternate"],
        default="canonical",
    )
    args = parser.parse_args()

    if not args.model or not args.api_key:
        raise SystemExit("Set --model/--api-key or K3LAB_MODEL/K3LAB_API_KEY.")

    task_sha256 = _sha256(args.tasks)
    run_config = _run_config(args, task_sha256)
    manifest_path = _write_manifest(args.out, run_config)

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
                history_mode=args.history_mode,
                harness_variant=args.harness_variant,
            )
            trajectory.metadata["run_config"] = run_config
            append_jsonl(args.out, trajectory)
            print(
                json.dumps(
                    {
                        "task_id": task["id"],
                        "task_type": trajectory.metadata["task_type"],
                        "history_mode": trajectory.metadata["history_mode"],
                        "harness_variant": trajectory.metadata["harness_variant"],
                        "success": success,
                        "reward": trajectory.metadata["reward"]["total"],
                        "tool_calls": trajectory.num_tool_calls,
                        "invalid_tool_calls": trajectory.invalid_tool_calls,
                        "manifest": str(manifest_path),
                    },
                    ensure_ascii=False,
                )
            )


if __name__ == "__main__":
    main()
