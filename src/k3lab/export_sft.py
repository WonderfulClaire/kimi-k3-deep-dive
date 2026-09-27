from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def is_verified_success(record: dict[str, Any]) -> bool:
    return bool(record.get("metadata", {}).get("verifier", {}).get("success", False))


def trajectory_to_sft(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata", {})
    messages: list[dict[str, Any]] = []
    system_prompt = metadata.get("system_prompt")
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": record.get("prompt", "")})

    by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for event in record.get("tool_events", []):
        by_step[int(event.get("step", 0))].append(event)

    for step, assistant in enumerate(record.get("assistant_messages", [])):
        messages.append(dict(assistant))
        for event in by_step.get(step, []):
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": event.get("tool_call_id"),
                    "content": event.get("observation", ""),
                }
            )

    return {
        "task_id": record.get("task_id"),
        "source_model": record.get("model"),
        "messages": messages,
        "metadata": {
            "verified_success": True,
            "task_type": metadata.get("task_type"),
            "harness_variant": metadata.get("harness_variant"),
            "history_mode": metadata.get("history_mode"),
            "reward": metadata.get("reward", {}).get("total"),
        },
    }


def export_verified(paths: list[Path], out: Path) -> int:
    out.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with out.open("w", encoding="utf-8") as target:
        for path in paths:
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if not is_verified_success(record):
                        continue
                    target.write(
                        json.dumps(trajectory_to_sft(record), ensure_ascii=False) + "\n"
                    )
                    count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export securely verified trajectories as SFT JSONL."
    )
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    count = export_verified(args.runs, args.out)
    print(json.dumps({"exported": count, "out": str(args.out)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
