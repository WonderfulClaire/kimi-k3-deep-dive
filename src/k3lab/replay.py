from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from k3lab.harness.tools import ToolRegistry, default_registry


def replay_record(
    record: dict[str, Any], registry: ToolRegistry
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for event in record.get("tool_events", []):
        ok, observation = registry.execute(event["name"], event.get("arguments", {}))
        checks.append(
            {
                "tool_call_id": event.get("tool_call_id"),
                "name": event["name"],
                "execution_ok": ok,
                "matches_recorded": ok and observation == event.get("observation"),
                "recorded_observation": event.get("observation"),
                "replayed_observation": observation,
            }
        )
    return {
        "task_id": record.get("task_id"),
        "all_match": all(item["matches_recorded"] for item in checks),
        "checks": checks,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay K3Lab tool trajectories.")
    parser.add_argument("trajectory_jsonl", type=Path)
    args = parser.parse_args()

    registry = default_registry()
    with args.trajectory_jsonl.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            report = replay_record(json.loads(line), registry)
            print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
