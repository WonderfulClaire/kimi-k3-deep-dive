from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _usage_totals(record: dict[str, Any]) -> tuple[int, int]:
    input_tokens = 0
    output_tokens = 0
    for usage in record.get("metadata", {}).get("usage", []):
        input_tokens += int(
            usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0
        )
        output_tokens += int(
            usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0
        )
    return input_tokens, output_tokens


def _success(record: dict[str, Any]) -> float:
    verifier = record.get("metadata", {}).get("verifier", {})
    if "success" in verifier:
        return float(bool(verifier["success"]))
    components = record.get("metadata", {}).get("reward", {}).get("components", {})
    return float(components.get("success", 0.0) > 0)


def _setting(record: dict[str, Any]) -> tuple[str, str, str]:
    metadata = record.get("metadata", {})
    return (
        str(record.get("model", "unknown")),
        str(metadata.get("history_mode", "unknown")),
        str(metadata.get("harness_variant", "unknown")),
    )


def summarize(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate runs without collapsing distinct harness/history settings.

    A model can behave very differently under another history policy or tool
    schema. Grouping by model alone silently mixes those experiments and can
    hide harness sensitivity, so the setting is part of the aggregation key.
    """
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[_setting(record)].append(record)

    rows: list[dict[str, Any]] = []
    for (model, history_mode, harness_variant), items in sorted(groups.items()):
        inputs, outputs = zip(*[_usage_totals(item) for item in items])
        rows.append(
            {
                "model": model,
                "history_mode": history_mode,
                "harness_variant": harness_variant,
                "tasks": len(items),
                "success_rate": mean(_success(item) for item in items),
                "avg_reward": mean(
                    float(item.get("metadata", {}).get("reward", {}).get("total", 0.0))
                    for item in items
                ),
                "avg_tool_calls": mean(len(item.get("tool_events", [])) for item in items),
                "avg_invalid_calls": mean(
                    sum(not event.get("ok", False) for event in item.get("tool_events", []))
                    for item in items
                ),
                "input_tokens": sum(inputs),
                "output_tokens": sum(outputs),
            }
        )
    return rows


def _load(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    return records


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Model | History | Harness | Tasks | Success | Avg reward | Avg tool calls | Avg invalid | Input tok | Output tok |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {model} | {history_mode} | {harness_variant} | {tasks} | "
            "{success_rate:.1%} | {avg_reward:.3f} | {avg_tool_calls:.2f} | "
            "{avg_invalid_calls:.2f} | {input_tokens} | {output_tokens} |".format(**row)
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize K3Lab JSONL runs.")
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    rows = summarize(_load(args.runs))
    if args.as_json:
        print(json.dumps(rows, ensure_ascii=False, indent=2))
    else:
        print(_markdown(rows))


if __name__ == "__main__":
    main()
