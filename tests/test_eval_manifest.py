from argparse import Namespace
import json

from k3lab.eval import _run_config, _sha256, _write_manifest


def test_eval_manifest_captures_task_hash_and_ablation_settings(tmp_path):
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text('{"id":"t1"}\n', encoding="utf-8")
    args = Namespace(
        model="model-a",
        tasks=tasks,
        max_steps=9,
        history_mode="no_reasoning",
        harness_variant="alternate",
    )
    digest = _sha256(tasks)
    config = _run_config(args, digest)
    assert config["tasks_sha256"] == digest
    assert config["model"] == "model-a"
    assert config["max_steps"] == 9
    assert config["history_mode"] == "no_reasoning"
    assert config["harness_variant"] == "alternate"

    out = tmp_path / "run.jsonl"
    manifest = _write_manifest(out, config)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["tasks_sha256"] == digest
    assert payload["history_mode"] == "no_reasoning"
    assert payload["harness_variant"] == "alternate"
    assert "created_at" in payload
