import json
from pathlib import Path

from k3lab.train.grpo_env import MiniRepoGRPOEnv, build_grpo_rows


def test_grpo_env_reward_uses_secure_verifier():
    env = MiniRepoGRPOEnv()
    env.reset(
        files_json=json.dumps({
            "SPEC.md": "f(x)=x*x+2*x+1",
            "solution.expr": "x*x+1",
            "tests_public.json": json.dumps([
                {"x": 0, "expected": 1},
                {"x": 1, "expected": 4},
            ]),
        }),
        hidden_tests_json=json.dumps([{"x": -3, "expected": 4}]),
        protected_files_json=json.dumps(["tests_public.json"]),
    )
    env.write_file("tests_public.json", "[]")
    assert json.loads(env.run_public_tests())["passed"] is True
    assert env.get_reward() == 0.0

    env.reset(
        files_json=json.dumps({
            "SPEC.md": "f(x)=x*x+2*x+1",
            "solution.expr": "x*x+1",
            "tests_public.json": json.dumps([
                {"x": 0, "expected": 1},
                {"x": 1, "expected": 4},
            ]),
        }),
        hidden_tests_json=json.dumps([{"x": -3, "expected": 4}]),
        protected_files_json=json.dumps(["tests_public.json"]),
    )
    env.write_file("solution.expr", "x*x + 2*x + 1")
    assert env.get_reward() == 1.0


def test_build_grpo_rows(tmp_path: Path):
    path = tmp_path / "tasks.jsonl"
    path.write_text(
        json.dumps({
            "id": "t",
            "type": "repo_patch",
            "prompt": "fix",
            "files": {
                "solution.expr": "x",
                "tests_public.json": "[]",
            },
            "hidden_tests": [],
            "protected_files": ["tests_public.json"],
        }) + "\n",
        encoding="utf-8",
    )
    rows = build_grpo_rows(path)
    assert len(rows) == 1
    assert rows[0]["prompt"][0]["content"] == "fix"
    assert rows[0]["task_id"] == "t"
