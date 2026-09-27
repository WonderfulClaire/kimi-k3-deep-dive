import json

from k3lab.envs import MiniRepoEnv


def _env():
    return MiniRepoEnv(
        files={
            "SPEC.md": "f(x)=x+1",
            "solution.expr": "x",
            "tests_public.json": json.dumps([{"x": 1, "expected": 2}]),
        },
        hidden_tests=[{"x": 5, "expected": 6}],
    )


def test_harness_variants_change_schema_not_semantics():
    expected_names = {
        "canonical": {"list_files", "read_file", "write_file", "run_public_tests"},
        "compact": {"ls", "read", "write", "test"},
        "alternate": {
            "show_repo_files",
            "inspect_file",
            "update_file",
            "check_visible_tests",
        },
    }
    for variant, names in expected_names.items():
        registry = _env().registry(variant=variant)
        schema_names = {
            item["function"]["name"]
            for item in registry.schemas()
        }
        assert schema_names == names
