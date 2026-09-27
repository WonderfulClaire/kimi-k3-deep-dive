from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from k3lab.envs import MiniRepoEnv


def build_grpo_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            task = json.loads(line)
            if task.get("type") != "repo_patch":
                continue
            rows.append(
                {
                    "prompt": [{"role": "user", "content": task["prompt"]}],
                    "files_json": json.dumps(task["files"], ensure_ascii=False),
                    "hidden_tests_json": json.dumps(
                        task.get("hidden_tests", []), ensure_ascii=False
                    ),
                    "protected_files_json": json.dumps(
                        task.get("protected_files", ["tests_public.json"]),
                        ensure_ascii=False,
                    ),
                    "task_id": task["id"],
                }
            )
    return rows


class MiniRepoGRPOEnv:
    """TRL-compatible stateful environment for MiniRepo agent training.

    Only the four task-facing methods below are public tools. reset/get_reward are
    reserved lifecycle methods in TRL and are not exposed to the policy.
    """

    def __init__(self) -> None:
        self._env: MiniRepoEnv | None = None

    def reset(
        self,
        files_json: str,
        hidden_tests_json: str,
        protected_files_json: str,
        **kwargs: Any,
    ) -> None:
        self._env = MiniRepoEnv(
            files=json.loads(files_json),
            hidden_tests=json.loads(hidden_tests_json),
            protected_files=json.loads(protected_files_json),
        )

    def get_reward(self) -> float:
        if self._env is None:
            return 0.0
        return float(self._env.verify().success)

    def list_files(self) -> str:
        """List files in the current synthetic repository.

        Returns:
            A JSON string containing the available file paths.
        """
        return json.dumps(self._require_env().list_files(), ensure_ascii=False)

    def read_file(self, path: str) -> str:
        """Read one file from the current synthetic repository.

        Args:
            path: Relative path of the file to read.

        Returns:
            The UTF-8 file contents.
        """
        return self._require_env().read_file(path)

    def write_file(self, path: str, content: str) -> str:
        """Replace or create one file in the current synthetic repository.

        Args:
            path: Relative path to update.
            content: Complete UTF-8 replacement contents.

        Returns:
            A JSON status string.
        """
        result = self._require_env().write_file(path, content)
        return json.dumps(result, ensure_ascii=False)

    def run_public_tests(self) -> str:
        """Run the mutable visible public tests.

        Returns:
            A JSON string with pass/fail counts and failures.
        """
        result = self._require_env().run_public_tests()
        return json.dumps(result, ensure_ascii=False)

    def _require_env(self) -> MiniRepoEnv:
        if self._env is None:
            raise RuntimeError("environment has not been reset")
        return self._env
