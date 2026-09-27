from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from k3lab.schema import Trajectory


@dataclass
class RewardResult:
    total: float
    components: dict[str, float] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)


class ExactAnswerVerifier:
    """Deterministic verifier for starter tasks."""

    def __init__(self, expected: str) -> None:
        self.expected = expected.strip()

    def __call__(self, answer: str) -> float:
        return float(answer.strip() == self.expected)


class CompositeReward:
    """Small transparent reward system for controlled ablations."""

    def __init__(
        self,
        *,
        success_weight: float = 1.0,
        invalid_call_penalty: float = 0.10,
        step_penalty: float = 0.01,
    ) -> None:
        self.success_weight = success_weight
        self.invalid_call_penalty = invalid_call_penalty
        self.step_penalty = step_penalty

    def score(self, trajectory: Trajectory, success: float) -> RewardResult:
        success_term = self.success_weight * success
        invalid_term = -self.invalid_call_penalty * trajectory.invalid_tool_calls
        step_term = -self.step_penalty * trajectory.num_tool_calls
        components = {
            "success": success_term,
            "invalid_tool_calls": invalid_term,
            "tool_steps": step_term,
        }
        return RewardResult(
            total=sum(components.values()),
            components=components,
            details={
                "num_tool_calls": trajectory.num_tool_calls,
                "invalid_tool_calls": trajectory.invalid_tool_calls,
            },
        )
