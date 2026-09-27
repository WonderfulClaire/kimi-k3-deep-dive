from k3lab.rewards import CompositeReward
from k3lab.schema import ToolEvent, Trajectory


def test_composite_reward_is_auditable():
    trajectory = Trajectory(task_id="x", model="m", prompt="p")
    trajectory.tool_events.extend(
        [
            ToolEvent(0, "1", "calculator", {"expression": "1+1"}, '{"result": 2}', True),
            ToolEvent(1, "2", "missing", {}, "unknown tool", False),
        ]
    )
    result = CompositeReward(
        success_weight=1.0,
        invalid_call_penalty=0.1,
        step_penalty=0.01,
    ).score(trajectory, success=1.0)
    assert result.total == 0.88
    assert result.components == {
        "success": 1.0,
        "invalid_tool_calls": -0.1,
        "tool_steps": -0.02,
    }
