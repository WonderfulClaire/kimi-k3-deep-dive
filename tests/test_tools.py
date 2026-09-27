from k3lab.harness.tools import calculator, default_registry


def test_calculator():
    assert calculator("(23 * 17) + 9") == {"result": 400}


def test_registry_unknown_tool():
    ok, observation = default_registry().execute("missing", {})
    assert not ok
    assert "unknown tool" in observation
