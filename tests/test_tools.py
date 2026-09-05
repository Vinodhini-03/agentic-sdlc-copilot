import pytest

from agentic_sdlc.providers.base import ToolSpec
from agentic_sdlc.tools.registry import ToolRegistry


@pytest.mark.asyncio
async def test_register_and_call_tool():
    reg = ToolRegistry()

    async def echo(text: str) -> str:
        return f"echo: {text}"

    reg.register(
        ToolSpec(name="echo", description="Echoes input", parameters={"type": "object"}),
        echo,
    )

    result = await reg.call("echo", {"text": "hello"})
    assert result == "echo: hello"


@pytest.mark.asyncio
async def test_call_unregistered_tool_raises():
    reg = ToolRegistry()
    with pytest.raises(KeyError):
        await reg.call("nope", {})


def test_specs_reflects_registered_tools():
    reg = ToolRegistry()

    async def noop() -> None:
        return None

    reg.register(ToolSpec(name="noop", description="does nothing", parameters={}), noop)
    names = [s.name for s in reg.specs()]
    assert names == ["noop"]


@pytest.mark.asyncio
async def test_run_tests_falls_back_when_model_hallucinates_path(tmp_path):
    """
    Regression test for a real failure observed with a local model: it called
    run_tests with test_path="path/to/tests" (a placeholder) instead of the
    schema default "tests/". The tool must not blindly trust that argument;
    it should fall back to the real default and say so.
    """
    from agentic_sdlc.tools.registry import run_tests

    real_tests_dir = tmp_path / "tests"
    real_tests_dir.mkdir()
    (real_tests_dir / "test_dummy.py").write_text("def test_ok():\n    assert True\n")

    result = await run_tests(repo_path=str(tmp_path), test_path="path/to/tests")

    assert result["passed"] is True
    assert "warning" in result
    assert "tests/" in result["warning"]


@pytest.mark.asyncio
async def test_run_tests_uses_given_path_when_it_exists(tmp_path):
    """When the model's test_path is real, no fallback should happen."""
    from agentic_sdlc.tools.registry import run_tests

    custom_dir = tmp_path / "custom_tests"
    custom_dir.mkdir()
    (custom_dir / "test_dummy.py").write_text("def test_ok():\n    assert True\n")

    result = await run_tests(repo_path=str(tmp_path), test_path="custom_tests")

    assert result["passed"] is True
    assert "warning" not in result