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
