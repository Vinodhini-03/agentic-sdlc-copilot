import pytest

from agentic_sdlc.governance.guardrails import AuditLog, GovernanceGate
from agentic_sdlc.graph.build import build_agent_graph
from agentic_sdlc.providers.base import ModelResponse, ToolCall, ToolSpec
from agentic_sdlc.tools.registry import ToolRegistry
from tests.fakes import FakeProvider


@pytest.mark.asyncio
async def test_happy_path_read_only_tool_completes_without_human(tmp_path):
    registry = ToolRegistry()

    async def fetch_diff(repo_path: str) -> str:
        return "diff --git a/x.py b/x.py\n+print('hi')"

    registry.register(
        ToolSpec(name="fetch_diff", description="get diff", parameters={"type": "object"}),
        fetch_diff,
    )

    plan_response = ModelResponse(
        text="I'll fetch the diff first.",
        tool_calls=[ToolCall(id="1", name="fetch_diff", arguments={"repo_path": "."})],
    )
    critique_response = ModelResponse(text="The diff was small and low risk.")
    provider = FakeProvider(responses=[plan_response, critique_response])

    gate = GovernanceGate(audit_log=AuditLog(path=tmp_path / "audit.jsonl"))
    graph = build_agent_graph(provider, registry, gate)

    result = await graph.ainvoke({
        "repo_path": ".",
        "task": "review_pr",
        "actor_role": "agent",
        "messages": [],
    })

    assert result["needs_human"] is False
    assert "Critique" in result["final_output"]
    assert result["tool_results"][0]["ok"] is True


@pytest.mark.asyncio
async def test_high_risk_tool_routes_to_human_gate(tmp_path):
    registry = ToolRegistry()

    async def merge_pr(pr: int) -> str:
        return f"merged {pr}"

    registry.register(
        ToolSpec(name="merge_pr", description="merge a PR", parameters={"type": "object"}),
        merge_pr,
    )

    plan_response = ModelResponse(
        text="I'll merge the PR.",
        tool_calls=[ToolCall(id="1", name="merge_pr", arguments={"pr": 42})],
    )
    provider = FakeProvider(responses=[plan_response])

    gate = GovernanceGate(audit_log=AuditLog(path=tmp_path / "audit.jsonl"))
    graph = build_agent_graph(provider, registry, gate)

    result = await graph.ainvoke({
        "repo_path": ".",
        "task": "merge_release",
        "actor_role": "agent",
        "human_approved": False,
        "messages": [],
    })

    assert result["needs_human"] is True
    assert "human approval" in result["final_output"].lower()
