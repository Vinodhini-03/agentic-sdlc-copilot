"""
Node implementations for the SDLC agent graph:

  intake -> plan -> act (tools) -> critique -> human_gate -> respond

Each node is a plain async function (state) -> partial_state_update, which
is what LangGraph expects. Nodes only depend on the LLMProvider / ToolRegistry
/ GovernanceGate protocols — never on a specific vendor.
"""
from __future__ import annotations

import structlog

from agentic_sdlc.governance.guardrails import GovernanceGate, PermissionDenied
from agentic_sdlc.graph.state import AgentState
from agentic_sdlc.providers.base import LLMProvider, ToolSpec
from agentic_sdlc.tools.registry import ToolRegistry

log = structlog.get_logger()

SYSTEM_PROMPT = """You are an SDLC engineering agent. You review code changes, \
run tests, and triage issues. You have access to tools for reading the repo \
and running its test suite. Be precise, cite file/line when giving feedback, \
and never claim a test passed without actually running it. If an action would \
merge, deploy, or delete anything, you must say so explicitly and rely on \
the human approval gate — do not attempt to bypass it."""


async def intake_node(state: AgentState, **_) -> dict:
    log.info("intake", task=state.get("task"), repo=state.get("repo_path"))
    return {
        "messages": state.get("messages", []) + [
            {"role": "user", "content": f"Task: {state['task']}\nRepo: {state['repo_path']}"}
        ],
        "needs_human": False,
    }


def make_plan_node(provider: LLMProvider, tool_specs: list[ToolSpec]):
    async def plan_node(state: AgentState, **_) -> dict:
        response = await provider.complete(
            state["messages"],
            system=SYSTEM_PROMPT,
            tools=tool_specs,
            max_tokens=800,
        )
        pending = [{"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in response.tool_calls]
        return {
            "plan": response.text,
            "pending_tool_calls": pending,
            "messages": state["messages"] + [{"role": "assistant", "content": response.text}],
        }
    return plan_node


def make_act_node(registry: ToolRegistry, gate: GovernanceGate):
    async def act_node(state: AgentState, **_) -> dict:
        results = []
        needs_human, human_reason = False, ""
        for call in state.get("pending_tool_calls", []):
            try:
                gate.authorize(
                    actor_role=state.get("actor_role", "agent"),
                    tool_name=call["name"],
                    arguments=call["arguments"],
                    human_approved=state.get("human_approved", False),
                )
                result = await registry.call(call["name"], call["arguments"])
                results.append({"tool": call["name"], "ok": True, "result": result})
            except PermissionDenied as e:
                needs_human = True
                human_reason = str(e)
                results.append({"tool": call["name"], "ok": False, "error": str(e)})
        return {"tool_results": results, "needs_human": needs_human, "human_reason": human_reason}
    return act_node


def make_critique_node(provider: LLMProvider):
    async def critique_node(state: AgentState, **_) -> dict:
        summary = "\n".join(
            f"- {r['tool']}: {'ok' if r['ok'] else 'FAILED - ' + r.get('error', '')}"
            for r in state.get("tool_results", [])
        )
        prompt = (
            f"Tool results:\n{summary}\n\n"
            "In 2-4 sentences, critique whether the plan's evidence actually "
            "supports the conclusion. Flag any unverified claims."
        )
        response = await provider.complete(
            state["messages"] + [{"role": "user", "content": prompt}],
            system=SYSTEM_PROMPT,
            max_tokens=300,
        )
        return {"critique": response.text}
    return critique_node


async def human_gate_node(state: AgentState, **_) -> dict:
    """
    Terminal node when a high-risk action is blocked. In the FastAPI service
    this maps to a 202 response with a pending audit-log id; a human approves
    via API and the graph is re-invoked with human_approved=True.
    """
    log.warning("human_gate", reason=state.get("human_reason"))
    return {
        "final_output": (
            "This run requires human approval before proceeding: "
            f"{state.get('human_reason')}"
        )
    }


async def respond_node(state: AgentState, **_) -> dict:
    output = (
        f"## Plan\n{state.get('plan', '')}\n\n"
        f"## Tool Results\n{state.get('tool_results', [])}\n\n"
        f"## Critique\n{state.get('critique', '')}\n"
    )
    return {"final_output": output}


def route_after_act(state: AgentState) -> str:
    return "human_gate" if state.get("needs_human") else "critique"
