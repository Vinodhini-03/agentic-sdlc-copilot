"""
Assembles the LangGraph state machine:

    intake -> plan -> act -> [critique | human_gate] -> respond

This is the one place that imports `langgraph` — if you ever swap orchestration
frameworks, only this file changes; nodes.py stays framework-agnostic.
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from agentic_sdlc.governance.guardrails import GovernanceGate
from agentic_sdlc.graph.nodes import (
    human_gate_node,
    intake_node,
    make_act_node,
    make_critique_node,
    make_plan_node,
    respond_node,
    route_after_act,
)
from agentic_sdlc.graph.state import AgentState
from agentic_sdlc.providers.base import LLMProvider
from agentic_sdlc.tools.registry import ToolRegistry


def build_agent_graph(
    provider: LLMProvider,
    registry: ToolRegistry,
    gate: GovernanceGate | None = None,
):
    gate = gate or GovernanceGate()
    tool_specs = registry.specs()

    graph = StateGraph(AgentState)
    graph.add_node("intake", intake_node)
    graph.add_node("plan", make_plan_node(provider, tool_specs))
    graph.add_node("act", make_act_node(registry, gate))
    graph.add_node("critique", make_critique_node(provider))
    graph.add_node("human_gate", human_gate_node)
    graph.add_node("respond", respond_node)

    graph.set_entry_point("intake")
    graph.add_edge("intake", "plan")
    graph.add_edge("plan", "act")
    graph.add_conditional_edges("act", route_after_act, {
        "critique": "critique",
        "human_gate": "human_gate",
    })
    graph.add_edge("critique", "respond")
    graph.add_edge("human_gate", END)
    graph.add_edge("respond", END)

    return graph.compile()
