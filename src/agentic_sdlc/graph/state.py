"""
Shared state passed between LangGraph nodes. Kept as a TypedDict so it's
trivially serializable (for tracing/checkpointing) and provider-agnostic.
"""
from __future__ import annotations

from typing import Any, TypedDict


class AgentState(TypedDict, total=False):
    repo_path: str
    task: str                       # e.g. "review_pr", "triage_incident"
    actor_role: str                 # governance role of whoever triggered this run
    human_approved: bool

    messages: list[dict[str, Any]]  # running conversation
    retrieved_context: list[str]    # RAG snippets
    pending_tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]

    plan: str
    critique: str
    final_output: str

    needs_human: bool
    human_reason: str
    trace_id: str
