"""
FastAPI service exposing the agent graph over HTTP so any client (GitHub App,
Slack bot, CI job, internal dashboard) can trigger a run without importing
Python directly. Run with:

    uvicorn agentic_sdlc.api.server:app --reload
"""
from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agentic_sdlc.governance.guardrails import GovernanceGate
from agentic_sdlc.graph.build import build_agent_graph
from agentic_sdlc.providers.factory import get_provider
from agentic_sdlc.tools.registry import default_registry

app = FastAPI(title="Agentic SDLC Copilot", version="0.1.0")

_provider = get_provider()  # defaults to Ollama; set AGENTIC_SDLC_PROVIDER to switch
_registry = default_registry()
_gate = GovernanceGate()
_graph = build_agent_graph(_provider, _registry, _gate)


class RunRequest(BaseModel):
    repo_path: str
    task: str
    actor_role: str = "agent"
    human_approved: bool = False


class RunResponse(BaseModel):
    trace_id: str
    status: str
    output: str | None = None
    human_reason: str | None = None


@app.post("/runs", response_model=RunResponse)
async def create_run(req: RunRequest) -> RunResponse:
    trace_id = str(uuid.uuid4())
    initial_state = {
        "repo_path": req.repo_path,
        "task": req.task,
        "actor_role": req.actor_role,
        "human_approved": req.human_approved,
        "messages": [],
        "trace_id": trace_id,
    }
    try:
        result = await _graph.ainvoke(initial_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    if result.get("needs_human"):
        return RunResponse(
            trace_id=trace_id,
            status="pending_human_approval",
            human_reason=result.get("human_reason"),
        )
    return RunResponse(trace_id=trace_id, status="completed", output=result.get("final_output"))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "provider": _provider.name}
