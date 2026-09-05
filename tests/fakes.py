from __future__ import annotations

from typing import Any

from agentic_sdlc.providers.base import ModelResponse, ToolSpec


class FakeProvider:
    """
    Deterministic stand-in for a real LLMProvider, used across tests so the
    graph/eval logic can be verified without network calls or a running
    Ollama daemon. Configure `responses` as a queue of ModelResponse objects
    to return on successive calls.
    """
    name = "fake"

    def __init__(self, responses: list[ModelResponse] | None = None) -> None:
        self.responses = responses or []
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages,
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        self.calls.append({"messages": messages, "system": system, "tools": tools})
        if self.responses:
            return self.responses.pop(0)
        return ModelResponse(text="ok", tool_calls=[])

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * 8 for _ in texts]
