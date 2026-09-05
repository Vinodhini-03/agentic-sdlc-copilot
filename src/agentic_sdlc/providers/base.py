"""
Platform-agnostic LLM provider interface.

Every concrete provider (Ollama, Anthropic, OpenAI, ...) implements this
Protocol. Nothing else in the codebase should import a vendor SDK directly —
only files inside `providers/` are allowed to know that Anthropic, OpenAI,
or Ollama exist. This is what keeps the rest of the framework swappable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolCall:
    """A single tool invocation requested by the model."""
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelResponse:
    """Normalized response shape, regardless of which vendor produced it."""
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None  # original provider payload, kept for debugging/tracing
    usage: dict[str, int] = field(default_factory=dict)  # {"input_tokens": .., "output_tokens": ..}
    stop_reason: str | None = None


@dataclass
class ToolSpec:
    """Vendor-neutral tool/function definition."""
    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


@runtime_checkable
class LLMProvider(Protocol):
    """
    Every provider adapter must implement this exact surface.
    Swapping providers should require changing ONE config value, never
    touching graph/tools/memory code.
    """

    name: str  # e.g. "ollama", "anthropic", "openai" — used in logs/traces

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        """Single non-streaming completion. Must normalize to ModelResponse."""
        ...

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embeddings for a batch of texts, for RAG indexing/retrieval."""
        ...
