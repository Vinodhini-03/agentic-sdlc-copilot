"""
Anthropic adapter. Optional — install with `pip install agentic-sdlc[anthropic]`
and set ANTHROPIC_API_KEY. Drop-in replacement for OllamaProvider; nothing
outside providers/ knows this file exists.
"""
from __future__ import annotations

from typing import Any

from agentic_sdlc.providers.base import ModelResponse, ToolCall, ToolSpec


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None) -> None:
        try:
            import anthropic
        except ImportError as e:
            raise ImportError(
                "Install the 'anthropic' extra: pip install agentic-sdlc[anthropic]"
            ) from e
        self.model = model
        self._client = anthropic.AsyncAnthropic(api_key=api_key)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {"name": t.name, "description": t.description, "input_schema": t.parameters}
                for t in tools
            ]

        resp = await self._client.messages.create(**kwargs)

        text_parts = []
        tool_calls = []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, arguments=block.input))

        return ModelResponse(
            text="".join(text_parts),
            tool_calls=tool_calls,
            raw=resp,
            usage={
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            },
            stop_reason=resp.stop_reason,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        # Anthropic has no native embeddings endpoint; use Voyage AI (recommended
        # partner) or another embed_provider for RAG when running on Anthropic.
        raise NotImplementedError(
            "AnthropicProvider has no embeddings. Configure a separate "
            "embed_provider (e.g. OllamaProvider or Voyage) in ProviderConfig."
        )
