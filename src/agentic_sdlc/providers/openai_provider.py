"""
OpenAI adapter. Optional — install with `pip install agentic-sdlc[openai]`
and set OPENAI_API_KEY.
"""
from __future__ import annotations

import json
from typing import Any

from agentic_sdlc.providers.base import ModelResponse, ToolCall, ToolSpec


class OpenAIProvider:
    name = "openai"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        try:
            import openai
        except ImportError as e:
            raise ImportError(
                "Install the 'openai' extra: pip install agentic-sdlc[openai]"
            ) from e
        self.model = model
        self.embed_model = embed_model
        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        full_messages = ([{"role": "system", "content": system}] if system else []) + messages
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]

        resp = await self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]
        msg = choice.message

        tool_calls = []
        for tc in msg.tool_calls or []:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"_raw": tc.function.arguments}
            tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        return ModelResponse(
            text=msg.content or "",
            tool_calls=tool_calls,
            raw=resp,
            usage={
                "input_tokens": resp.usage.prompt_tokens if resp.usage else 0,
                "output_tokens": resp.usage.completion_tokens if resp.usage else 0,
            },
            stop_reason=choice.finish_reason,
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._client.embeddings.create(model=self.embed_model, input=texts)
        return [d.embedding for d in resp.data]
