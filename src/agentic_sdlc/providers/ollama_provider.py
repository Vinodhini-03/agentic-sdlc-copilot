"""
Ollama adapter — runs models locally (llama3.1, qwen2.5-coder, etc.) at zero
API cost. This is the default provider so the framework is usable without
any paid API key. Swap to Anthropic/OpenAI adapters for production quality
by changing config only — no other module needs to change.

Requires a running Ollama daemon: https://ollama.com (e.g. `ollama serve`,
`ollama pull llama3.1`).
"""
from __future__ import annotations

import json
from typing import Any

import httpx

from agentic_sdlc.providers.base import ModelResponse, ToolCall, ToolSpec


class OllamaProvider:
    name = "ollama"

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str = "http://localhost:11434",
        embed_model: str = "nomic-embed-text",
        timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.embed_model = embed_model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    async def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        system: str | None = None,
        tools: list[ToolSpec] | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> ModelResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": ([{"role": "system", "content": system}] if system else []) + messages,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        if tools:
            payload["tools"] = [
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

        resp = await self._client.post("/api/chat", json=payload)
        resp.raise_for_status()
        data = resp.json()

        message = data.get("message", {})
        tool_calls = []
        for tc in message.get("tool_calls", []) or []:
            fn = tc.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {"_raw": args}
            tool_calls.append(ToolCall(id=tc.get("id", fn.get("name", "")), name=fn.get("name", ""), arguments=args))

        return ModelResponse(
            text=message.get("content", "") or "",
            tool_calls=tool_calls,
            raw=data,
            usage={
                "input_tokens": data.get("prompt_eval_count", 0),
                "output_tokens": data.get("eval_count", 0),
            },
            stop_reason=data.get("done_reason"),
        )

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            resp = await self._client.post(
                "/api/embeddings", json={"model": self.embed_model, "prompt": text}
            )
            resp.raise_for_status()
            out.append(resp.json()["embedding"])
        return out

    async def aclose(self) -> None:
        await self._client.aclose()
