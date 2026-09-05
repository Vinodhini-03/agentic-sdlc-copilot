"""
Single place that turns config into a concrete LLMProvider. This is the
ONLY function the rest of the app should call to get a provider — it's
what makes "switch vendors" a one-line config change instead of a refactor.
"""
from __future__ import annotations

import os

from agentic_sdlc.providers.base import LLMProvider


def get_provider(name: str | None = None, **kwargs) -> LLMProvider:
    """
    name: "ollama" (default, free/local), "anthropic", or "openai".
    Falls back to the AGENTIC_SDLC_PROVIDER env var, then "ollama".
    """
    name = (name or os.getenv("AGENTIC_SDLC_PROVIDER") or "ollama").lower()

    if name == "ollama":
        from agentic_sdlc.providers.ollama_provider import OllamaProvider
        return OllamaProvider(**kwargs)

    if name == "anthropic":
        from agentic_sdlc.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider(**kwargs)

    if name == "openai":
        from agentic_sdlc.providers.openai_provider import OpenAIProvider
        return OpenAIProvider(**kwargs)

    raise ValueError(f"Unknown provider '{name}'. Expected: ollama, anthropic, openai.")
