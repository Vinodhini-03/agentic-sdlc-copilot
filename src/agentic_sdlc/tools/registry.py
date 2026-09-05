"""
Vendor-neutral tool registry. Tools are plain async Python functions with a
JSON-schema spec attached — they work identically regardless of which
LLMProvider is calling them.
"""
from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from agentic_sdlc.providers.base import ToolSpec


@dataclass
class RegisteredTool:
    spec: ToolSpec
    handler: Callable[..., Awaitable[Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, RegisteredTool] = {}

    def register(self, spec: ToolSpec, handler: Callable[..., Awaitable[Any]]) -> None:
        self._tools[spec.name] = RegisteredTool(spec=spec, handler=handler)

    def specs(self) -> list[ToolSpec]:
        return [t.spec for t in self._tools.values()]

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if name not in self._tools:
            raise KeyError(f"No tool registered with name '{name}'")
        return await self._tools[name].handler(**arguments)


# ---------------------------------------------------------------------------
# Concrete tool implementations. Keep these small and testable — the agent
# only ever sees the ToolSpec + return value, never internal implementation.
#
# subprocess.run is blocking, so every call here is offloaded to a thread via
# asyncio.to_thread — this keeps the async event loop free for concurrent
# tool calls / API requests instead of stalling on a single git/pytest run.
# ---------------------------------------------------------------------------

def _run_subprocess(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, check=False, **kwargs)


async def fetch_diff(repo_path: str, base: str = "main", head: str = "HEAD") -> str:
    """Return the unified diff between two refs in a local git repo."""
    result = await asyncio.to_thread(
        _run_subprocess, ["git", "-C", repo_path, "diff", f"{base}...{head}"]
    )
    return result.stdout or f"(no diff output; stderr: {result.stderr[:500]})"

async def run_tests(repo_path: str, test_path: str = "tests/") -> dict[str, Any]:
    """
    Run pytest in the target repo and return a structured pass/fail summary.

    Defensive validation: smaller/local LLMs sometimes hallucinate a
    plausible-looking placeholder path (e.g. "path/to/tests") instead of
    respecting the schema's default or omitting the argument. Rather than
    trusting the model's input blindly, verify the path exists under
    repo_path first and fall back to the real default with a warning note
    if not - this is what a production tool boundary should do regardless
    of which model is calling it.
    """
    from pathlib import Path

    resolved = Path(repo_path) / test_path
    fallback_used = False
    if not resolved.exists():
        fallback_used = True
        test_path = "tests/"
        resolved = Path(repo_path) / test_path

    result = await asyncio.to_thread(
        _run_subprocess,
        ["python", "-m", "pytest", test_path, "-q", "--tb=short"],
        cwd=repo_path,
    )
    output = {
        "returncode": result.returncode,
        "passed": result.returncode == 0,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-1000:],
    }
    if fallback_used:
        output["warning"] = (
            f"Requested test_path did not exist under repo; fell back to "
            f"default '{test_path}'."
        )
    return output
  


async def search_codebase(repo_path: str, query: str, max_results: int = 8) -> list[dict[str, str]]:
    """Very simple grep-based retrieval fallback (use RAG store for semantic search)."""
    result = await asyncio.to_thread(
        _run_subprocess, ["grep", "-rn", "--include=*.py", query, repo_path]
    )
    lines = result.stdout.splitlines()[:max_results]
    hits = []
    for line in lines:
        parts = line.split(":", 2)
        if len(parts) == 3:
            hits.append({"file": parts[0], "line": parts[1], "content": parts[2].strip()})
    return hits


def default_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="fetch_diff",
            description="Get the unified diff between two git refs in a repo.",
            parameters={
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string"},
                    "base": {"type": "string", "default": "main"},
                    "head": {"type": "string", "default": "HEAD"},
                },
                "required": ["repo_path"],
            },
        ),
        fetch_diff,
    )
    reg.register(
        ToolSpec(
            name="run_tests",
            description="Run the test suite in a repo and report pass/fail with output.",
            parameters={
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string"},
                    "test_path": {"type": "string", "default": "tests/"},
                },
                "required": ["repo_path"],
            },
        ),
        run_tests,
    )
    reg.register(
        ToolSpec(
            name="search_codebase",
            description="Keyword-search the codebase for relevant files/lines.",
            parameters={
                "type": "object",
                "properties": {
                    "repo_path": {"type": "string"},
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 8},
                },
                "required": ["repo_path", "query"],
            },
        ),
        search_codebase,
    )
    return reg