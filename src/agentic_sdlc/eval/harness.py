"""
Minimal eval harness: run the agent (or a single provider call) against a
golden dataset of (input, expected_criteria) pairs and score outputs. This
is what lets you change a prompt or swap providers without flying blind —
run `python -m agentic_sdlc.eval.harness` before merging prompt changes.

Swap in LangSmith for production-grade tracing/eval by wrapping `provider.complete`
with `@traceable` — this harness stays useful as a fast local sanity check.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from agentic_sdlc.providers.base import LLMProvider


@dataclass
class EvalCase:
    id: str
    input_messages: list[dict[str, Any]]
    system: str | None = None
    # each check receives the raw output text and returns True/False
    checks: list[Callable[[str], bool]] = field(default_factory=list)


@dataclass
class EvalResult:
    case_id: str
    passed: int
    total: int
    latency_s: float
    output: str


async def run_eval_suite(provider: LLMProvider, cases: list[EvalCase]) -> list[EvalResult]:
    results = []
    for case in cases:
        start = time.perf_counter()
        response = await provider.complete(case.input_messages, system=case.system)
        elapsed = time.perf_counter() - start
        passed = sum(1 for check in case.checks if check(response.text))
        results.append(
            EvalResult(
                case_id=case.id,
                passed=passed,
                total=len(case.checks),
                latency_s=elapsed,
                output=response.text,
            )
        )
    return results


def summarize(results: list[EvalResult]) -> dict[str, Any]:
    total_checks = sum(r.total for r in results)
    total_passed = sum(r.passed for r in results)
    return {
        "cases": len(results),
        "checks_passed": total_passed,
        "checks_total": total_checks,
        "pass_rate": round(total_passed / total_checks, 3) if total_checks else None,
        "avg_latency_s": round(sum(r.latency_s for r in results) / len(results), 3) if results else None,
    }


# --- example golden dataset -------------------------------------------------

def contains(substr: str) -> Callable[[str], bool]:
    return lambda text: substr.lower() in text.lower()


EXAMPLE_CASES = [
    EvalCase(
        id="review_mentions_file",
        input_messages=[
            {"role": "user", "content": "Review this diff:\n--- a/app.py\n+++ b/app.py\n+def f(): pass"}
        ],
        system="You are a terse code reviewer. Always cite the file you're reviewing.",
        checks=[contains("app.py")],
    ),
]


async def _main() -> None:
    from agentic_sdlc.providers.factory import get_provider

    provider = get_provider()
    results = await run_eval_suite(provider, EXAMPLE_CASES)
    print(json.dumps(summarize(results), indent=2))


if __name__ == "__main__":
    asyncio.run(_main())
