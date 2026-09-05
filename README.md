# Agentic SDLC Copilot

A platform-agnostic, governed agentic framework for automating software
development workflows — code review, test execution, and triage — built on
LangGraph, with pluggable LLM providers and a human-in-the-loop governance
layer.

```
intake → plan → act (tools) → [critique | human_gate] → respond
```

## Why this exists

Most "AI agent" demos are a single LangChain call with no tests, no
observability, and a hard dependency on one vendor's API. This project is
the opposite: a small, typed, tested framework where the LLM is a
swappable component, every tool call is governed, and every run is
auditable.

## Architecture

```
src/agentic_sdlc/
├── providers/       # LLMProvider Protocol + Ollama/Anthropic/OpenAI adapters
│   ├── base.py          # vendor-neutral interface (ModelResponse, ToolSpec, ...)
│   ├── ollama_provider.py    # free, local default
│   ├── anthropic_provider.py
│   ├── openai_provider.py
│   └── factory.py        # get_provider() — the ONLY place vendors are chosen
├── tools/           # function-calling tools (fetch_diff, run_tests, search_codebase)
├── graph/           # LangGraph state machine (state, nodes, build)
├── governance/      # guardrails, RBAC, audit log, human-approval gate
├── memory/          # SQLite-backed per-repo decision memory
├── eval/            # golden-dataset eval harness for regression-testing prompts
└── api/             # FastAPI service exposing the agent over HTTP
```

**Key design rule:** only files under `providers/` import a vendor SDK
(`anthropic`, `openai`) or talk to Ollama's HTTP API. Everything else
(`graph/`, `tools/`, `governance/`) depends only on the `LLMProvider`
Protocol in `providers/base.py`. Swapping GPT-4o for Claude or a local
Llama model is a one-line config change (`AGENTIC_SDLC_PROVIDER=anthropic`),
never a refactor.

## Quickstart (free, local — no API key)

```bash
# 1. Install Ollama and pull a model: https://ollama.com
ollama pull llama3.1
ollama serve &

# 2. Install this package
pip install -e ".[dev]"

# 3. Run the test suite (uses a FakeProvider, no Ollama needed for tests)
pytest -v

# 4. Start the API (talks to your local Ollama by default)
uvicorn agentic_sdlc.api.server:app --reload
```

Trigger a run:

```bash
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d '{"repo_path": ".", "task": "review_pr", "actor_role": "agent"}'
```

## Switching providers

```bash
export AGENTIC_SDLC_PROVIDER=anthropic   # or openai, or ollama (default)
export ANTHROPIC_API_KEY=sk-...          # pip install -e ".[anthropic]" first
```

No other code changes. See `providers/factory.py`.

## Governance model

Every tool call passes through `GovernanceGate.authorize()`:

1. **Content guardrails** — regex-level block on obviously destructive
   payloads (`rm -rf /`, `DROP TABLE`, fork bombs), independent of role.
2. **Risk classification** — each tool is `read_only`, `low_risk`, or
   `high_risk` (unknown tools default to `high_risk` — fail closed).
3. **RBAC** — roles (`agent`, `reviewer`, `admin`) define the max risk
   level they may auto-execute.
4. **Human-in-the-loop** — anything above a role's ceiling routes the
   graph to `human_gate` instead of executing, and the API returns
   `pending_human_approval` with a reason. Re-invoke with
   `human_approved=true` after a human signs off.
5. **Audit log** — every authorization attempt (approved, blocked, or
   escalated) is appended to `audit_log.jsonl`, including who/what/when.

See `governance/guardrails.py` and `tests/test_guardrails.py`.

## Evaluation

`eval/harness.py` runs a golden dataset of prompts against the configured
provider and scores outputs against simple checks (substring match today;
swap in an LLM-judge or LangSmith dataset for production use). Run before
merging any prompt change:

```bash
python -m agentic_sdlc.eval.harness
```

For production tracing, wrap provider calls with LangSmith's `@traceable`
decorator (`pip install -e ".[tracing]"`, set `LANGCHAIN_TRACING_V2=true`).

## CI/CD

- `.github/workflows/ci.yml` — lint (ruff), type-check (mypy), test with
  coverage across Python 3.10–3.12.
- `.github/workflows/agent-pr-review.yml` — the agent *augments* CI: on
  every PR it fetches the diff, runs tests, and posts a review comment.
  Any high-risk action (merging, deploying) is blocked by the governance
  gate and requires a human — the workflow demonstrates augmentation,
  not autonomous merging.

## Extending

- **New provider**: implement `LLMProvider` in `providers/your_provider.py`,
  register it in `providers/factory.py`. Nothing else changes.
- **New tool**: write an async function + `ToolSpec`, register with
  `ToolRegistry`, add a `TOOL_RISK` entry in `governance/guardrails.py`.
- **New task**: add a node to `graph/nodes.py` and wire it into
  `graph/build.py`.

## License

MIT — see `LICENSE`.
