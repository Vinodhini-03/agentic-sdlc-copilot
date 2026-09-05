"""
Governance primitives: guardrails (input/output validation), role-based
access control, an append-only audit log, and a human-approval gate for
any action classified as destructive/high-risk.
"""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class RiskLevel(str, Enum):
    READ_ONLY = "read_only"      # e.g. search, fetch_diff — auto-approved
    LOW_RISK = "low_risk"        # e.g. posting a PR comment — auto-approved, logged
    HIGH_RISK = "high_risk"      # e.g. merge, deploy, delete — requires human approval


# Static risk classification for known tool names. Unknown tools default to
# HIGH_RISK — fail closed, never fail open.
TOOL_RISK: dict[str, RiskLevel] = {
    "fetch_diff": RiskLevel.READ_ONLY,
    "search_codebase": RiskLevel.READ_ONLY,
    "run_tests": RiskLevel.LOW_RISK,
    "create_pr_comment": RiskLevel.LOW_RISK,
    "merge_pr": RiskLevel.HIGH_RISK,
    "deploy": RiskLevel.HIGH_RISK,
    "delete_branch": RiskLevel.HIGH_RISK,
}

# Patterns that block a tool call outright, regardless of role or approval —
# a last-resort guardrail against obviously destructive shell content.
BLOCKED_PATTERNS = [
    re.compile(r"rm\s+-rf\s+/"),
    re.compile(r":\(\)\s*\{\s*:\|\s*:\s*&\s*\}\s*;"),  # fork bomb
    re.compile(r"DROP\s+TABLE", re.IGNORECASE),
]


class GuardrailViolation(Exception):
    pass


class PermissionDenied(Exception):
    pass


@dataclass
class Role:
    name: str
    max_risk: RiskLevel  # highest risk level this role may auto-execute


ROLES: dict[str, Role] = {
    "agent": Role(name="agent", max_risk=RiskLevel.LOW_RISK),
    "reviewer": Role(name="reviewer", max_risk=RiskLevel.HIGH_RISK),
    "admin": Role(name="admin", max_risk=RiskLevel.HIGH_RISK),
}


@dataclass
class AuditEvent:
    id: str
    ts: float
    actor: str
    action: str
    risk: str
    payload: dict[str, Any]
    approved_by: str | None = None
    outcome: str = "pending"


class AuditLog:
    """Append-only JSONL audit trail. Never mutate past entries."""

    def __init__(self, path: str | Path = "audit_log.jsonl") -> None:
        self.path = Path(path)

    def record(self, event: AuditEvent) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event.__dict__) + "\n")

    def all_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]


def check_content_guardrails(text: str) -> None:
    """Raise GuardrailViolation if text matches a blocked destructive pattern."""
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(text):
            raise GuardrailViolation(f"Blocked content matched pattern: {pattern.pattern}")


@dataclass
class GovernanceGate:
    """
    Central checkpoint every tool call passes through:
      1. content guardrail check
      2. risk classification
      3. RBAC check against the acting role
      4. if HIGH_RISK and role can't auto-execute -> require human approval
      5. audit log entry either way
    """
    audit_log: AuditLog = field(default_factory=AuditLog)

    def classify(self, tool_name: str) -> RiskLevel:
        return TOOL_RISK.get(tool_name, RiskLevel.HIGH_RISK)  # fail closed

    def authorize(
        self,
        *,
        actor_role: str,
        tool_name: str,
        arguments: dict[str, Any],
        human_approved: bool = False,
    ) -> AuditEvent:
        check_content_guardrails(json.dumps(arguments))

        risk = self.classify(tool_name)
        role = ROLES.get(actor_role)
        if role is None:
            raise PermissionDenied(f"Unknown role '{actor_role}'")

        risk_order = list(RiskLevel)
        needs_human = risk_order.index(risk) > risk_order.index(role.max_risk)

        event = AuditEvent(
            id=str(uuid.uuid4()),
            ts=time.time(),
            actor=actor_role,
            action=tool_name,
            risk=risk.value,
            payload=arguments,
        )

        if needs_human and not human_approved:
            event.outcome = "blocked_pending_human_approval"
            self.audit_log.record(event)
            raise PermissionDenied(
                f"Tool '{tool_name}' is {risk.value} and requires human approval "
                f"(role '{actor_role}' may only auto-execute up to {role.max_risk.value})."
            )

        event.outcome = "authorized"
        event.approved_by = "human" if (needs_human and human_approved) else actor_role
        self.audit_log.record(event)
        return event
