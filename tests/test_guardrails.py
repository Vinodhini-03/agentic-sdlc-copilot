import pytest

from agentic_sdlc.governance.guardrails import (
    AuditLog,
    GovernanceGate,
    GuardrailViolation,
    PermissionDenied,
    RiskLevel,
)


@pytest.fixture()
def gate(tmp_path):
    return GovernanceGate(audit_log=AuditLog(path=tmp_path / "audit.jsonl"))


def test_read_only_tool_auto_authorized(gate):
    event = gate.authorize(actor_role="agent", tool_name="fetch_diff", arguments={"repo_path": "."})
    assert event.outcome == "authorized"
    assert event.risk == RiskLevel.READ_ONLY.value


def test_high_risk_tool_blocked_for_agent_role(gate):
    with pytest.raises(PermissionDenied):
        gate.authorize(actor_role="agent", tool_name="merge_pr", arguments={"pr": 1})


def test_high_risk_tool_allowed_with_human_approval(gate):
    event = gate.authorize(
        actor_role="agent", tool_name="merge_pr", arguments={"pr": 1}, human_approved=True
    )
    assert event.outcome == "authorized"
    assert event.approved_by == "human"


def test_reviewer_role_can_auto_execute_high_risk(gate):
    event = gate.authorize(actor_role="reviewer", tool_name="deploy", arguments={})
    assert event.outcome == "authorized"


def test_unknown_tool_defaults_to_high_risk_fail_closed(gate):
    with pytest.raises(PermissionDenied):
        gate.authorize(actor_role="agent", tool_name="some_unregistered_tool", arguments={})


def test_blocked_content_pattern_raises_guardrail_violation(gate):
    with pytest.raises(GuardrailViolation):
        gate.authorize(
            actor_role="admin",
            tool_name="run_tests",
            arguments={"repo_path": "x", "cmd": "rm -rf /"},
        )


def test_audit_log_records_every_call(gate, tmp_path):
    gate.authorize(actor_role="agent", tool_name="fetch_diff", arguments={"repo_path": "."})
    events = gate.audit_log.all_events()
    assert len(events) == 1
    assert events[0]["action"] == "fetch_diff"
