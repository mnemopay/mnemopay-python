"""
Tests for MnemoPay Governance modules (Policies, Risk, Audit Chain, and Action Ledger).
"""

import os
import json
import tempfile
from datetime import datetime, timedelta, timezone
import pytest

from mnemopay.governance.policy import (
    PolicyAction,
    PolicyRule,
    Policy,
    compile_policy,
    evaluate_action,
    InMemoryRateCounter,
)
from mnemopay.governance.risk import (
    RiskPolicyOptions,
    classify_risk,
    build_risk_policy,
)
from mnemopay.governance.audit_chain import (
    AuditChain,
    canonicalize,
    sha256_hex,
    verify_bundle,
)
from mnemopay.governance.action_ledger import (
    ActionLedger,
    BeginActionInput,
    ActionUpdate,
)


class TestPolicyEngine:
    def test_compile_and_outright_block(self):
        rule = PolicyRule(id="rule-1", outright_block=True, target_in=["dangerous_tool"])
        policy = Policy(id="pol-1", version=1, rules=[rule])
        compiled = compile_policy(policy)

        action_allowed = PolicyAction(kind="tool_call", target="safe_tool")
        action_blocked = PolicyAction(kind="tool_call", target="dangerous_tool")

        v_allowed = evaluate_action(compiled, action_allowed)
        assert v_allowed.allowed is True

        v_blocked = evaluate_action(compiled, action_blocked)
        assert v_blocked.allowed is False
        assert v_blocked.matched_rule == "rule-1"
        assert v_blocked.reason == "outright_block"

    def test_arg_pattern_blocking(self):
        rule = PolicyRule(id="rule-arg", arg_pattern_blocks="drop.*table")
        policy = Policy(id="pol-2", version=1, rules=[rule])
        compiled = compile_policy(policy)

        a_safe = PolicyAction(kind="tool_call", target="sql", args_text="select * from users")
        a_blocked = PolicyAction(kind="tool_call", target="sql", args_text="drop table users")

        assert evaluate_action(compiled, a_safe).allowed is True
        v = evaluate_action(compiled, a_blocked)
        assert v.allowed is False
        assert v.matched_rule == "rule-arg"

    def test_locale_boundaries(self):
        rule = PolicyRule(id="eu-only", allow_only_locales=["FR", "DE"])
        policy = Policy(id="pol-3", version=1, rules=[rule])
        compiled = compile_policy(policy)

        a_eu = PolicyAction(kind="tool_call", target="run", locale="FR")
        a_us = PolicyAction(kind="tool_call", target="run", locale="US")

        assert evaluate_action(compiled, a_eu).allowed is True
        assert evaluate_action(compiled, a_us).allowed is False

    def test_spend_limits_and_approvals(self):
        rule = PolicyRule(id="spend-limits", hard_cap_usd=100.0, approval_threshold_usd=10.0)
        policy = Policy(id="pol-4", version=1, rules=[rule])
        compiled = compile_policy(policy)

        a_low = PolicyAction(kind="payment", target="pay", estimated_usd=5.0)
        a_mid = PolicyAction(kind="payment", target="pay", estimated_usd=25.0)
        a_high = PolicyAction(kind="payment", target="pay", estimated_usd=150.0)

        assert evaluate_action(compiled, a_low).allowed is True

        v_mid = evaluate_action(compiled, a_mid)
        assert v_mid.allowed is False
        assert v_mid.needs_approval is True
        assert v_mid.reason == "approval_threshold_usd"

        v_high = evaluate_action(compiled, a_high)
        assert v_high.allowed is False
        assert v_high.needs_approval is False
        assert v_high.reason == "hard_cap_usd"


class TestRateCounter:
    def test_rate_counter_observes_and_prunes(self):
        counter = InMemoryRateCounter()
        now = datetime.now(timezone.utc).timestamp() * 1000

        counter.observe("key-1", now - 500)
        counter.observe("key-1", now - 200)
        counter.observe("key-1", now)

        # 3 observations within last 1 second (1000ms)
        assert counter.count_within("key-1", 1000, now) == 3

        # Prune older than 300ms ago
        counter.prune(now - 300)
        assert counter.count_within("key-1", 1000, now) == 2


class TestRiskTaxonomy:
    def test_risk_classification(self):
        a_low = PolicyAction(kind="llm_call", target="gpt-4")
        assessment_low = classify_risk(a_low)
        assert assessment_low.level == "low"

        a_esc = PolicyAction(kind="tool_call", target="execute", args_text="wire transfer money to client")
        assessment_esc = classify_risk(a_esc)
        assert assessment_esc.level == "critical"
        assert assessment_esc.rationale == "irreversible money/contract/destructive action"

    def test_risk_value_escalation(self):
        a_pay_low = PolicyAction(kind="tool_call", target="merchant", estimated_usd=5.0)
        a_pay_high = PolicyAction(kind="tool_call", target="merchant", estimated_usd=200.0)
    
        assert classify_risk(a_pay_low).level == "medium"
        assert classify_risk(a_pay_high).level == "high"


class TestAuditChain:
    def test_canonicalize_sorts_keys(self):
        d1 = {"z": 1, "a": 2, "m": {"y": 3, "x": 4}}
        d2 = {"a": 2, "z": 1, "m": {"x": 4, "y": 3}}
        assert canonicalize(d1) == canonicalize(d2)
        assert canonicalize(d1) == '{"a":2,"m":{"x":4,"y":3},"z":1}'

    def test_merkle_root_generation(self):
        chain = AuditChain()
        chain.emit("test.start", {"info": "starting"})
        chain.emit("test.action", {"step": 1})
        root = chain.roll_merkle_root()
        assert len(root) == 64  # valid sha256 hex root

        bundle = chain.to_bundle(meta={"run_id": "test-run"})
        assert bundle.merkle_root == root
        assert verify_bundle(bundle)["ok"] is True

    def test_file_appending_audit_chain(self):
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp_path = tmp.name

        try:
            chain = AuditChain(path=tmp_path)
            chain.emit("test.event", {"val": 42})
            chain.emit("test.event2", {"val": 84})

            with open(tmp_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                assert len(lines) == 2
                event1 = json.loads(lines[0])
                assert event1["kind"] == "test.event"
                assert event1["payload"]["val"] == 42
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class TestActionLedger:
    def test_ledger_lifecycle(self):
        ledger = ActionLedger()
        rec = ledger.begin(BeginActionInput(agent_id="agt-1", intent="fix a bug", plan="read index.py"))

        assert rec.status == "planned"
        assert rec.plan == "read index.py"

        ledger.update(rec.id, ActionUpdate(tools_used=["read_file"], cost_usd=0.005))
        updated = ledger.get(rec.id)
        assert "read_file" in updated.tools_used
        assert updated.cost_usd == 0.005

        ledger.await_approval(rec.id, "app-123")
        assert updated.status == "awaiting_approval"

        ledger.resolve_approval(rec.id, "app-123", "approved", "admin-1")
        assert updated.approvals[0].status == "approved"

        ledger.mark_executing(rec.id)
        assert updated.status == "executing"

        ledger.complete(rec.id, "success")
        assert updated.status == "completed"
        assert updated.result == "success"
