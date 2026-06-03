"""
MnemoPay Action Ledger -- structured, typed agent action records ("MnemoLedger").

Builds on top of AuditChain to record intent, plans, tools, cost, approvals, and outcomes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional

from .audit_chain import AuditChain, ChainEvent
from .risk import RiskLevel

ActionStatus = Literal[
    "planned",
    "awaiting_approval",
    "executing",
    "completed",
    "failed",
    "blocked",
    "rolled_back",
]


@dataclass
class ActionApproval:
    approval_id: str
    status: Literal["pending", "approved", "rejected", "expired"]
    decided_by: Optional[str] = None


@dataclass
class AgentActionRecord:
    id: str
    agent_id: str
    intent: str
    status: ActionStatus
    tools_used: List[str] = field(default_factory=list)
    memories_used: List[str] = field(default_factory=list)
    files_accessed: List[str] = field(default_factory=list)
    sites_visited: List[str] = field(default_factory=list)
    approvals: List[ActionApproval] = field(default_factory=list)
    cost_usd: float = 0.0
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
    plan: Optional[str] = None
    risk: Optional[RiskLevel] = None
    rollback: Optional[str] = None
    result: Optional[str] = None
    error: Optional[str] = None
    ended_at: Optional[str] = None


@dataclass
class BeginActionInput:
    agent_id: str
    intent: str
    plan: Optional[str] = None
    risk: Optional[RiskLevel] = None


@dataclass
class ActionUpdate:
    tools_used: Optional[List[str]] = None
    memories_used: Optional[List[str]] = None
    files_accessed: Optional[List[str]] = None
    sites_visited: Optional[List[str]] = None
    cost_usd: Optional[float] = None
    plan: Optional[str] = None
    rollback: Optional[str] = None


EVENT_PREFIX = "action."


class ActionLedger:
    """Typed action ledger wrapping an AuditChain to trace the entire lifecycle of agent actions."""

    def __init__(self, chain: Optional[AuditChain] = None) -> None:
        self.chain = chain or AuditChain()
        self._actions: Dict[str, AgentActionRecord] = {}

    def audit_chain(self) -> AuditChain:
        return self.chain

    def begin(self, input_data: BeginActionInput) -> AgentActionRecord:
        rec = AgentActionRecord(
            id=str(uuid.uuid4()),
            agent_id=input_data.agent_id,
            intent=input_data.intent,
            status="planned",
            plan=input_data.plan,
            risk=input_data.risk,
        )
        self._actions[rec.id] = rec

        payload = {
            "action_id": rec.id,
            "agent_id": rec.agent_id,
            "intent": rec.intent,
        }
        if rec.plan:
            payload["plan"] = rec.plan
        if rec.risk:
            payload["risk"] = rec.risk

        self.chain.emit(f"{EVENT_PREFIX}begin", payload)
        return rec

    def update(self, action_id: str, patch: ActionUpdate) -> AgentActionRecord:
        rec = self._require(action_id)
        if patch.tools_used:
            rec.tools_used = _dedupe(rec.tools_used + patch.tools_used)
        if patch.memories_used:
            rec.memories_used = _dedupe(rec.memories_used + patch.memories_used)
        if patch.files_accessed:
            rec.files_accessed = _dedupe(rec.files_accessed + patch.files_accessed)
        if patch.sites_visited:
            rec.sites_visited = _dedupe(rec.sites_visited + patch.sites_visited)
        if patch.cost_usd is not None:
            rec.cost_usd += patch.cost_usd
        if patch.plan:
            rec.plan = patch.plan
        if patch.rollback:
            rec.rollback = patch.rollback

        payload: Dict[str, Any] = {"action_id": action_id}
        if patch.tools_used:
            payload["tools_used"] = patch.tools_used
        if patch.memories_used:
            payload["memories_used"] = patch.memories_used
        if patch.files_accessed:
            payload["files_accessed"] = patch.files_accessed
        if patch.sites_visited:
            payload["sites_visited"] = patch.sites_visited
        if patch.cost_usd is not None:
            payload["cost_usd"] = patch.cost_usd
        if patch.plan:
            payload["plan"] = patch.plan
        if patch.rollback:
            payload["rollback"] = patch.rollback

        self.chain.emit(f"{EVENT_PREFIX}update", payload)
        return rec

    def await_approval(self, action_id: str, approval_id: str) -> AgentActionRecord:
        rec = self._require(action_id)
        rec.status = "awaiting_approval"
        rec.approvals.append(ActionApproval(approval_id=approval_id, status="pending"))
        self.chain.emit(
            f"{EVENT_PREFIX}approval.requested",
            {"action_id": action_id, "approval_id": approval_id},
        )
        return rec

    def resolve_approval(
        self,
        action_id: str,
        approval_id: str,
        status: Literal["approved", "rejected", "expired"],
        decided_by: Optional[str] = None,
    ) -> AgentActionRecord:
        rec = self._require(action_id)
        for ap in rec.approvals:
            if ap.approval_id == approval_id:
                ap.status = status
                if decided_by:
                    ap.decided_by = decided_by
                break

        payload = {
            "action_id": action_id,
            "approval_id": approval_id,
            "status": status,
        }
        if decided_by:
            payload["decided_by"] = decided_by

        self.chain.emit(f"{EVENT_PREFIX}approval.resolved", payload)
        return rec

    def mark_executing(self, action_id: str) -> AgentActionRecord:
        rec = self._require(action_id)
        rec.status = "executing"
        self.chain.emit(f"{EVENT_PREFIX}executing", {"action_id": action_id})
        return rec

    def complete(self, action_id: str, result: Optional[str] = None) -> AgentActionRecord:
        return self._end(action_id, "completed", result=result)

    def fail(self, action_id: str, error: str) -> AgentActionRecord:
        return self._end(action_id, "failed", error=error)

    def block(self, action_id: str, reason: str) -> AgentActionRecord:
        return self._end(action_id, "blocked", error=reason)

    def roll_back(self, action_id: str, note: Optional[str] = None) -> AgentActionRecord:
        return self._end(action_id, "rolled_back", result=note)

    def _end(
        self,
        action_id: str,
        status: ActionStatus,
        result: Optional[str] = None,
        error: Optional[str] = None,
    ) -> AgentActionRecord:
        rec = self._require(action_id)
        rec.status = status
        if result is not None:
            rec.result = result
        if error is not None:
            rec.error = error
        rec.ended_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        payload = {
            "action_id": action_id,
            "status": status,
            "cost_usd": rec.cost_usd,
        }
        if result is not None:
            payload["result"] = result
        if error is not None:
            payload["error"] = error

        self.chain.emit(f"{EVENT_PREFIX}end", payload)
        return rec

    def get(self, action_id: str) -> Optional[AgentActionRecord]:
        return self._actions.get(action_id)

    def list(self, agent_id: Optional[str] = None) -> List[AgentActionRecord]:
        all_actions = list(self._actions.values())
        if agent_id:
            return [a for a in all_actions if a.agent_id == agent_id]
        return all_actions

    def events(self) -> List[ChainEvent]:
        return self.chain.events()

    def _require(self, action_id: str) -> AgentActionRecord:
        rec = self._actions.get(action_id)
        if not rec:
            raise ValueError(f"action-ledger: {action_id} not found")
        return rec


def _dedupe(arr: List[str]) -> List[str]:
    # Deduplicate while preserving order
    seen = set()
    result = []
    for x in arr:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result
