"""
MnemoPay Action Risk Taxonomy -- "MnemoGuard" risk tiering.

Classifies default action dangers into low/medium/high/critical tiers and generates default policies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

from .policy import Policy, PolicyAction, PolicyRule

RiskLevel = Literal["low", "medium", "high", "critical"]

RISK_ORDER: Tuple[RiskLevel, ...] = ("low", "medium", "high", "critical")


def risk_rank(level: RiskLevel) -> int:
    return RISK_ORDER.index(level)


KIND_BASELINE: Dict[str, RiskLevel] = {
    "llm_call": "low",
    "tool_call": "medium",
    "http_request": "medium",
    "file_write": "high",
    "payment": "high",
}

ESCALATIONS: List[Dict[str, Any]] = [
    {
        "level": "critical",
        "re": re.compile(
            r"\b(wire|payout|transfer|withdraw|send\s+money|move\s+money|sign|contract|delete|drop\s+table|destroy|revoke|deactivate|close\s+account)\b",
            re.I,
        ),
        "why": "irreversible money/contract/destructive action",
    },
    {
        "level": "high",
        "re": re.compile(
            r"\b(upload|ssn|passport|bank\s+statement|id\s+document|kyc|purchase|checkout|pay|refund|disburse|deploy|production)\b",
            re.I,
        ),
        "why": "sensitive upload / spend / production mutation",
    },
    {
        "level": "medium",
        "re": re.compile(
            r"\b(send|email|message|dm|post|submit|fill|book|reserve|schedule)\b",
            re.I,
        ),
        "why": "externally-visible send / form submission",
    },
]


@dataclass
class RiskAssessment:
    level: RiskLevel
    rationale: str


def classify_risk(action: PolicyAction) -> RiskAssessment:
    """Classify an action into a risk tier based on kind and keywords."""
    level = KIND_BASELINE.get(action.kind, "medium")
    rationale = f"{action.kind} baseline"

    # Normalize separators (snake_case / dotted tool ids) to spaces
    haystack = f"{action.target} {action.args_text or ''}"
    haystack = re.sub(r"[_.\-/:]+", " ", haystack)

    for esc in ESCALATIONS:
        if esc["re"].search(haystack):
            if risk_rank(esc["level"]) > risk_rank(level):
                level = esc["level"]
                rationale = esc["why"]
            break  # Order is severe-first

    usd = action.estimated_usd if action.estimated_usd is not None else 0.0
    if action.kind == "payment" or usd > 0:
        amount_level: Optional[RiskLevel] = None
        if usd > 1000:
            amount_level = "critical"
        elif usd > 100:
            amount_level = "high"
        elif usd > 0:
            amount_level = "medium"

        if amount_level and risk_rank(amount_level) > risk_rank(level):
            level = amount_level
            rationale = f"spend ${usd:.2f}"

    return RiskAssessment(level=level, rationale=rationale)


@dataclass
class RiskPolicyOptions:
    approval_threshold_usd: Optional[float] = 50.0
    hard_cap_usd: Optional[float] = 5000.0
    block_targets: Optional[List[str]] = None
    id: Optional[str] = None
    version: Optional[int] = None


def build_risk_policy(opts: Optional[RiskPolicyOptions] = None) -> Policy:
    """Build a ready-to-compile Policy based on risk presets (hard spend limits, thresholds, and target blocks)."""
    opts = opts or RiskPolicyOptions()
    approval_threshold_usd = opts.approval_threshold_usd if opts.approval_threshold_usd is not None else 50.0
    hard_cap_usd = opts.hard_cap_usd if opts.hard_cap_usd is not None else 5000.0
    rules = []

    for target in opts.block_targets or []:
        rules.append(
            PolicyRule(
                id=f"block:{target}",
                description=f"risk preset — {target} requires explicit human action",
                target_in=[target],
                outright_block=True,
            )
        )

    rules.append(
        PolicyRule(
            id="risk:spend-hard-cap",
            description=f"block any single action above ${hard_cap_usd}",
            applies_to=["payment", "tool_call", "http_request"],
            hard_cap_usd=hard_cap_usd,
        )
    )

    rules.append(
        PolicyRule(
            id="risk:spend-approval",
            description=f"require approval above ${approval_threshold_usd}",
            applies_to=["payment", "tool_call", "http_request"],
            approval_threshold_usd=approval_threshold_usd,
        )
    )

    return Policy(
        id=opts.id or "mnemoguard-risk-default",
        version=opts.version or 1,
        rules=rules,
    )
