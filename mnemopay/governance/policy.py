"""
MnemoPay Governance Policy Engine -- sub-second policy enforcement.

Answers "is this agent action allowed right now?" sync, without I/O or LLMs.
Designed for the EU AI Act enforcement timer: evaluation overhead is micro-second level.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Pattern, Set, Union

RateWindow = Literal["second", "minute", "hour"]


@dataclass
class PolicyAction:
    """What the agent wants to do."""
    kind: Literal["tool_call", "llm_call", "http_request", "file_write", "payment"]
    target: str
    estimated_usd: Optional[float] = 0.0
    args_text: Optional[str] = None
    locale: Optional[str] = None
    at: Optional[datetime] = None


@dataclass
class PolicyRule:
    """A single governance policy rule."""
    id: str
    description: Optional[str] = None
    applies_to: Optional[List[str]] = None
    target_in: Optional[List[str]] = None
    target_pattern: Optional[str] = None
    arg_pattern_blocks: Optional[str] = None
    block_locales: Optional[List[str]] = None
    allow_only_locales: Optional[List[str]] = None
    hard_cap_usd: Optional[float] = None
    approval_threshold_usd: Optional[float] = None
    rate_limit: Optional[Dict[str, Any]] = None  # keys: "window" (RateWindow), "max" (int)
    outright_block: Optional[bool] = False


@dataclass
class Policy:
    """A group of versioned rules."""
    id: str
    version: int
    rules: List[PolicyRule]


@dataclass
class CompiledRule:
    raw: PolicyRule
    target_pattern_re: Optional[Pattern] = None
    arg_pattern_re: Optional[Pattern] = None
    target_in_set: Optional[Set[str]] = None
    block_locales_set: Optional[Set[str]] = None
    allow_only_locales_set: Optional[Set[str]] = None
    applies_to_set: Optional[Set[str]] = None


@dataclass
class CompiledPolicy:
    policy: Policy
    rules: List[CompiledRule]


@dataclass
class PolicyVerdict:
    allowed: bool
    matched_rules: Optional[List[str]] = None
    matched_rule: Optional[str] = None
    reason: Optional[str] = None
    needs_approval: Optional[bool] = False
    latency_ns: int = 0


# ─── Compile ────────────────────────────────────────────────────────────────


def compile_policy(policy: Policy) -> CompiledPolicy:
    """Compile raw policy rules into highly efficient compiled states (with regex/set caching)."""
    compiled_rules = []
    for rule in policy.rules:
        compiled_rules.append(
            CompiledRule(
                raw=rule,
                target_pattern_re=re.compile(rule.target_pattern) if rule.target_pattern else None,
                arg_pattern_re=re.compile(rule.arg_pattern_blocks) if rule.arg_pattern_blocks else None,
                target_in_set=set(rule.target_in) if rule.target_in else None,
                block_locales_set=set(rule.block_locales) if rule.block_locales else None,
                allow_only_locales_set=set(rule.allow_only_locales) if rule.allow_only_locales else None,
                applies_to_set=set(rule.applies_to) if rule.applies_to else None,
            )
        )
    return CompiledPolicy(policy=policy, rules=compiled_rules)


# ─── Rate-limit counters ────────────────────────────────────────────────────

_WINDOW_MS: Dict[RateWindow, int] = {
    "second": 1000,
    "minute": 60000,
    "hour": 3600000,
}


class InMemoryRateCounter:
    """In-memory rate limit counter."""

    def __init__(self) -> None:
        self._buckets: Dict[str, List[float]] = {}

    def observe(self, key: str, at_ms: float) -> None:
        self._buckets.setdefault(key, []).append(at_ms)

    def count_within(self, key: str, window_ms: float, at_ms: float) -> int:
        arr = self._buckets.get(key)
        if not arr:
            return 0
        cutoff = at_ms - window_ms
        count = 0
        for ts in arr:
            if ts >= cutoff:
                count += 1
        return count

    def prune(self, before_ms: float) -> None:
        to_delete = []
        for k, arr in self._buckets.items():
            next_arr = [t for t in arr if t >= before_ms]
            if not next_arr:
                to_delete.append(k)
            else:
                self._buckets[k] = next_arr
        for k in to_delete:
            del self._buckets[k]


# ─── Evaluate ───────────────────────────────────────────────────────────────


def evaluate_action(
    compiled: CompiledPolicy,
    action: PolicyAction,
    rate_counter: Optional[InMemoryRateCounter] = None,
    now_fn: Optional[lambda: datetime] = None,
) -> PolicyVerdict:
    """Evaluate an action against a compiled policy, checking limits, rules, and cap filters."""
    start = time.perf_counter_ns()
    now_date = now_fn() if now_fn else (action.at or datetime.now(timezone.utc))
    now_ms = now_date.timestamp() * 1000
    matched: List[str] = []
    approval_rule: Optional[Dict[str, str]] = None

    for rule in compiled.rules:
        if not _rule_applies(rule, action):
            continue
        matched.append(rule.raw.id)

        if rule.raw.outright_block:
            return _block_verdict(rule.raw.id, "outright_block", start)

        if rule.arg_pattern_re and action.args_text and rule.arg_pattern_re.search(action.args_text):
            return _block_verdict(rule.raw.id, "arg_pattern", start)

        if rule.block_locales_set and action.locale and action.locale in rule.block_locales_set:
            return _block_verdict(rule.raw.id, "blocked_locale", start)

        if rule.allow_only_locales_set and (not action.locale or action.locale not in rule.allow_only_locales_set):
            return _block_verdict(rule.raw.id, "locale_not_allowlisted", start)

        est_usd = action.estimated_usd if action.estimated_usd is not None else 0.0
        if rule.raw.hard_cap_usd is not None and est_usd > rule.raw.hard_cap_usd:
            return _block_verdict(rule.raw.id, "hard_cap_usd", start)

        if rule.raw.approval_threshold_usd is not None and est_usd > rule.raw.approval_threshold_usd:
            approval_rule = {"id": rule.raw.id, "reason": "approval_threshold_usd"}

        if rule.raw.rate_limit and rate_counter:
            window = rule.raw.rate_limit.get("window", "second")
            limit_max = rule.raw.rate_limit.get("max", 0)
            key = f"{rule.raw.id}:{action.kind}:{action.target}"
            count = rate_counter.count_within(key, _WINDOW_MS.get(window, 1000), now_ms)
            if count >= limit_max:
                return _block_verdict(rule.raw.id, "rate_limit", start)
            rate_counter.observe(key, now_ms)

    if approval_rule:
        latency_ns = time.perf_counter_ns() - start
        return PolicyVerdict(
            allowed=False,
            needs_approval=True,
            reason=approval_rule["reason"],
            matched_rule=approval_rule["id"],
            latency_ns=latency_ns,
        )

    latency_ns = time.perf_counter_ns() - start
    return PolicyVerdict(allowed=True, matched_rules=matched, latency_ns=latency_ns)


def _rule_applies(rule: CompiledRule, action: PolicyAction) -> bool:
    if rule.applies_to_set and action.kind not in rule.applies_to_set:
        return False
    if rule.target_in_set and action.target not in rule.target_in_set:
        return False
    if rule.target_pattern_re and not rule.target_pattern_re.search(action.target):
        return False
    return True


def _block_verdict(rule_id: string, reason: string, start: int) -> PolicyVerdict:
    latency_ns = time.perf_counter_ns() - start
    return PolicyVerdict(allowed=False, matched_rule=rule_id, reason=reason, latency_ns=latency_ns)
