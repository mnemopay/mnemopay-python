"""
Agent Credit Score -- FICO-style behavioral credit scoring for AI agents.

**Deprecated module name.** This module is preserved for backwards
compatibility under the legacy `mnemopay.fico` import path. New code
should import from `mnemopay.agent_credit_score` (or the top-level
`mnemopay` package), which re-exports the same engine under canonical
names:

  AgentFICO              -> AgentCreditScore
  FICOConfig             -> AgentCreditScoreConfig
  FICOInput              -> AgentCreditScoreInput
  FICOResult             -> AgentCreditScoreResult
  FICOComponent          -> AgentCreditScoreComponent
  FICOTransaction        -> AgentCreditScoreTransaction
  FICORating             -> AgentCreditRating

The legacy names continue to work and produce identical output.
Removal is targeted for v2.0.0.

Trademark notice
----------------
FICO is a registered trademark of Fair Isaac Corporation. MnemoPay's
Agent Credit Score is not affiliated with or endorsed by Fair Isaac
Corporation. The 300-850 range and five-component methodology are used
in the agent-credit-score sense.

Methodology (unchanged):

  1. Payment History    (35%) -- on-time settlements, disputes, late payments
  2. Credit Utilization (20%) -- spend vs budget cap, sweet spot 10-30%
  3. History Length      (15%) -- account age weighted by activity density
  4. Behavior Diversity  (15%) -- counterparties, categories, amount range
  5. Fraud Record        (15%) -- fraud flags, disputes lost, warnings

Score is deterministic given inputs. No randomness, no hidden state.

References:
  - FICO Score Open Access: myfico.com/credit-education
  - Fair Isaac Corporation, "Understanding FICO Scores" (2024)
  - MnemoPay Master Strategy, Part 3.3 (April 2026)
"""

from __future__ import annotations

import json
import math
import warnings
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .types import (
    FICOComponent,
    FICOConfig,
    FICOInput,
    FICORating,
    FICOResult,
    FICOTransaction,
    TransactionStatus,
    TrustLevel,
)

# ── Score Tiers ────────────────────────────────────────────────────────────

_SCORE_TIERS: List[Tuple[int, FICORating, TrustLevel, float, bool]] = [
    (800, FICORating.EXCEPTIONAL, TrustLevel.FULL,     0.010, False),
    (740, FICORating.VERY_GOOD,   TrustLevel.HIGH,     0.013, False),
    (670, FICORating.GOOD,        TrustLevel.STANDARD,  0.015, False),
    (580, FICORating.FAIR,        TrustLevel.REDUCED,   0.019, False),
    (300, FICORating.POOR,        TrustLevel.MINIMAL,   0.025, True),
]

# ── Known transaction categories ──────────────────────────────────────────

_CATEGORIES = [
    "purchase", "subscription", "service", "api", "compute", "storage",
    "transfer", "fee", "refund", "payment", "invoice", "tip", "donation",
    "license", "hosting", "data", "analysis", "research", "test",
]


def _interpret_score(score: int) -> Dict[str, Any]:
    clamped = max(300, min(850, score))
    for min_score, rating, trust, fee_rate, hitl in _SCORE_TIERS:
        if clamped >= min_score:
            return {
                "rating": rating,
                "trust_level": trust,
                "fee_rate": fee_rate,
                "requires_hitl": hitl,
            }
    return {
        "rating": FICORating.POOR,
        "trust_level": TrustLevel.MINIMAL,
        "fee_rate": 0.025,
        "requires_hitl": True,
    }


def _clamp(score: float) -> float:
    return max(0.0, min(100.0, round(score * 100) / 100))


def _extract_category(reason: str) -> str:
    if not reason or not isinstance(reason, str):
        return "unknown"
    lower = reason.lower().strip()
    for cat in _CATEGORIES:
        if cat in lower:
            return cat
    import re
    first_word = re.split(r"\s+", lower)[0] if lower else ""
    first_word = re.sub(r"[^a-z]", "", first_word)
    return first_word or "unknown"


_warned_agent_fico_deprecated = False


def _maybe_warn_agent_fico_deprecated() -> None:
    """Emit a one-process DeprecationWarning the first time AgentFICO is
    instantiated. DeprecationWarning is silent by default under CPython, so
    this is non-intrusive for end users while still showing up under
    `-W default` or pytest's default filters."""
    global _warned_agent_fico_deprecated
    if _warned_agent_fico_deprecated:
        return
    _warned_agent_fico_deprecated = True
    warnings.warn(
        "mnemopay.fico.AgentFICO is deprecated; use "
        "mnemopay.AgentCreditScore (alias of the same engine) instead. "
        "The legacy name will be removed in v2.0.0. FICO is a registered "
        "trademark of Fair Isaac Corporation; MnemoPay is not affiliated "
        "with Fair Isaac Corporation.",
        DeprecationWarning,
        stacklevel=3,
    )


class AgentFICO:
    """FICO-style agent credit scoring engine.

    Deterministic: same inputs, same score.

    .. deprecated::
        Use :class:`mnemopay.AgentCreditScore` instead. This class name will be
        removed in v2.0.0. FICO is a registered trademark of Fair Isaac
        Corporation; MnemoPay is not affiliated with Fair Isaac Corporation.
    """

    def __init__(self, config: Optional[FICOConfig] = None) -> None:
        _maybe_warn_agent_fico_deprecated()
        self.config = config or FICOConfig()

        # Validate weights sum to 1.0
        weight_sum = (
            self.config.w1 + self.config.w2 + self.config.w3
            + self.config.w4 + self.config.w5
        )
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(f"FICO weights must sum to 1.0, got {weight_sum:.4f}")

        # Validate all weights are positive
        if any(w <= 0 for w in [
            self.config.w1, self.config.w2, self.config.w3,
            self.config.w4, self.config.w5,
        ]):
            raise ValueError("All FICO weights must be positive")

    def compute(self, inp: FICOInput) -> FICOResult:
        """Compute Agent FICO score from transaction history and agent metadata."""
        self._validate_input(inp)

        now = datetime.now(timezone.utc).timestamp() * 1000  # ms
        txs = inp.transactions

        # 1. Payment History (35%)
        payment_history = self._compute_payment_history(txs, now)

        # 2. Credit Utilization (20%)
        credit_utilization = self._compute_credit_utilization(
            txs, inp.budget_cap, inp.budget_period_days, now,
        )

        # 3. History Length (15%)
        history_length = self._compute_history_length(inp.created_at, txs, now)

        # 4. Behavior Diversity (15%)
        behavior_diversity = self._compute_behavior_diversity(txs, inp.memories_count)

        # 5. Fraud Record (15%)
        fraud_record = self._compute_fraud_record(
            inp.fraud_flags, inp.dispute_count, inp.disputes_lost, inp.warnings,
        )

        # Weighted composite: 0-100 scale
        composite = (
            payment_history["score"] * self.config.w1
            + credit_utilization["score"] * self.config.w2
            + history_length["score"] * self.config.w3
            + behavior_diversity["score"] * self.config.w4
            + fraud_record["score"] * self.config.w5
        )

        # Map 0-100 to 300-850 range
        raw_score = 300 + (composite / 100) * 550
        score = round(max(300, min(850, raw_score)))

        interpretation = _interpret_score(score)
        total_tx = len(txs)
        stable = total_tx >= self.config.min_stable_transactions

        # Confidence: logarithmic growth
        if total_tx == 0:
            confidence = 0.0
        else:
            confidence = min(
                1.0,
                math.log(1 + total_tx) / math.log(1 + self.config.min_stable_transactions * 2),
            )

        def _make_component(raw: Dict, weight: float) -> FICOComponent:
            return FICOComponent(
                score=raw["score"],
                weight=weight,
                weighted=round(raw["score"] * weight * 100) / 100,
                factors=raw["factors"],
            )

        return FICOResult(
            score=score,
            rating=interpretation["rating"],
            trust_level=interpretation["trust_level"],
            fee_rate=interpretation["fee_rate"],
            requires_hitl=interpretation["requires_hitl"],
            components={
                "payment_history": _make_component(payment_history, self.config.w1),
                "credit_utilization": _make_component(credit_utilization, self.config.w2),
                "history_length": _make_component(history_length, self.config.w3),
                "behavior_diversity": _make_component(behavior_diversity, self.config.w4),
                "fraud_record": _make_component(fraud_record, self.config.w5),
            },
            transaction_count=total_tx,
            stable=stable,
            confidence=round(confidence * 1000) / 1000,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

    # ── Component 1: Payment History (35%) ────────────────────────────────

    def _compute_payment_history(
        self, txs: List[FICOTransaction], now: float,
    ) -> Dict[str, Any]:
        factors: List[str] = []

        if not txs:
            factors.append("No transaction history")
            return {"score": 50.0, "factors": factors}

        completed = [t for t in txs if t.status == TransactionStatus.COMPLETED]
        refunded = [t for t in txs if t.status == TransactionStatus.REFUNDED]
        disputed = [t for t in txs if t.status == TransactionStatus.DISPUTED]
        expired = [t for t in txs if t.status == TransactionStatus.EXPIRED]

        non_pending = [t for t in txs if t.status != TransactionStatus.PENDING]
        success_rate = len(completed) / len(non_pending) if non_pending else 0.0
        score = success_rate * 100

        # Recency weighting
        half_life = self.config.recency_half_life_days
        recent_success_weighted = 0.0
        recent_total_weighted = 0.0

        for tx in non_pending:
            days_ago = (now - tx.created_at.timestamp() * 1000) / 86_400_000
            weight = 2 ** (-days_ago / half_life)
            recent_total_weighted += weight
            if tx.status == TransactionStatus.COMPLETED:
                recent_success_weighted += weight

        if recent_total_weighted > 0:
            recent_rate = recent_success_weighted / recent_total_weighted
            score = recent_rate * 60 + success_rate * 40

        # Penalties
        if disputed:
            penalty = min(30, len(disputed) * 10)
            score -= penalty
            factors.append(f"{len(disputed)} disputed transaction(s) (-{penalty})")

        if refunded and non_pending:
            refund_rate = len(refunded) / len(non_pending)
            if refund_rate > 0.1:
                penalty = min(20, round(refund_rate * 40))
                score -= penalty
                factors.append(f"High refund rate {refund_rate * 100:.0f}% (-{penalty})")

        if expired:
            penalty = min(15, len(expired) * 5)
            score -= penalty
            factors.append(f"{len(expired)} expired transaction(s) (-{penalty})")

        # Bonus for consistent long-term success
        if len(completed) >= 100 and success_rate >= 0.95:
            score = min(100, score + 5)
            factors.append("Excellent long-term track record (+5)")

        if not factors:
            factors.append(
                f"{len(completed)}/{len(non_pending)} successful ({success_rate * 100:.0f}%)"
            )

        return {"score": _clamp(score), "factors": factors}

    # ── Component 2: Credit Utilization (20%) ─────────────────────────────

    def _compute_credit_utilization(
        self,
        txs: List[FICOTransaction],
        budget_cap: float,
        period_days: int,
        now: float,
    ) -> Dict[str, Any]:
        factors: List[str] = []

        if not txs:
            factors.append("No spending history")
            return {"score": 100.0, "factors": factors}

        period_start = now - period_days * 86_400_000
        recent_txs = [
            t for t in txs
            if t.created_at.timestamp() * 1000 >= period_start
            and t.status in (TransactionStatus.COMPLETED, TransactionStatus.PENDING)
        ]
        total_spend = sum(abs(t.amount) for t in recent_txs)
        utilization = total_spend / budget_cap if budget_cap > 0 else 0.0

        if utilization <= 0.10:
            score = 100.0
            factors.append(f"Very low utilization ({utilization * 100:.0f}%)")
        elif utilization <= 0.30:
            score = 90 + (0.30 - utilization) / 0.20 * 10
            factors.append(f"Healthy utilization ({utilization * 100:.0f}%)")
        elif utilization <= 0.50:
            score = 70 + (0.50 - utilization) / 0.20 * 20
            factors.append(f"Moderate utilization ({utilization * 100:.0f}%)")
        elif utilization <= 0.75:
            score = 40 + (0.75 - utilization) / 0.25 * 30
            factors.append(f"High utilization ({utilization * 100:.0f}%)")
        elif utilization <= 1.0:
            score = 10 + (1.0 - utilization) / 0.25 * 30
            factors.append(f"Near-limit utilization ({utilization * 100:.0f}%)")
        else:
            score = max(0, 10 - (utilization - 1.0) * 20)
            factors.append(f"Over budget ({utilization * 100:.0f}%)")

        return {"score": _clamp(score), "factors": factors}

    # ── Component 3: History Length (15%) ──────────────────────────────────

    def _compute_history_length(
        self,
        created_at: datetime,
        txs: List[FICOTransaction],
        now: float,
    ) -> Dict[str, Any]:
        factors: List[str] = []

        age_days = (now - created_at.timestamp() * 1000) / 86_400_000
        if age_days <= 0:
            factors.append("Brand new account")
            return {"score": 0.0, "factors": factors}

        age_score = min(100, (age_days / self.config.full_history_days) * 100)

        # Activity density
        active_days: set[str] = set()
        for tx in txs:
            day = tx.created_at.strftime("%Y-%m-%d")
            active_days.add(day)
        density = min(1.0, len(active_days) / max(1, age_days))

        # Blend: 70% age + 30% density
        score = age_score * 0.7 + (density * 100) * 0.3

        if age_days < 7:
            factors.append(f"Account is {int(age_days)} day(s) old")
        elif age_days < 30:
            factors.append(f"Account is {int(age_days / 7)} week(s) old")
        else:
            factors.append(f"Account is {int(age_days / 30)} month(s) old")

        factors.append(f"Activity density: {density * 100:.0f}% ({len(active_days)} active days)")

        return {"score": _clamp(score), "factors": factors}

    # ── Component 4: Behavior Diversity (15%) ─────────────────────────────

    def _compute_behavior_diversity(
        self, txs: List[FICOTransaction], memories_count: int,
    ) -> Dict[str, Any]:
        factors: List[str] = []

        if not txs:
            factors.append("No transaction data")
            return {"score": 0.0, "factors": factors}

        # A: Unique counterparties
        counterparties: set[str] = set()
        for tx in txs:
            if tx.counterparty_id:
                counterparties.add(tx.counterparty_id)
        cp_score = min(1.0, len(counterparties) / self.config.max_expected_counterparties)

        # B: Unique categories
        categories: set[str] = set()
        for tx in txs:
            cat = _extract_category(tx.reason)
            if cat:
                categories.add(cat)
        cat_score = min(1.0, len(categories) / self.config.max_expected_categories)

        # C: Amount range diversity (coefficient of variation)
        amounts = [t.amount for t in txs if t.amount > 0]
        amount_diversity = 0.0
        if len(amounts) >= 2:
            mean = sum(amounts) / len(amounts)
            if mean > 0:
                variance = sum((a - mean) ** 2 for a in amounts) / len(amounts)
                cv = math.sqrt(variance) / mean
                if cv <= 1.5:
                    amount_diversity = min(1.0, cv / 0.8)
                else:
                    amount_diversity = max(0.0, 1 - (cv - 1.5) / 2)

        # D: Memory richness
        mem_score = min(1.0, memories_count / 100)

        score = cp_score * 30 + cat_score * 25 + amount_diversity * 25 + mem_score * 20

        factors.append(f"{len(counterparties)} unique counterparties")
        factors.append(f"{len(categories)} transaction categories")
        if len(amounts) >= 2:
            factors.append(f"Amount range: ${min(amounts):.2f}-${max(amounts):.2f}")
        factors.append(f"{memories_count} memories stored")

        return {"score": _clamp(score), "factors": factors}

    # ── Component 5: Fraud Record (15%) ───────────────────────────────────

    def _compute_fraud_record(
        self,
        fraud_flags: int,
        dispute_count: int,
        disputes_lost: int,
        warnings: int,
    ) -> Dict[str, Any]:
        factors: List[str] = []
        score = 100.0

        if fraud_flags > 0:
            penalty = min(80, fraud_flags * 25)
            score -= penalty
            factors.append(f"{fraud_flags} confirmed fraud flag(s) (-{penalty})")

        if disputes_lost > 0:
            penalty = min(40, disputes_lost * 15)
            score -= penalty
            factors.append(f"{disputes_lost} dispute(s) lost (-{penalty})")

        if dispute_count > disputes_lost:
            won_or_pending = dispute_count - disputes_lost
            penalty = min(10, won_or_pending * 3)
            score -= penalty
            factors.append(f"{won_or_pending} other dispute(s) (-{penalty})")

        if warnings > 0:
            penalty = min(15, warnings * 5)
            score -= penalty
            factors.append(f"{warnings} warning(s) (-{penalty})")

        if not factors:
            factors.append("Clean record")

        return {"score": _clamp(score), "factors": factors}

    # ── Input Validation ──────────────────────────────────────────────────

    def _validate_input(self, inp: FICOInput) -> None:
        if not isinstance(inp, FICOInput):
            raise ValueError("FICOInput is required")
        if not isinstance(inp.transactions, list):
            raise ValueError("FICOInput.transactions must be a list")
        if not isinstance(inp.created_at, datetime):
            raise ValueError("FICOInput.created_at must be a datetime")
        if not isinstance(inp.fraud_flags, (int, float)) or inp.fraud_flags < 0:
            raise ValueError("FICOInput.fraud_flags must be a non-negative number")
        if not isinstance(inp.dispute_count, (int, float)) or inp.dispute_count < 0:
            raise ValueError("FICOInput.dispute_count must be a non-negative number")
        if not isinstance(inp.disputes_lost, (int, float)) or inp.disputes_lost < 0:
            raise ValueError("FICOInput.disputes_lost must be a non-negative number")
        if inp.disputes_lost > inp.dispute_count:
            raise ValueError("FICOInput.disputes_lost cannot exceed dispute_count")
        if not isinstance(inp.warnings, (int, float)) or inp.warnings < 0:
            raise ValueError("FICOInput.warnings must be a non-negative number")
        if inp.budget_cap is not None and (
            not isinstance(inp.budget_cap, (int, float)) or inp.budget_cap <= 0
        ):
            raise ValueError("FICOInput.budget_cap must be a positive number")

        for tx in inp.transactions:
            if not isinstance(tx.amount, (int, float)) or not math.isfinite(tx.amount):
                raise ValueError(f"Transaction {tx.id}: amount must be a finite number")
            if not isinstance(tx.created_at, datetime):
                raise ValueError(f"Transaction {tx.id}: created_at must be a datetime")

    # ── Serialization ─────────────────────────────────────────────────────

    @staticmethod
    def serialize(result: FICOResult) -> str:
        """Serialize a FICOResult to JSON."""
        def _to_dict(obj: Any) -> Any:
            if hasattr(obj, "__dataclass_fields__"):
                d = {}
                for k, v in obj.__dict__.items():
                    d[k] = _to_dict(v)
                return d
            if isinstance(obj, dict):
                return {k: _to_dict(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_dict(i) for i in obj]
            if isinstance(obj, (FICORating, TrustLevel)):
                return obj.value
            return obj
        return json.dumps(_to_dict(result))

    @staticmethod
    def deserialize(json_str: str) -> FICOResult:
        """Deserialize from JSON with validation."""
        data = json.loads(json_str)
        if not isinstance(data.get("score"), (int, float)) or data["score"] < 300 or data["score"] > 850:
            raise ValueError("Invalid FICO score: must be 300-850")

        # Re-clamp component scores on load
        for key in [
            "payment_history", "credit_utilization", "history_length",
            "behavior_diversity", "fraud_record",
        ]:
            if data.get("components", {}).get(key):
                data["components"][key]["score"] = _clamp(data["components"][key]["score"])

        return data  # type: ignore
