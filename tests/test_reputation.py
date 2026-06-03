"""
Tests for Agent Reputation Scoring and backwards compatibility/deprecation wrappers.
"""

import json
import warnings
from datetime import datetime, timedelta, timezone

import pytest

from mnemopay import (
    AgentReputationScoring,
    AgentReputationInput,
    AgentReputationConfig,
    AgentReputationTransaction,
    AgentReputationRating,
    TransactionStatus,
    TrustLevel,
)
from mnemopay.fico import AgentFICO
from mnemopay.agent_credit_score import AgentCreditScore


def _make_tx(
    status: TransactionStatus = TransactionStatus.COMPLETED,
    amount: float = 100.0,
    days_ago: float = 0,
    reason: str = "purchase",
    counterparty_id: str = None,
) -> AgentReputationTransaction:
    return AgentReputationTransaction(
        id=f"tx-{id(object())}",
        amount=amount,
        status=status,
        created_at=datetime.now(timezone.utc) - timedelta(days=days_ago),
        reason=reason,
        counterparty_id=counterparty_id,
    )


def _make_input(
    n_completed: int = 10,
    n_refunded: int = 0,
    n_disputed: int = 0,
    n_expired: int = 0,
    fraud_flags: int = 0,
    dispute_count: int = 0,
    disputes_lost: int = 0,
    warnings_count: int = 0,
    age_days: float = 30,
    budget_cap: float = 5000.0,
    memories_count: int = 0,
) -> AgentReputationInput:
    txs = []
    for i in range(n_completed):
        txs.append(_make_tx(TransactionStatus.COMPLETED, 100.0, i, f"purchase {i}", f"cp-{i % 5}"))
    for i in range(n_refunded):
        txs.append(_make_tx(TransactionStatus.REFUNDED, 100.0, i))
    for i in range(n_disputed):
        txs.append(_make_tx(TransactionStatus.DISPUTED, 100.0, i))
    for i in range(n_expired):
        txs.append(_make_tx(TransactionStatus.EXPIRED, 100.0, i))
    return AgentReputationInput(
        transactions=txs,
        created_at=datetime.now(timezone.utc) - timedelta(days=age_days),
        fraud_flags=fraud_flags,
        dispute_count=dispute_count,
        disputes_lost=disputes_lost,
        warnings=warnings_count,
        budget_cap=budget_cap,
        memories_count=memories_count,
    )


class TestReputationConfig:
    def test_default_weights(self):
        rep = AgentReputationScoring()
        assert rep.config.w1 == 0.35
        assert rep.config.w2 == 0.20
        assert rep.config.w3 == 0.15
        assert rep.config.w4 == 0.15
        assert rep.config.w5 == 0.15

    def test_weights_sum_to_1(self):
        rep = AgentReputationScoring()
        total = sum([rep.config.w1, rep.config.w2, rep.config.w3, rep.config.w4, rep.config.w5])
        assert abs(total - 1.0) < 0.001

    def test_invalid_weights_raises(self):
        with pytest.raises(ValueError, match="weights must sum to 1.0"):
            AgentReputationScoring(AgentReputationConfig(w1=0.5, w2=0.5, w3=0.5, w4=0.5, w5=0.5))

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="positive"):
            AgentReputationScoring(AgentReputationConfig(w1=-0.1, w2=0.275, w3=0.275, w4=0.275, w5=0.275))


class TestReputationScoreRange:
    def test_score_within_300_850(self):
        rep = AgentReputationScoring()
        result = rep.compute(_make_input())
        assert 300 <= result.score <= 850

    def test_perfect_agent_scores_high(self):
        rep = AgentReputationScoring()
        result = rep.compute(_make_input(
            n_completed=100, age_days=365, memories_count=100,
        ))
        assert result.score >= 700


class TestDeprecationWarnings:
    def test_agent_fico_deprecation(self):
        import mnemopay.fico
        mnemopay.fico._warned_agent_fico_deprecated = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AgentFICO()
            assert any(
                issubclass(warn.category, DeprecationWarning)
                and "AgentFICO is deprecated" in str(warn.message)
                for warn in w
            )

    def test_agent_credit_score_deprecation(self):
        import mnemopay.agent_credit_score
        mnemopay.agent_credit_score._warned_agent_credit_score_deprecated = False
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            AgentCreditScore()
            assert any(
                issubclass(warn.category, DeprecationWarning)
                and "AgentCreditScore is deprecated" in str(warn.message)
                for warn in w
            )
