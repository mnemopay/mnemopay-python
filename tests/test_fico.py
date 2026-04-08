"""
Tests for Agent FICO credit scoring.
"""

import json
import math
from datetime import datetime, timedelta, timezone

import pytest

from mnemopay.fico import AgentFICO, _clamp, _extract_category
from mnemopay.types import (
    FICOConfig,
    FICOInput,
    FICORating,
    FICOResult,
    FICOTransaction,
    TransactionStatus,
    TrustLevel,
)


def _make_tx(
    status: TransactionStatus = TransactionStatus.COMPLETED,
    amount: float = 100.0,
    days_ago: float = 0,
    reason: str = "purchase",
    counterparty_id: str = None,
) -> FICOTransaction:
    return FICOTransaction(
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
    warnings: int = 0,
    age_days: float = 30,
    budget_cap: float = 5000.0,
    memories_count: int = 0,
) -> FICOInput:
    txs = []
    for i in range(n_completed):
        txs.append(_make_tx(TransactionStatus.COMPLETED, 100.0, i, f"purchase {i}", f"cp-{i % 5}"))
    for i in range(n_refunded):
        txs.append(_make_tx(TransactionStatus.REFUNDED, 100.0, i))
    for i in range(n_disputed):
        txs.append(_make_tx(TransactionStatus.DISPUTED, 100.0, i))
    for i in range(n_expired):
        txs.append(_make_tx(TransactionStatus.EXPIRED, 100.0, i))
    return FICOInput(
        transactions=txs,
        created_at=datetime.now(timezone.utc) - timedelta(days=age_days),
        fraud_flags=fraud_flags,
        dispute_count=dispute_count,
        disputes_lost=disputes_lost,
        warnings=warnings,
        budget_cap=budget_cap,
        memories_count=memories_count,
    )


class TestFICOConfig:
    def test_default_weights(self):
        fico = AgentFICO()
        assert fico.config.w1 == 0.35
        assert fico.config.w2 == 0.20
        assert fico.config.w3 == 0.15
        assert fico.config.w4 == 0.15
        assert fico.config.w5 == 0.15

    def test_weights_sum_to_1(self):
        fico = AgentFICO()
        total = sum([fico.config.w1, fico.config.w2, fico.config.w3, fico.config.w4, fico.config.w5])
        assert abs(total - 1.0) < 0.001

    def test_invalid_weights_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            AgentFICO(FICOConfig(w1=0.5, w2=0.5, w3=0.5, w4=0.5, w5=0.5))

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="positive"):
            AgentFICO(FICOConfig(w1=-0.1, w2=0.275, w3=0.275, w4=0.275, w5=0.275))

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError, match="positive"):
            AgentFICO(FICOConfig(w1=0, w2=0.25, w3=0.25, w4=0.25, w5=0.25))

    def test_custom_config(self):
        config = FICOConfig(w1=0.30, w2=0.20, w3=0.20, w4=0.15, w5=0.15)
        fico = AgentFICO(config)
        assert fico.config.w1 == 0.30


class TestFICOScoreRange:
    def test_score_within_300_850(self):
        fico = AgentFICO()
        result = fico.compute(_make_input())
        assert 300 <= result.score <= 850

    def test_perfect_agent_scores_high(self):
        fico = AgentFICO()
        result = fico.compute(_make_input(
            n_completed=100, age_days=365, memories_count=100,
        ))
        assert result.score >= 700

    def test_bad_agent_scores_low(self):
        fico = AgentFICO()
        result = fico.compute(_make_input(
            n_completed=2, n_disputed=5, n_expired=3,
            fraud_flags=3, dispute_count=5, disputes_lost=3,
            warnings=5, age_days=5,
        ))
        assert result.score < 500

    def test_empty_history_neutral(self):
        fico = AgentFICO()
        result = fico.compute(FICOInput(
            transactions=[], created_at=datetime.now(timezone.utc),
            fraud_flags=0, dispute_count=0, disputes_lost=0, warnings=0,
        ))
        # Score should be moderate (not 300, not 850)
        assert 300 <= result.score <= 600

    def test_score_formula(self):
        """Verify: score = 300 + (composite / 100) * 550"""
        fico = AgentFICO()
        result = fico.compute(_make_input(n_completed=50))
        # Recompute from components
        composite = sum(
            comp.score * comp.weight
            for comp in result.components.values()
        )
        expected = round(max(300, min(850, 300 + (composite / 100) * 550)))
        assert abs(result.score - expected) <= 1  # Allow rounding


class TestFICORatings:
    def test_exceptional_rating(self):
        fico = AgentFICO()
        result = fico.compute(_make_input(
            n_completed=200, age_days=400, memories_count=150,
        ))
        if result.score >= 800:
            assert result.rating == FICORating.EXCEPTIONAL

    def test_poor_rating(self):
        fico = AgentFICO()
        result = fico.compute(_make_input(
            n_completed=1, n_disputed=10, fraud_flags=5,
            dispute_count=10, disputes_lost=8, warnings=10, age_days=1,
        ))
        assert result.rating == FICORating.POOR

    def test_trust_levels(self):
        fico = AgentFICO()
        # Poor score should have minimal trust
        result = fico.compute(_make_input(
            n_completed=1, fraud_flags=5, dispute_count=5,
            disputes_lost=5, warnings=10, age_days=1,
        ))
        assert result.trust_level in (TrustLevel.MINIMAL, TrustLevel.REDUCED)

    def test_hitl_required_for_poor(self):
        fico = AgentFICO()
        result = fico.compute(_make_input(
            n_completed=1, fraud_flags=5, dispute_count=5,
            disputes_lost=5, warnings=10, age_days=1,
        ))
        if result.score < 580:
            assert result.requires_hitl is True

    def test_fee_rate_tiers(self):
        fico = AgentFICO()
        # Perfect agent should get low fee
        perfect = fico.compute(_make_input(
            n_completed=200, age_days=400, memories_count=200,
        ))
        bad = fico.compute(_make_input(
            n_completed=1, fraud_flags=5, dispute_count=5,
            disputes_lost=5, warnings=10, age_days=1,
        ))
        assert perfect.fee_rate <= bad.fee_rate


class TestPaymentHistory:
    def test_all_completed_high_score(self):
        fico = AgentFICO()
        result = fico.compute(_make_input(n_completed=50))
        ph = result.components["payment_history"]
        assert ph.score >= 80

    def test_disputes_lower_score(self):
        fico = AgentFICO()
        clean = fico.compute(_make_input(n_completed=50))
        dirty = fico.compute(_make_input(n_completed=50, n_disputed=5))
        assert dirty.components["payment_history"].score < clean.components["payment_history"].score

    def test_refunds_lower_score_at_high_rate(self):
        fico = AgentFICO()
        clean = fico.compute(_make_input(n_completed=50))
        refund_heavy = fico.compute(_make_input(n_completed=50, n_refunded=20))
        assert refund_heavy.components["payment_history"].score < clean.components["payment_history"].score

    def test_expired_penalty(self):
        fico = AgentFICO()
        clean = fico.compute(_make_input(n_completed=50))
        expired = fico.compute(_make_input(n_completed=50, n_expired=5))
        assert expired.components["payment_history"].score < clean.components["payment_history"].score

    def test_long_term_bonus(self):
        fico = AgentFICO()
        result = fico.compute(_make_input(n_completed=150, age_days=365))
        ph = result.components["payment_history"]
        assert any("+5" in f for f in ph.factors)

    def test_no_history_neutral(self):
        fico = AgentFICO()
        result = fico.compute(FICOInput(
            transactions=[], created_at=datetime.now(timezone.utc),
            fraud_flags=0, dispute_count=0, disputes_lost=0, warnings=0,
        ))
        assert result.components["payment_history"].score == 50.0


class TestCreditUtilization:
    def test_no_spending_perfect(self):
        fico = AgentFICO()
        result = fico.compute(FICOInput(
            transactions=[], created_at=datetime.now(timezone.utc),
            fraud_flags=0, dispute_count=0, disputes_lost=0, warnings=0,
        ))
        assert result.components["credit_utilization"].score == 100.0

    def test_low_utilization_high_score(self):
        fico = AgentFICO()
        txs = [_make_tx(amount=10)]  # $10 of $5000 = 0.2%
        result = fico.compute(FICOInput(
            transactions=txs, created_at=datetime.now(timezone.utc) - timedelta(days=30),
            fraud_flags=0, dispute_count=0, disputes_lost=0, warnings=0,
        ))
        assert result.components["credit_utilization"].score >= 90

    def test_over_budget_low_score(self):
        fico = AgentFICO()
        txs = [_make_tx(amount=6000)]  # 120% of $5000
        result = fico.compute(FICOInput(
            transactions=txs, created_at=datetime.now(timezone.utc) - timedelta(days=30),
            fraud_flags=0, dispute_count=0, disputes_lost=0, warnings=0,
        ))
        assert result.components["credit_utilization"].score < 20


class TestHistoryLength:
    def test_new_account_low_score(self):
        fico = AgentFICO()
        result = fico.compute(_make_input(age_days=1))
        # 1 day old: age_score = (1/365)*100 * 0.7 = 0.19
        # But density can be high with many same-day txs, boosting up to ~30
        assert result.components["history_length"].score < 40

    def test_old_account_high_score(self):
        fico = AgentFICO()
        result = fico.compute(_make_input(n_completed=50, age_days=400))
        assert result.components["history_length"].score > 50

    def test_brand_new_zero(self):
        fico = AgentFICO()
        result = fico.compute(FICOInput(
            transactions=[], created_at=datetime.now(timezone.utc),
            fraud_flags=0, dispute_count=0, disputes_lost=0, warnings=0,
        ))
        hl = result.components["history_length"]
        assert hl.score <= 1  # Essentially zero for brand new


class TestBehaviorDiversity:
    def test_no_transactions_zero(self):
        fico = AgentFICO()
        result = fico.compute(FICOInput(
            transactions=[], created_at=datetime.now(timezone.utc),
            fraud_flags=0, dispute_count=0, disputes_lost=0, warnings=0,
        ))
        assert result.components["behavior_diversity"].score == 0

    def test_diverse_counterparties(self):
        fico = AgentFICO()
        few_cp = _make_input(n_completed=10)
        # Input with more counterparties
        many_cp_txs = [
            _make_tx(counterparty_id=f"cp-{i}", reason=f"service {i}", amount=50 + i * 10)
            for i in range(20)
        ]
        many_cp = FICOInput(
            transactions=many_cp_txs,
            created_at=datetime.now(timezone.utc) - timedelta(days=30),
            fraud_flags=0, dispute_count=0, disputes_lost=0, warnings=0,
        )
        r_few = fico.compute(few_cp)
        r_many = fico.compute(many_cp)
        assert r_many.components["behavior_diversity"].score >= r_few.components["behavior_diversity"].score

    def test_memories_boost_diversity(self):
        fico = AgentFICO()
        no_mem = fico.compute(_make_input(n_completed=10, memories_count=0))
        with_mem = fico.compute(_make_input(n_completed=10, memories_count=100))
        assert with_mem.components["behavior_diversity"].score >= no_mem.components["behavior_diversity"].score


class TestFraudRecord:
    def test_clean_record_100(self):
        fico = AgentFICO()
        result = fico.compute(_make_input())
        assert result.components["fraud_record"].score == 100

    def test_fraud_flags_heavy_penalty(self):
        fico = AgentFICO()
        result = fico.compute(_make_input(fraud_flags=3))
        assert result.components["fraud_record"].score <= 30

    def test_disputes_lost_penalty(self):
        fico = AgentFICO()
        result = fico.compute(_make_input(dispute_count=5, disputes_lost=3))
        fr = result.components["fraud_record"]
        assert fr.score < 100
        assert any("lost" in f for f in fr.factors)

    def test_warnings_penalty(self):
        fico = AgentFICO()
        result = fico.compute(_make_input(warnings=3))
        assert result.components["fraud_record"].score < 100

    def test_multiple_penalties_stack(self):
        fico = AgentFICO()
        result = fico.compute(_make_input(
            fraud_flags=2, dispute_count=5, disputes_lost=3, warnings=5,
        ))
        assert result.components["fraud_record"].score < 30

    def test_penalty_caps(self):
        """Fraud penalties have max caps to prevent score going below 0."""
        fico = AgentFICO()
        result = fico.compute(_make_input(
            fraud_flags=100, dispute_count=100, disputes_lost=100, warnings=100,
        ))
        assert result.components["fraud_record"].score >= 0


class TestStability:
    def test_stable_at_50_transactions(self):
        fico = AgentFICO()
        result = fico.compute(_make_input(n_completed=60))
        assert result.stable is True

    def test_unstable_below_50(self):
        fico = AgentFICO()
        result = fico.compute(_make_input(n_completed=10))
        assert result.stable is False

    def test_confidence_zero_with_no_tx(self):
        fico = AgentFICO()
        result = fico.compute(FICOInput(
            transactions=[], created_at=datetime.now(timezone.utc),
            fraud_flags=0, dispute_count=0, disputes_lost=0, warnings=0,
        ))
        assert result.confidence == 0.0

    def test_confidence_grows_with_tx(self):
        fico = AgentFICO()
        r_few = fico.compute(_make_input(n_completed=5))
        r_many = fico.compute(_make_input(n_completed=50))
        assert r_many.confidence > r_few.confidence

    def test_confidence_capped_at_1(self):
        fico = AgentFICO()
        result = fico.compute(_make_input(n_completed=1000))
        assert result.confidence <= 1.0


class TestDeterminism:
    def test_same_inputs_same_score(self):
        fico = AgentFICO()
        inp = _make_input(n_completed=30, age_days=100)
        r1 = fico.compute(inp)
        r2 = fico.compute(inp)
        assert r1.score == r2.score

    def test_component_weighted_sums(self):
        fico = AgentFICO()
        result = fico.compute(_make_input(n_completed=30))
        for name, comp in result.components.items():
            expected_weighted = round(comp.score * comp.weight * 100) / 100
            assert comp.weighted == pytest.approx(expected_weighted, abs=0.01)


class TestInputValidation:
    def test_negative_fraud_flags_raises(self):
        fico = AgentFICO()
        with pytest.raises(ValueError, match="non-negative"):
            fico.compute(_make_input(fraud_flags=-1))

    def test_disputes_lost_exceeds_count_raises(self):
        fico = AgentFICO()
        with pytest.raises(ValueError, match="cannot exceed"):
            fico.compute(_make_input(dispute_count=3, disputes_lost=5))

    def test_negative_warnings_raises(self):
        fico = AgentFICO()
        with pytest.raises(ValueError, match="non-negative"):
            fico.compute(_make_input(warnings=-1))

    def test_nan_amount_raises(self):
        fico = AgentFICO()
        inp = _make_input()
        inp.transactions[0].amount = float("nan")
        with pytest.raises(ValueError, match="finite"):
            fico.compute(inp)


class TestSerialization:
    def test_serialize_deserialize(self):
        fico = AgentFICO()
        result = fico.compute(_make_input(n_completed=30))
        json_str = AgentFICO.serialize(result)
        data = json.loads(json_str)
        assert data["score"] == result.score

    def test_deserialize_invalid_score(self):
        with pytest.raises(ValueError, match="300-850"):
            AgentFICO.deserialize('{"score": 100}')

    def test_deserialize_valid(self):
        fico = AgentFICO()
        result = fico.compute(_make_input(n_completed=30))
        json_str = AgentFICO.serialize(result)
        loaded = AgentFICO.deserialize(json_str)
        assert loaded["score"] == result.score


class TestHelpers:
    def test_clamp_normal(self):
        assert _clamp(50.0) == 50.0

    def test_clamp_below_0(self):
        assert _clamp(-10.0) == 0.0

    def test_clamp_above_100(self):
        assert _clamp(150.0) == 100.0

    def test_extract_category_known(self):
        assert _extract_category("API compute request") == "api"
        assert _extract_category("Monthly subscription") == "subscription"

    def test_extract_category_unknown(self):
        # Non-alpha chars stripped from first word: "xyz123" -> "xyz"
        assert _extract_category("xyz123") == "xyz"

    def test_extract_category_empty(self):
        assert _extract_category("") == "unknown"
        assert _extract_category(None) == "unknown"
