"""
Tests for BehavioralEngine: prospect theory, discounting, cooling-off, etc.
"""

import time

import pytest

from mnemopay.behavioral import BehavioralEngine
from mnemopay.types import (
    AssetMetrics,
    BehavioralConfig,
    FinancialGoal,
    HerdSeverity,
    RegretEntry,
    RiskLevel,
    TradeEntry,
)

# ── Prospect Theory ───────────────────────────────────────────────────────


class TestProspectTheory:
    def test_gain_positive_value(self):
        engine = BehavioralEngine()
        pv = engine.prospect_value(100)
        assert pv.value > 0
        assert pv.domain == "gain"
        assert pv.loss_aversion == 1.0

    def test_loss_negative_value(self):
        engine = BehavioralEngine()
        pv = engine.prospect_value(-100)
        assert pv.value < 0
        assert pv.domain == "loss"
        assert pv.loss_aversion == 2.25

    def test_loss_aversion_ratio(self):
        """Losses hurt 2.25x more than equivalent gains."""
        engine = BehavioralEngine()
        gain = engine.prospect_value(100)
        loss = engine.prospect_value(-100)
        ratio = abs(loss.value) / gain.value
        assert ratio == pytest.approx(2.25, abs=0.01)

    def test_diminishing_sensitivity_gains(self):
        """Gain function is concave (alpha=0.88 < 1)."""
        engine = BehavioralEngine()
        v100 = engine.prospect_value(100)
        v200 = engine.prospect_value(200)
        assert v200.value < 2 * v100.value

    def test_diminishing_sensitivity_losses(self):
        """Loss function is convex (beta=0.88 < 1)."""
        engine = BehavioralEngine()
        v100 = engine.prospect_value(-100)
        v200 = engine.prospect_value(-200)
        assert abs(v200.value) < 2 * abs(v100.value)

    def test_zero_amount(self):
        engine = BehavioralEngine()
        pv = engine.prospect_value(0)
        assert pv.value == 0
        assert pv.domain == "gain"

    def test_gain_formula_exact(self):
        """v(x) = x^0.88 for gains."""
        engine = BehavioralEngine()
        pv = engine.prospect_value(100)
        expected = 100 ** 0.88
        assert pv.value == pytest.approx(expected, rel=1e-6)

    def test_loss_formula_exact(self):
        """v(x) = -2.25 * (-x)^0.88 for losses."""
        engine = BehavioralEngine()
        pv = engine.prospect_value(-100)
        expected = -2.25 * (100 ** 0.88)
        assert pv.value == pytest.approx(expected, rel=1e-6)

    def test_nan_raises(self):
        engine = BehavioralEngine()
        with pytest.raises(ValueError, match="finite"):
            engine.prospect_value(float("nan"))

    def test_inf_raises(self):
        engine = BehavioralEngine()
        with pytest.raises(ValueError, match="finite"):
            engine.prospect_value(float("inf"))


class TestCompareFraming:
    def test_ratio_approximately_2_25(self):
        engine = BehavioralEngine()
        result = engine.compare_framing(100)
        assert result["ratio"] == pytest.approx(2.25, abs=0.01)

    def test_gain_value_positive(self):
        engine = BehavioralEngine()
        result = engine.compare_framing(50)
        assert result["gain_value"] > 0
        assert result["loss_value"] < 0

    def test_zero_amount_raises(self):
        engine = BehavioralEngine()
        with pytest.raises(ValueError, match="positive"):
            engine.compare_framing(0)

    def test_negative_amount_raises(self):
        engine = BehavioralEngine()
        with pytest.raises(ValueError, match="positive"):
            engine.compare_framing(-50)


# ── Quasi-Hyperbolic Discounting ──────────────────────────────────────────


class TestDiscounting:
    def test_zero_periods_returns_1(self):
        engine = BehavioralEngine()
        assert engine.discount(0) == 1.0

    def test_one_period_beta_times_delta(self):
        """D(1) = beta * delta = 0.70 * 0.96 = 0.672"""
        engine = BehavioralEngine()
        d = engine.discount(1)
        expected = 0.70 * 0.96
        assert d == pytest.approx(expected, rel=1e-6)

    def test_discount_decreases_with_periods(self):
        engine = BehavioralEngine()
        d1 = engine.discount(1)
        d5 = engine.discount(5)
        d10 = engine.discount(10)
        assert d1 > d5 > d10

    def test_present_bias_gap(self):
        """Big gap between D(0)=1 and D(1)=beta*delta (present bias)."""
        engine = BehavioralEngine()
        gap = engine.discount(0) - engine.discount(1)
        assert gap > 0.3  # 1 - 0.672 = 0.328

    def test_negative_periods_raises(self):
        engine = BehavioralEngine()
        with pytest.raises(ValueError, match="non-negative"):
            engine.discount(-1)

    def test_present_value(self):
        engine = BehavioralEngine()
        pv = engine.present_value(1000, 5)
        assert pv["discounted_value"] < 1000
        assert pv["lost_value"] > 0
        assert pv["discount_factor"] < 1

    def test_present_value_zero_periods(self):
        engine = BehavioralEngine()
        pv = engine.present_value(1000, 0)
        assert pv["discounted_value"] == 1000

    def test_present_value_negative_amount_raises(self):
        engine = BehavioralEngine()
        with pytest.raises(ValueError, match="positive"):
            engine.present_value(-100, 5)


# ── Cooling-Off Period ────────────────────────────────────────────────────


class TestCoolingOff:
    def test_below_threshold_no_cooling(self):
        engine = BehavioralEngine()
        result = engine.cooling_off(50, 5000)  # $50 < $100 threshold
        assert result.recommended is False
        assert result.hours == 0

    def test_above_threshold_recommends(self):
        engine = BehavioralEngine()
        result = engine.cooling_off(500, 5000)
        assert result.hours > 0

    def test_large_purchase_long_cooling(self):
        engine = BehavioralEngine()
        small = engine.cooling_off(200, 5000)
        large = engine.cooling_off(2500, 5000)
        assert large.hours > small.hours

    def test_risk_levels(self):
        engine = BehavioralEngine()
        # < 5% income = low
        low = engine.cooling_off(200, 5000)
        assert low.risk_level == RiskLevel.LOW
        # 5-15% = medium
        med = engine.cooling_off(500, 5000)
        assert med.risk_level == RiskLevel.MEDIUM
        # 15-50% = high
        high = engine.cooling_off(1500, 5000)
        assert high.risk_level == RiskLevel.HIGH
        # > 50% = extreme
        extreme = engine.cooling_off(3000, 5000)
        assert extreme.risk_level == RiskLevel.EXTREME

    def test_hours_capped(self):
        engine = BehavioralEngine()
        result = engine.cooling_off(100000, 1000)
        assert result.hours <= 168  # 1 week max

    def test_zero_income_raises(self):
        engine = BehavioralEngine()
        with pytest.raises(ValueError, match="positive"):
            engine.cooling_off(100, 0)

    def test_negative_amount_raises(self):
        engine = BehavioralEngine()
        with pytest.raises(ValueError, match="non-negative"):
            engine.cooling_off(-50, 5000)

    def test_custom_user_beta(self):
        engine = BehavioralEngine()
        default = engine.cooling_off(500, 5000)
        impulsive = engine.cooling_off(500, 5000, user_beta=0.3)
        assert impulsive.hours >= default.hours  # Lower beta = more impulsive


# ── Loss-Framed Nudges ────────────────────────────────────────────────────


class TestLossFrame:
    def test_goal_delay_calculated(self):
        engine = BehavioralEngine()
        goal = FinancialGoal(name="House", target=100000, current=50000, monthly_savings=1000)
        result = engine.loss_frame(500, goal)
        assert result.goal_delay_days == 15  # 500/1000 * 30 = 15 days

    def test_effectiveness_multiplier(self):
        engine = BehavioralEngine()
        goal = FinancialGoal(name="Car", target=30000, current=10000, monthly_savings=500)
        result = engine.loss_frame(100, goal)
        assert result.effectiveness_multiplier == 2.25

    def test_gain_message_provided(self):
        engine = BehavioralEngine()
        goal = FinancialGoal(name="Vacation", target=5000, current=2000, monthly_savings=200)
        result = engine.loss_frame(100, goal)
        assert "Skipping" in result.gain_message

    def test_zero_spend_raises(self):
        engine = BehavioralEngine()
        goal = FinancialGoal(name="Test", target=1000, current=0, monthly_savings=100)
        with pytest.raises(ValueError, match="positive"):
            engine.loss_frame(0, goal)

    def test_invalid_goal_raises(self):
        engine = BehavioralEngine()
        with pytest.raises(ValueError):
            engine.loss_frame(100, FinancialGoal(name="Bad", target=0, current=0, monthly_savings=0))


# ── Commitment Device (SMarT) ─────────────────────────────────────────────


class TestCommitmentDevice:
    def test_rate_increases(self):
        engine = BehavioralEngine()
        result = engine.commitment_device(0.05, 0.03, 4)
        assert result.final_rate > 0.05
        assert len(result.projected_rates) == 5  # initial + 4 cycles

    def test_savings_accumulate(self):
        engine = BehavioralEngine()
        result = engine.commitment_device(0.05, 0.03, 4)
        assert result.projected_savings[-1] > 0

    def test_rate_capped_at_50_percent(self):
        engine = BehavioralEngine()
        result = engine.commitment_device(0.45, 0.10, 10)
        assert result.final_rate <= 0.50

    def test_invalid_savings_rate_raises(self):
        engine = BehavioralEngine()
        with pytest.raises(ValueError, match="0-1"):
            engine.commitment_device(1.5, 0.03, 4)
        with pytest.raises(ValueError, match="0-1"):
            engine.commitment_device(-0.1, 0.03, 4)

    def test_invalid_raise_percent_raises(self):
        engine = BehavioralEngine()
        with pytest.raises(ValueError, match="0-0.5"):
            engine.commitment_device(0.05, 0.6, 4)

    def test_invalid_cycles_raises(self):
        engine = BehavioralEngine()
        with pytest.raises(ValueError, match="1-20"):
            engine.commitment_device(0.05, 0.03, 0)
        with pytest.raises(ValueError, match="1-20"):
            engine.commitment_device(0.05, 0.03, 25)


# ── Regret Prediction ─────────────────────────────────────────────────────


class TestRegretPrediction:
    def test_insufficient_history_returns_50(self):
        engine = BehavioralEngine()
        pred = engine.predict_regret(100, "electronics")
        assert pred.probability == 0.5
        assert pred.confidence == 0.1

    def test_high_regret_history(self):
        engine = BehavioralEngine()
        for i in range(10):
            engine.record_regret(RegretEntry(amount=100, category="electronics", regret_score=8))
        pred = engine.predict_regret(100, "electronics")
        assert pred.probability > 0.5

    def test_low_regret_history(self):
        engine = BehavioralEngine()
        for i in range(10):
            engine.record_regret(RegretEntry(amount=100, category="food", regret_score=2))
        pred = engine.predict_regret(100, "food")
        assert pred.probability < 0.5

    def test_category_specificity(self):
        engine = BehavioralEngine()
        for i in range(5):
            engine.record_regret(RegretEntry(amount=100, category="electronics", regret_score=9))
        for i in range(5):
            engine.record_regret(RegretEntry(amount=100, category="food", regret_score=1))
        pred_elec = engine.predict_regret(100, "electronics")
        pred_food = engine.predict_regret(100, "food")
        assert pred_elec.probability > pred_food.probability

    def test_trigger_cooling_off(self):
        engine = BehavioralEngine()
        for i in range(10):
            engine.record_regret(RegretEntry(amount=200, category="tech", regret_score=8))
        pred = engine.predict_regret(200, "tech")
        assert pred.trigger_cooling_off is True

    def test_invalid_regret_score_raises(self):
        engine = BehavioralEngine()
        with pytest.raises(ValueError, match="0-10"):
            engine.record_regret(RegretEntry(amount=100, category="test", regret_score=11))

    def test_regret_history_capped(self):
        engine = BehavioralEngine()
        for i in range(600):
            engine.record_regret(RegretEntry(amount=10, category="test", regret_score=3))
        assert len(engine.get_regret_history()) <= 500


# ── Overconfidence Brake ──────────────────────────────────────────────────


class TestOverconfidenceBrake:
    def _make_trades(self, count: int, days_ago: float = 0) -> list:
        now = time.time() * 1000
        return [
            TradeEntry(
                timestamp=now - days_ago * 86_400_000,
                amount=100,
                direction="buy",
                asset="BTC",
            )
            for _ in range(count)
        ]

    def test_normal_frequency(self):
        engine = BehavioralEngine()
        trades = self._make_trades(5)
        result = engine.overconfidence_brake(trades, 30)
        assert result.detected is False
        assert result.performance_drag == 0

    def test_overtrading_detected(self):
        engine = BehavioralEngine()
        trades = self._make_trades(50)
        result = engine.overconfidence_brake(trades, 30)
        assert result.detected is True
        assert result.performance_drag > 0

    def test_disposition_effect(self):
        engine = BehavioralEngine()
        now = time.time() * 1000
        trades = [
            TradeEntry(timestamp=now, amount=100, direction="sell", asset="A", realized_pl=50),
            TradeEntry(timestamp=now, amount=100, direction="sell", asset="B", realized_pl=30),
            TradeEntry(timestamp=now, amount=100, direction="sell", asset="C", realized_pl=20),
            TradeEntry(timestamp=now, amount=100, direction="sell", asset="D", realized_pl=-10),
        ]
        result = engine.overconfidence_brake(trades, 30)
        assert result.disposition_effect is True

    def test_empty_trades_ok(self):
        engine = BehavioralEngine()
        result = engine.overconfidence_brake([], 30)
        assert result.detected is False

    def test_invalid_period_raises(self):
        engine = BehavioralEngine()
        with pytest.raises(ValueError, match="positive"):
            engine.overconfidence_brake([], 0)


# ── Expense Reframing ─────────────────────────────────────────────────────


class TestExpenseReframing:
    def test_daily_to_annual(self):
        engine = BehavioralEngine()
        result = engine.reframe_expense(5, "daily")
        assert result.annual == pytest.approx(5 * 365, abs=0.01)

    def test_monthly_to_annual(self):
        engine = BehavioralEngine()
        result = engine.reframe_expense(10, "monthly")
        assert result.annual == pytest.approx(120, abs=0.01)

    def test_weekly_to_annual(self):
        engine = BehavioralEngine()
        result = engine.reframe_expense(20, "weekly")
        assert result.annual == pytest.approx(20 * 52, abs=0.01)

    def test_annual_identity(self):
        engine = BehavioralEngine()
        result = engine.reframe_expense(1000, "annual")
        assert result.annual == 1000

    def test_opportunity_cost_included(self):
        engine = BehavioralEngine()
        result = engine.reframe_expense(100, "monthly")
        assert "7%" in result.opportunity_cost

    def test_invalid_frequency_raises(self):
        engine = BehavioralEngine()
        with pytest.raises(ValueError, match="daily, weekly"):
            engine.reframe_expense(100, "hourly")

    def test_negative_amount_raises(self):
        engine = BehavioralEngine()
        with pytest.raises(ValueError, match="non-negative"):
            engine.reframe_expense(-10, "daily")


# ── Endowed Progress ──────────────────────────────────────────────────────


class TestEndowedProgress:
    def test_zero_progress(self):
        engine = BehavioralEngine()
        goal = FinancialGoal(name="Fund", target=10000, current=0, monthly_savings=500)
        result = engine.endowed_progress(goal)
        assert result.percent_complete == 0
        assert result.expected_completion_rate == 0.19

    def test_some_progress(self):
        engine = BehavioralEngine()
        goal = FinancialGoal(name="Fund", target=10000, current=3000, monthly_savings=500)
        result = engine.endowed_progress(goal)
        assert result.percent_complete == 30
        assert result.expected_completion_rate == 0.34

    def test_past_halfway(self):
        engine = BehavioralEngine()
        goal = FinancialGoal(name="Fund", target=10000, current=6000, monthly_savings=500)
        result = engine.endowed_progress(goal)
        assert result.percent_complete == 60
        assert "halfway" in result.message

    def test_near_complete(self):
        engine = BehavioralEngine()
        goal = FinancialGoal(name="Fund", target=10000, current=9000, monthly_savings=500)
        result = engine.endowed_progress(goal)
        assert result.percent_complete == 90
        assert "finish line" in result.message

    def test_capped_at_100(self):
        engine = BehavioralEngine()
        goal = FinancialGoal(name="Fund", target=10000, current=15000, monthly_savings=500)
        result = engine.endowed_progress(goal)
        assert result.percent_complete == 100

    def test_invalid_target_raises(self):
        engine = BehavioralEngine()
        with pytest.raises(ValueError, match="positive"):
            engine.endowed_progress(FinancialGoal(name="Bad", target=0, current=0, monthly_savings=0))


# ── Anti-Herd Alert ───────────────────────────────────────────────────────


class TestAntiHerdAlert:
    def test_normal_valuation(self):
        engine = BehavioralEngine()
        metrics = AssetMetrics(pe_ratio=16, historical_mean_pe=16, historical_std_pe=5, recent_return_30d=2, volume_ratio=1.0)
        result = engine.anti_herd_alert(metrics)
        assert result.detected is False

    def test_extreme_overvaluation(self):
        engine = BehavioralEngine()
        metrics = AssetMetrics(pe_ratio=44, historical_mean_pe=16, historical_std_pe=5, recent_return_30d=15, volume_ratio=1.0)
        result = engine.anti_herd_alert(metrics)
        assert result.detected is True
        assert result.severity == HerdSeverity.HIGH

    def test_undervaluation(self):
        engine = BehavioralEngine()
        metrics = AssetMetrics(pe_ratio=5, historical_mean_pe=16, historical_std_pe=5, recent_return_30d=-5, volume_ratio=1.0)
        result = engine.anti_herd_alert(metrics)
        assert result.detected is True
        assert result.severity == HerdSeverity.MEDIUM

    def test_volume_spike(self):
        engine = BehavioralEngine()
        metrics = AssetMetrics(pe_ratio=16, historical_mean_pe=16, historical_std_pe=5, recent_return_30d=0, volume_ratio=3.0)
        result = engine.anti_herd_alert(metrics)
        assert result.detected is True

    def test_sigma_calculation(self):
        engine = BehavioralEngine()
        metrics = AssetMetrics(pe_ratio=26, historical_mean_pe=16, historical_std_pe=5, recent_return_30d=0, volume_ratio=1.0)
        result = engine.anti_herd_alert(metrics)
        assert result.valuation_sigma == pytest.approx(2.0, abs=0.01)

    def test_invalid_std_raises(self):
        engine = BehavioralEngine()
        with pytest.raises(ValueError, match="positive"):
            engine.anti_herd_alert(AssetMetrics(pe_ratio=16, historical_mean_pe=16, historical_std_pe=0, recent_return_30d=0, volume_ratio=1.0))


# ── Config Validation ─────────────────────────────────────────────────────


class TestConfigValidation:
    def test_default_config(self):
        engine = BehavioralEngine()
        assert engine.config.lambda_ == 2.25
        assert engine.config.alpha == 0.88
        assert engine.config.beta_pt == 0.88
        assert engine.config.beta_discount == 0.70
        assert engine.config.delta == 0.96

    def test_invalid_lambda_raises(self):
        with pytest.raises(ValueError, match="Lambda"):
            BehavioralEngine(BehavioralConfig(lambda_=0))

    def test_invalid_alpha_raises(self):
        with pytest.raises(ValueError, match="Alpha"):
            BehavioralEngine(BehavioralConfig(alpha=0))

    def test_invalid_beta_pt_raises(self):
        with pytest.raises(ValueError, match="Beta"):
            BehavioralEngine(BehavioralConfig(beta_pt=0))

    def test_invalid_delta_raises(self):
        with pytest.raises(ValueError, match="Delta"):
            BehavioralEngine(BehavioralConfig(delta=0))

    def test_custom_config(self):
        config = BehavioralConfig(lambda_=3.0, alpha=0.90)
        engine = BehavioralEngine(config)
        assert engine.config.lambda_ == 3.0
        assert engine.config.alpha == 0.90


# ── Serialization ─────────────────────────────────────────────────────────


class TestBehavioralSerialization:
    def test_serialize_roundtrip(self):
        engine = BehavioralEngine()
        engine.record_regret(RegretEntry(amount=100, category="test", regret_score=7))
        data = engine.serialize()
        restored = BehavioralEngine.deserialize(data)
        assert len(restored.get_regret_history()) == 1
        assert restored.config.lambda_ == 2.25

    def test_deserialize_invalid_entries_skipped(self):
        data = {
            "config": {},
            "regret_history": [
                {"amount": 100, "category": "ok", "regret_score": 5},
                {"amount": "bad", "category": "bad", "regret_score": 5},
            ],
        }
        engine = BehavioralEngine.deserialize(data)
        assert len(engine.get_regret_history()) == 1
