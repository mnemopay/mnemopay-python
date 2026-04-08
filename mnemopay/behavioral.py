"""
Behavioral Finance Engine -- Psychology-Backed Financial Intelligence

Implements Nobel Prize-winning behavioral economics research:
  1. Prospect Theory value function (Kahneman & Tversky, 1979/1992)
  2. Quasi-hyperbolic discounting (Laibson, 1997)
  3. Dynamic cooling-off periods (MnemoPay original)
  4. Loss-framed nudges (Tversky & Kahneman, 1981)
  5. Commitment devices / Save More Tomorrow (Thaler & Benartzi, 2004)
  6. Regret memory + prediction (Zeelenberg & Pieters, 2007)
  7. Overconfidence brake (Barber & Odean, 2000)
  8. Intelligent expense reframing (Thaler, 1985 mental accounting)
  9. Endowed progress effect (Nunes & Dreze, 2006)
  10. Anti-herd alerting (Shiller, 2000)

All parameters sourced from peer-reviewed publications.
Zero external dependencies.
"""

from __future__ import annotations

import math
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .types import (
    AssetMetrics,
    BehavioralConfig,
    CommitmentResult,
    CoolingOffResult,
    EndowedProgress,
    FinancialGoal,
    HerdAlert,
    HerdSeverity,
    LossFrame,
    OverconfidenceResult,
    ProspectValue,
    ReframedExpense,
    RegretEntry,
    RegretPrediction,
    RiskLevel,
    TradeEntry,
)


class BehavioralEngine:
    """Psychology-backed financial intelligence engine."""

    MAX_REGRET_HISTORY = 500

    def __init__(self, config: Optional[BehavioralConfig] = None) -> None:
        self.config = config or BehavioralConfig()
        self._regret_history: List[RegretEntry] = []
        self._validate_config()

    # ── 1. Prospect Theory Value Function ─────────────────────────────────
    # v(x) = x^alpha              for gains (x >= 0)
    # v(x) = -lambda * (-x)^beta  for losses (x < 0)

    def prospect_value(self, amount: float) -> ProspectValue:
        """Compute the psychologically perceived value of a monetary amount."""
        if not isinstance(amount, (int, float)) or not math.isfinite(amount):
            raise ValueError("Amount must be a finite number")

        if amount >= 0:
            value = amount ** self.config.alpha
            return ProspectValue(
                value=value,
                amount=amount,
                domain="gain",
                loss_aversion=1.0,
                explanation=(
                    f"A ${amount:.2f} gain feels like {value:.2f} units of satisfaction. "
                    f"Diminishing returns: doubling the gain doesn't double the joy."
                ),
            )
        else:
            abs_amount = abs(amount)
            value = -self.config.lambda_ * (abs_amount ** self.config.beta_pt)
            return ProspectValue(
                value=value,
                amount=amount,
                domain="loss",
                loss_aversion=self.config.lambda_,
                explanation=(
                    f"A ${abs_amount:.2f} loss feels like {abs(value):.2f} units of pain "
                    f"-- {self.config.lambda_}x worse than an equivalent gain."
                ),
            )

    def compare_framing(self, amount: float) -> Dict[str, Any]:
        """Compare gain vs loss framing of the same amount."""
        if amount <= 0:
            raise ValueError("Amount must be positive for framing comparison")
        gain = self.prospect_value(amount)
        loss = self.prospect_value(-amount)
        ratio = abs(loss.value) / gain.value
        return {
            "gain_value": gain.value,
            "loss_value": loss.value,
            "ratio": round(ratio * 100) / 100,
            "insight": (
                f"Losing ${amount:.2f} hurts {ratio:.1f}x more than gaining it feels good. "
                f"Frame savings interventions as loss prevention for {ratio:.1f}x effectiveness."
            ),
        }

    # ── 2. Quasi-Hyperbolic Discounting ───────────────────────────────────
    # D(0) = 1
    # D(t) = beta * delta^t  for t >= 1

    def discount(self, periods: float) -> float:
        """Compute quasi-hyperbolic discount factor for given periods."""
        if not isinstance(periods, (int, float)) or not math.isfinite(periods) or periods < 0:
            raise ValueError("Periods must be a non-negative number")
        if periods == 0:
            return 1.0
        return self.config.beta_discount * (self.config.delta ** periods)

    def present_value(self, future_amount: float, periods: float) -> Dict[str, Any]:
        """Calculate the present-biased value of a future amount."""
        if future_amount <= 0:
            raise ValueError("Future amount must be positive")
        factor = self.discount(periods)
        discounted = future_amount * factor
        return {
            "discounted_value": round(discounted * 100) / 100,
            "discount_factor": round(factor * 10000) / 10000,
            "lost_value": round((future_amount - discounted) * 100) / 100,
            "explanation": (
                f"${future_amount:.2f} in {periods} period(s) feels worth only "
                f"${discounted:.2f} now ({factor * 100:.1f}% of face value). "
                f"Present bias loses ${future_amount - discounted:.2f} of perceived value."
            ),
        }

    # ── 3. Dynamic Cooling-Off Period ─────────────────────────────────────

    def cooling_off(
        self,
        amount: float,
        monthly_income: float,
        user_beta: Optional[float] = None,
    ) -> CoolingOffResult:
        """Calculate recommended cooling-off period for a purchase."""
        if not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount < 0:
            raise ValueError("Amount must be a non-negative finite number")
        if not isinstance(monthly_income, (int, float)) or not math.isfinite(monthly_income) or monthly_income <= 0:
            raise ValueError("Monthly income must be a positive number")

        if amount < self.config.cooling_threshold:
            return CoolingOffResult(
                recommended=False,
                hours=0.0,
                reason=(
                    f"Amount (${amount:.2f}) below cooling threshold "
                    f"(${self.config.cooling_threshold})"
                ),
                risk_level=RiskLevel.LOW,
                regret_probability=0.0,
            )

        beta = max(0.1, min(1.0, user_beta if user_beta is not None else self.config.beta_discount))
        regret_ratio = self._compute_regret_ratio()
        income_ratio = amount / monthly_income

        # Core formula
        hours = self.config.base_cooling_hours * income_ratio * (1 / beta) * max(0.1, regret_ratio)
        hours = max(0.5, min(168, hours))  # 30 min to 1 week

        # Risk level
        if income_ratio < 0.05:
            risk_level = RiskLevel.LOW
        elif income_ratio < 0.15:
            risk_level = RiskLevel.MEDIUM
        elif income_ratio < 0.50:
            risk_level = RiskLevel.HIGH
        else:
            risk_level = RiskLevel.EXTREME

        regret_probability = min(0.95, regret_ratio * 1.2)

        if hours > 0.5:
            reason = (
                f"Purchase is {income_ratio * 100:.0f}% of monthly income. "
                f"Historical regret rate: {regret_ratio * 100:.0f}%. "
                f"Wait {hours:.1f} hours."
            )
        else:
            reason = "Low-risk purchase relative to income"

        return CoolingOffResult(
            recommended=hours >= 1,
            hours=round(hours * 10) / 10,
            reason=reason,
            risk_level=risk_level,
            regret_probability=round(regret_probability * 100) / 100,
        )

    # ── 4. Loss-Framed Nudges ─────────────────────────────────────────────

    def loss_frame(self, spend_amount: float, goal: FinancialGoal) -> LossFrame:
        """Frame spending as goal delay."""
        if spend_amount <= 0:
            raise ValueError("Spend amount must be positive")
        if not goal or goal.target <= 0 or goal.monthly_savings <= 0:
            raise ValueError("Goal must have positive target and monthly savings")

        remaining = goal.target - goal.current
        delay_months = spend_amount / goal.monthly_savings if goal.monthly_savings > 0 else 0
        delay_days = round(delay_months * 30)

        if delay_days > 0:
            message = (
                f'This ${spend_amount:.2f} purchase delays your "{goal.name}" goal by '
                f"{delay_days} day(s). You'll reach ${goal.target:,.0f} "
                f"{delay_days} day(s) later."
            )
        else:
            message = f'This purchase has minimal impact on your "{goal.name}" goal.'

        gain_message = (
            f'Skipping this purchase saves ${spend_amount:.2f} toward your "{goal.name}" goal.'
        )

        return LossFrame(
            message=message,
            gain_message=gain_message,
            effectiveness_multiplier=self.config.lambda_,
            goal_delay_days=delay_days,
        )

    # ── 5. Commitment Devices / Save More Tomorrow ────────────────────────

    def commitment_device(
        self,
        current_savings_rate: float,
        raise_percent: float,
        cycles: int,
    ) -> CommitmentResult:
        """Project savings growth using SMarT (Save More Tomorrow) model."""
        if current_savings_rate < 0 or current_savings_rate > 1:
            raise ValueError("Savings rate must be 0-1")
        if raise_percent <= 0 or raise_percent > 0.5:
            raise ValueError("Raise percent must be 0-0.5")
        if cycles < 1 or cycles > 20 or not isinstance(cycles, int):
            raise ValueError("Cycles must be integer 1-20")

        allocation_rate = 0.5
        rates = [current_savings_rate]
        savings = [0.0]

        rate = current_savings_rate
        cumulative_savings = 0.0
        base_salary = 100_000

        for _ in range(cycles):
            raise_amount = base_salary * raise_percent
            additional_savings = raise_amount * allocation_rate
            rate = min(0.50, rate + (additional_savings / base_salary))
            cumulative_savings += base_salary * rate
            rates.append(round(rate * 10000) / 10000)
            savings.append(round(cumulative_savings))

        return CommitmentResult(
            projected_rates=rates,
            projected_savings=savings,
            final_rate=rates[-1],
            explanation=(
                f"SMarT projection: savings rate grows from "
                f"{current_savings_rate * 100:.1f}% to {rate * 100:.1f}% over "
                f"{cycles} raise cycle(s). Thaler & Benartzi (2004) observed "
                f"3.5% -> 13.6% in 4 cycles. Key insight: people don't feel the "
                f"loss because it comes from future raises, not current income."
            ),
        )

    # ── 6. Regret Memory & Prediction ─────────────────────────────────────

    def record_regret(self, entry: RegretEntry) -> None:
        """Record a regret entry for future prediction."""
        if not isinstance(entry.amount, (int, float)) or not math.isfinite(entry.amount):
            raise ValueError("Regret entry requires a valid amount")
        if not isinstance(entry.regret_score, (int, float)) or entry.regret_score < 0 or entry.regret_score > 10:
            raise ValueError("Regret score must be 0-10")

        self._regret_history.append(RegretEntry(
            amount=entry.amount,
            category=(entry.category or "unknown").lower(),
            regret_score=entry.regret_score,
            timestamp=entry.timestamp or datetime.now(timezone.utc).isoformat(),
        ))

        if len(self._regret_history) > self.MAX_REGRET_HISTORY:
            self._regret_history = self._regret_history[-self.MAX_REGRET_HISTORY:]

    def predict_regret(self, amount: float, category: str) -> RegretPrediction:
        """Predict regret probability for a potential purchase."""
        if not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount < 0:
            raise ValueError("Amount must be a non-negative finite number")

        cat = (category or "unknown").lower()

        if len(self._regret_history) < 3:
            return RegretPrediction(
                probability=0.5,
                confidence=0.1,
                historical_regret_rate=0.0,
                category_regret_rate=0.0,
                recommendation="Not enough purchase history to predict regret. Consider waiting.",
                trigger_cooling_off=amount >= self.config.cooling_threshold,
            )

        # Overall regret rate (regretScore > 5 = regretted)
        regretted = sum(1 for r in self._regret_history if r.regret_score > 5)
        historical_rate = regretted / len(self._regret_history)

        # Category-specific rate
        cat_entries = [r for r in self._regret_history if r.category == cat]
        cat_regretted = sum(1 for r in cat_entries if r.regret_score > 5)
        category_rate = cat_regretted / len(cat_entries) if len(cat_entries) >= 2 else historical_rate

        # Amount-weighted
        similar_amount = [
            r for r in self._regret_history
            if r.amount >= amount * 0.5 and r.amount <= amount * 2
        ]
        amount_regretted = sum(1 for r in similar_amount if r.regret_score > 5)
        amount_rate = amount_regretted / len(similar_amount) if len(similar_amount) >= 2 else historical_rate

        # Blend: 40% category, 30% amount-similar, 30% overall
        probability = min(0.95, category_rate * 0.4 + amount_rate * 0.3 + historical_rate * 0.3)
        confidence = min(0.9, len(self._regret_history) / 50)

        if probability >= 0.7:
            recommendation = (
                f"High regret risk ({probability * 100:.0f}%). You've regretted "
                f"{category_rate * 100:.0f}% of similar \"{cat}\" purchases. "
                f"Strongly consider waiting."
            )
        elif probability >= 0.4:
            recommendation = (
                f"Moderate regret risk ({probability * 100:.0f}%). Consider whether "
                f"this purchase aligns with your goals."
            )
        else:
            recommendation = (
                f"Low regret risk ({probability * 100:.0f}%). This type of purchase "
                f"has been satisfying historically."
            )

        return RegretPrediction(
            probability=round(probability * 100) / 100,
            confidence=round(confidence * 100) / 100,
            historical_regret_rate=round(historical_rate * 100) / 100,
            category_regret_rate=round(category_rate * 100) / 100,
            recommendation=recommendation,
            trigger_cooling_off=probability >= 0.5 and amount >= self.config.cooling_threshold,
        )

    # ── 7. Overconfidence Brake ───────────────────────────────────────────

    def overconfidence_brake(
        self, trades: List[TradeEntry], period_days: int = 30,
    ) -> OverconfidenceResult:
        """Detect overconfidence and disposition effect in trading."""
        if not isinstance(trades, list):
            raise ValueError("Trades must be a list")
        if period_days <= 0:
            raise ValueError("Period must be positive")

        now = time.time() * 1000  # ms
        period_ms = period_days * 86_400_000
        recent_trades = [t for t in trades if (now - t.timestamp) < period_ms]

        frequency = len(recent_trades)
        optimal = self.config.max_trades_per_month * (period_days / 30)

        excess_trades = max(0, frequency - optimal)
        performance_drag = excess_trades * (self.config.excess_trade_cost_bps / 100)

        # Disposition effect
        sells = [
            t for t in recent_trades
            if t.direction == "sell" and t.realized_pl is not None
        ]
        winner_sells = sum(1 for t in sells if (t.realized_pl or 0) > 0)
        loser_sells = sum(1 for t in sells if (t.realized_pl or 0) < 0)
        disposition_effect = len(sells) >= 3 and winner_sells > loser_sells * 1.3

        detected = frequency > optimal or disposition_effect

        if frequency > optimal * 2:
            recommendation = (
                f"Trading {frequency} times in {period_days} days is "
                f"{frequency / optimal:.1f}x the optimal rate. Estimated annual "
                f"performance drag: {performance_drag:.1f}%. Consider a rules-based approach."
            )
        elif disposition_effect:
            recommendation = (
                f"Disposition effect detected: selling winners ({winner_sells}) "
                f"more than losers ({loser_sells}). This pattern costs ~4% annually "
                f"(Odean 1998). Let winners run, cut losers."
            )
        elif frequency > optimal:
            recommendation = "Slightly above optimal trading frequency. Monitor for escalation."
        else:
            recommendation = "Trading frequency is within optimal range."

        return OverconfidenceResult(
            detected=detected,
            frequency=frequency,
            optimal_frequency=round(optimal),
            performance_drag=round(performance_drag * 100) / 100,
            disposition_effect=disposition_effect,
            recommendation=recommendation,
        )

    # ── 8. Intelligent Expense Reframing ──────────────────────────────────

    def reframe_expense(
        self, amount: float, frequency: str,
    ) -> ReframedExpense:
        """Convert expenses between timeframes to reveal true cost."""
        if not isinstance(amount, (int, float)) or not math.isfinite(amount) or amount < 0:
            raise ValueError("Amount must be a non-negative finite number")

        if frequency == "daily":
            annual = amount * 365
        elif frequency == "weekly":
            annual = amount * 52
        elif frequency == "monthly":
            annual = amount * 12
        elif frequency == "annual":
            annual = amount
        else:
            raise ValueError("Frequency must be daily, weekly, monthly, or annual")

        daily = annual / 365
        weekly = annual / 52
        monthly = annual / 12

        if frequency in ("monthly", "weekly"):
            impact_frame = f"That's ${annual:.2f}/year. Over 10 years: ${annual * 10:.0f}."
        elif frequency == "daily":
            impact_frame = (
                f"That ${amount:.2f}/day habit costs ${annual:.0f}/year "
                f"-- ${annual * 10:.0f} over a decade."
            )
        else:
            impact_frame = f"That's ${daily:.2f}/day or ${monthly:.2f}/month."

        invested_10yr = annual * ((1.07 ** 10 - 1) / 0.07)
        opportunity_cost = (
            f"Invested instead at 7% annual return: "
            f"${round(invested_10yr):,} in 10 years."
        )

        return ReframedExpense(
            original={"amount": amount, "frequency": frequency},
            daily=round(daily * 100) / 100,
            weekly=round(weekly * 100) / 100,
            monthly=round(monthly * 100) / 100,
            annual=round(annual * 100) / 100,
            impact_frame=impact_frame,
            opportunity_cost=opportunity_cost,
        )

    # ── 9. Endowed Progress Effect ────────────────────────────────────────

    def endowed_progress(self, goal: FinancialGoal) -> EndowedProgress:
        """Frame progress using the endowed progress effect."""
        if not goal or goal.target <= 0:
            raise ValueError("Goal must have positive target")

        percent = min(100, max(0, (goal.current / goal.target) * 100))

        if percent >= 80:
            message = (
                f"You're {percent:.0f}% there! Only "
                f"${goal.target - goal.current:.0f} to go for \"{goal.name}\". "
                f"The finish line is right there."
            )
        elif percent >= 50:
            message = (
                f"You're past the halfway mark on \"{goal.name}\" -- "
                f"{percent:.0f}% complete! ${goal.current:.0f} saved so far."
            )
        elif percent >= 20:
            message = (
                f"Great start on \"{goal.name}\" -- {percent:.0f}% done! "
                f"You've already saved ${goal.current:.0f}."
            )
        elif percent > 0:
            message = (
                f"You've started \"{goal.name}\" -- every dollar counts. "
                f"{percent:.1f}% complete (${goal.current:.0f} saved)."
            )
        else:
            message = (
                f"Ready to start \"{goal.name}\"? Setting aside even $1 today "
                f"puts you ahead of 80% of people who never start."
            )

        expected_completion_rate = 0.34 if percent > 0 else 0.19

        return EndowedProgress(
            percent_complete=round(percent * 100) / 100,
            message=message,
            expected_completion_rate=expected_completion_rate,
        )

    # ── 10. Anti-Herd Alert ───────────────────────────────────────────────

    def anti_herd_alert(self, metrics: AssetMetrics) -> HerdAlert:
        """Detect herd behavior using valuation analysis."""
        if not metrics or not isinstance(metrics.pe_ratio, (int, float)):
            raise ValueError("Valid asset metrics required")
        if metrics.historical_std_pe <= 0:
            raise ValueError("Historical std dev must be positive")

        sigma = (metrics.pe_ratio - metrics.historical_mean_pe) / metrics.historical_std_pe

        detected = False
        severity = HerdSeverity.LOW
        message = ""
        recommendation = ""

        if sigma > 2 and metrics.recent_return_30d > 10:
            detected = True
            severity = HerdSeverity.HIGH
            message = (
                f"Extreme overvaluation: P/E {metrics.pe_ratio:.1f} is "
                f"{sigma:.1f}\u03c3 above mean ({metrics.historical_mean_pe:.1f}). "
                f"Recent 30-day return of {metrics.recent_return_30d:.1f}% "
                f"suggests momentum chasing."
            )
            recommendation = (
                "Contrarian signal: reduce exposure. Historical mean reversion "
                "is strong above 2\u03c3."
            )
        elif sigma > 1.5:
            detected = True
            severity = HerdSeverity.MEDIUM
            message = (
                f"Elevated valuation: P/E {metrics.pe_ratio:.1f} is "
                f"{sigma:.1f}\u03c3 above mean. Watch for reversal."
            )
            recommendation = "Consider trimming position or tightening stops."
        elif sigma < -1.5:
            detected = True
            severity = HerdSeverity.MEDIUM
            message = (
                f"Potential undervaluation: P/E {metrics.pe_ratio:.1f} is "
                f"{abs(sigma):.1f}\u03c3 below mean. Market may be irrationally pessimistic."
            )
            recommendation = (
                "Contrarian opportunity: evaluate fundamentals. Beaten-down "
                "assets often recover (De Bondt & Thaler 1985: 25% outperformance "
                "over 3 years)."
            )
        else:
            message = f"Valuation within normal range ({sigma:.1f}\u03c3 from mean)."
            recommendation = "No herd-driven distortion detected."

        if metrics.volume_ratio > 2.5:
            detected = True
            if severity == HerdSeverity.LOW:
                severity = HerdSeverity.MEDIUM
            message += f" Volume is {metrics.volume_ratio:.1f}x normal -- unusual activity."

        return HerdAlert(
            detected=detected,
            severity=severity,
            valuation_sigma=round(sigma * 100) / 100,
            message=message,
            recommendation=recommendation,
        )

    # ── Helpers ───────────────────────────────────────────────────────────

    def _compute_regret_ratio(self) -> float:
        if not self._regret_history:
            return 0.5
        regretted = sum(1 for r in self._regret_history if r.regret_score > 5)
        return regretted / len(self._regret_history)

    def _validate_config(self) -> None:
        c = self.config
        if c.lambda_ <= 0 or c.lambda_ > 10:
            raise ValueError("Lambda must be in (0, 10]")
        if c.alpha <= 0 or c.alpha > 1:
            raise ValueError("Alpha must be in (0, 1]")
        if c.beta_pt <= 0 or c.beta_pt > 1:
            raise ValueError("Beta (PT) must be in (0, 1]")
        if c.beta_discount <= 0 or c.beta_discount > 1:
            raise ValueError("Beta (discount) must be in (0, 1]")
        if c.delta <= 0 or c.delta > 1:
            raise ValueError("Delta must be in (0, 1]")
        if c.base_cooling_hours <= 0:
            raise ValueError("Base cooling hours must be positive")
        if c.cooling_threshold < 0:
            raise ValueError("Cooling threshold must be non-negative")

    def get_regret_history(self) -> List[RegretEntry]:
        """Get a copy of regret history."""
        return list(self._regret_history)

    def serialize(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "config": self.config.__dict__,
            "regret_history": [
                {
                    "amount": r.amount,
                    "category": r.category,
                    "regret_score": r.regret_score,
                    "timestamp": r.timestamp,
                }
                for r in self._regret_history
            ],
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> BehavioralEngine:
        """Deserialize with validation."""
        config = BehavioralConfig(**data.get("config", {})) if data.get("config") else None
        engine = cls(config)
        if isinstance(data.get("regret_history"), list):
            for entry in data["regret_history"]:
                if (
                    isinstance(entry.get("amount"), (int, float))
                    and math.isfinite(entry["amount"])
                    and isinstance(entry.get("regret_score"), (int, float))
                    and 0 <= entry["regret_score"] <= 10
                ):
                    engine._regret_history.append(RegretEntry(
                        amount=entry["amount"],
                        category=str(entry.get("category", "unknown")),
                        regret_score=entry["regret_score"],
                        timestamp=str(entry.get("timestamp", "")),
                    ))
        return engine
