"""
Circuit Breaker + AIMD Rate Limiting for MnemoPay agents.

Port of TypeScript v0.9.3 features:
  - Circuit breaker (closed → open → half-open) for payment rail failures
  - Asymmetric AIMD rate limiting (additive increase, multiplicative decrease)
  - Anti-gaming protections (rapid-fire detection, amount manipulation)
  - PSI drift detection (behavioral parameter shift indicator)

Zero external dependencies (stdlib only).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ── Circuit Breaker ──────────────────────────────────────────────────────


class CircuitState(Enum):
    CLOSED = "closed"        # Normal operation
    OPEN = "open"            # Failing — block requests
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5       # Failures before opening
    recovery_timeout: float = 30.0   # Seconds before trying half-open
    half_open_max: int = 3           # Max requests in half-open
    success_threshold: int = 2       # Successes in half-open to close


class CircuitBreaker:
    """Circuit breaker for payment rail reliability.

    Usage:
        cb = CircuitBreaker()
        if cb.can_execute():
            try:
                result = do_payment()
                cb.record_success()
            except Exception:
                cb.record_failure()
        else:
            # Circuit is open — fail fast
            raise RuntimeError("Payment rail unavailable")
    """

    def __init__(self, config: Optional[CircuitBreakerConfig] = None) -> None:
        self._config = config or CircuitBreakerConfig()
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = 0.0
        self._half_open_requests = 0

    @property
    def state(self) -> CircuitState:
        # Auto-transition from OPEN to HALF_OPEN after timeout
        if self._state == CircuitState.OPEN:
            elapsed = time.monotonic() - self._last_failure_time
            if elapsed >= self._config.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_requests = 0
                self._success_count = 0
        return self._state

    def can_execute(self) -> bool:
        """Check if a request should be allowed."""
        state = self.state
        if state == CircuitState.CLOSED:
            return True
        if state == CircuitState.HALF_OPEN:
            return self._half_open_requests < self._config.half_open_max
        return False  # OPEN

    def record_success(self) -> None:
        """Record a successful operation."""
        state = self.state
        if state == CircuitState.HALF_OPEN:
            self._success_count += 1
            self._half_open_requests += 1
            if self._success_count >= self._config.success_threshold:
                self._state = CircuitState.CLOSED
                self._failure_count = 0
                self._success_count = 0
        elif state == CircuitState.CLOSED:
            # Decay failure count on success
            if self._failure_count > 0:
                self._failure_count -= 1

    def record_failure(self) -> None:
        """Record a failed operation."""
        state = self.state
        if state == CircuitState.HALF_OPEN:
            # Any failure in half-open → back to open
            self._state = CircuitState.OPEN
            self._last_failure_time = time.monotonic()
            self._failure_count = self._config.failure_threshold
        elif state == CircuitState.CLOSED:
            self._failure_count += 1
            if self._failure_count >= self._config.failure_threshold:
                self._state = CircuitState.OPEN
                self._last_failure_time = time.monotonic()

    def reset(self) -> None:
        """Force reset to closed state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_requests = 0


# ── AIMD Rate Limiter ────────────────────────────────────────────────────


@dataclass
class AIMDConfig:
    initial_rate: float = 10.0       # Requests per window
    min_rate: float = 1.0            # Floor
    max_rate: float = 100.0          # Ceiling
    additive_increase: float = 1.0   # Increase per successful window
    multiplicative_decrease: float = 0.5  # Decrease factor on failure
    window_seconds: float = 60.0     # Rate window


class AIMDRateLimiter:
    """Asymmetric AIMD rate limiter.

    Slow to increase trust, fast to decrease on abuse.
    """

    def __init__(self, config: Optional[AIMDConfig] = None) -> None:
        self._config = config or AIMDConfig()
        self._current_rate = self._config.initial_rate
        self._window_count = 0
        self._window_start = time.monotonic()
        self._total_allowed = 0
        self._total_rejected = 0

    @property
    def current_rate(self) -> float:
        return self._current_rate

    def try_acquire(self) -> bool:
        """Try to acquire a rate limit token. Returns True if allowed."""
        self._check_window()
        if self._window_count < self._current_rate:
            self._window_count += 1
            self._total_allowed += 1
            return True
        self._total_rejected += 1
        return False

    def record_success(self) -> None:
        """Record successful operation — additive increase."""
        self._current_rate = min(
            self._current_rate + self._config.additive_increase,
            self._config.max_rate,
        )

    def record_failure(self) -> None:
        """Record failure — multiplicative decrease."""
        self._current_rate = max(
            self._current_rate * self._config.multiplicative_decrease,
            self._config.min_rate,
        )

    def stats(self) -> Dict[str, Any]:
        return {
            "current_rate": self._current_rate,
            "window_count": self._window_count,
            "total_allowed": self._total_allowed,
            "total_rejected": self._total_rejected,
        }

    def _check_window(self) -> None:
        now = time.monotonic()
        if now - self._window_start >= self._config.window_seconds:
            self._window_start = now
            self._window_count = 0


# ── Anti-Gaming ──────────────────────────────────────────────────────────


@dataclass
class AntiGamingAlert:
    alert_type: str   # "rapid_fire", "amount_manipulation", "split_evasion"
    severity: str     # "low", "medium", "high"
    message: str
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


class AntiGamingEngine:
    """Detects gaming attempts against the payment/reputation system.

    Checks for:
      - Rapid-fire transactions (burst detection)
      - Amount manipulation (just-under-limit charges)
      - Split evasion (breaking large charges into small ones)
    """

    def __init__(
        self,
        rapid_fire_window: float = 10.0,  # seconds
        rapid_fire_threshold: int = 5,     # max txns in window
        split_window: float = 300.0,       # 5 min
        split_threshold: int = 3,          # min fragments
        split_total_ratio: float = 0.8,    # fragments/limit ratio
        manipulation_margin: float = 0.05, # 5% under limit
    ) -> None:
        self._rapid_window = rapid_fire_window
        self._rapid_threshold = rapid_fire_threshold
        self._split_window = split_window
        self._split_threshold = split_threshold
        self._split_ratio = split_total_ratio
        self._manip_margin = manipulation_margin
        self._recent_txns: List[Dict[str, Any]] = []
        self._alerts: List[AntiGamingAlert] = []

    def check_transaction(
        self,
        amount: float,
        reputation_ceiling: float,
    ) -> List[AntiGamingAlert]:
        """Check a transaction for gaming patterns. Returns any alerts."""
        now = time.monotonic()
        self._recent_txns.append({"amount": amount, "time": now})
        # Prune old transactions
        self._recent_txns = [
            t for t in self._recent_txns
            if now - t["time"] < self._split_window
        ]

        alerts: List[AntiGamingAlert] = []

        # 1. Rapid-fire detection
        recent = [t for t in self._recent_txns if now - t["time"] < self._rapid_window]
        if len(recent) > self._rapid_threshold:
            alert = AntiGamingAlert(
                alert_type="rapid_fire",
                severity="high",
                message=f"{len(recent)} transactions in {self._rapid_window}s window",
                timestamp=now,
                details={"count": len(recent), "window": self._rapid_window},
            )
            alerts.append(alert)

        # 2. Amount manipulation (just under ceiling)
        if reputation_ceiling > 0 and amount > 0:
            ratio = amount / reputation_ceiling
            if 1.0 - ratio < self._manip_margin and ratio < 1.0:
                alert = AntiGamingAlert(
                    alert_type="amount_manipulation",
                    severity="medium",
                    message=f"Amount ${amount:.2f} is {ratio*100:.1f}% of ceiling ${reputation_ceiling:.2f}",
                    timestamp=now,
                    details={"amount": amount, "ceiling": reputation_ceiling, "ratio": ratio},
                )
                alerts.append(alert)

        # 3. Split evasion (many small charges summing near ceiling)
        if len(self._recent_txns) >= self._split_threshold and reputation_ceiling > 0:
            total = sum(t["amount"] for t in self._recent_txns)
            if total / reputation_ceiling >= self._split_ratio:
                # Check if individual amounts are small (< 30% of ceiling)
                small_count = sum(
                    1 for t in self._recent_txns
                    if t["amount"] < reputation_ceiling * 0.3
                )
                if small_count >= self._split_threshold:
                    alert = AntiGamingAlert(
                        alert_type="split_evasion",
                        severity="high",
                        message=f"{small_count} small charges totaling ${total:.2f} "
                                f"({total/reputation_ceiling*100:.0f}% of ceiling)",
                        timestamp=now,
                        details={
                            "fragments": small_count,
                            "total": total,
                            "ceiling": reputation_ceiling,
                        },
                    )
                    alerts.append(alert)

        self._alerts.extend(alerts)
        return alerts

    def recent_alerts(self, limit: int = 20) -> List[AntiGamingAlert]:
        return self._alerts[-limit:]

    def clear(self) -> None:
        self._recent_txns.clear()
        self._alerts.clear()


# ── PSI Drift Detection ─────────────────────────────────────────────────


@dataclass
class PSIDriftResult:
    drifted: bool
    drift_score: float       # 0-1, higher = more drift
    parameters_changed: List[str]
    message: str


class PSIDriftDetector:
    """Detects behavioral parameter shift (PSI drift).

    Monitors changes in agent behavior patterns over time:
    - Average transaction size shifts
    - Recall pattern changes
    - Activity frequency changes
    """

    def __init__(self, sensitivity: float = 0.3) -> None:
        self._sensitivity = min(max(sensitivity, 0.0), 1.0)
        self._baseline: Dict[str, float] = {}
        self._current: Dict[str, float] = {}
        self._sample_count = 0

    def update(self, metrics: Dict[str, float]) -> None:
        """Update current metrics snapshot."""
        self._sample_count += 1

        for key, value in metrics.items():
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                continue

            if key not in self._baseline:
                self._baseline[key] = value
            else:
                # Exponential moving average for baseline
                alpha = 0.05
                self._baseline[key] = (1 - alpha) * self._baseline[key] + alpha * value

            self._current[key] = value

    def check(self) -> PSIDriftResult:
        """Check for behavioral drift against baseline."""
        if self._sample_count < 10:
            return PSIDriftResult(
                drifted=False, drift_score=0.0,
                parameters_changed=[], message="Insufficient samples",
            )

        changed: List[str] = []
        total_drift = 0.0
        count = 0

        for key in self._baseline:
            if key not in self._current:
                continue
            baseline = self._baseline[key]
            current = self._current[key]

            if baseline == 0:
                if current != 0:
                    drift = 1.0
                else:
                    drift = 0.0
            else:
                drift = abs(current - baseline) / abs(baseline)

            if drift > self._sensitivity:
                changed.append(key)

            total_drift += min(drift, 1.0)
            count += 1

        avg_drift = total_drift / count if count > 0 else 0.0
        drifted = len(changed) > 0 and avg_drift > self._sensitivity

        return PSIDriftResult(
            drifted=drifted,
            drift_score=round(avg_drift, 4),
            parameters_changed=changed,
            message=f"Drift detected in {len(changed)} parameters" if drifted
                    else "No significant drift",
        )
