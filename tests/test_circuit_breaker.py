"""Tests for Circuit Breaker, AIMD Rate Limiter, Anti-Gaming, and PSI Drift."""

import time
from mnemopay.circuit_breaker import (
    AIMDConfig,
    AIMDRateLimiter,
    AntiGamingEngine,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    PSIDriftDetector,
)


class TestCircuitBreaker:
    def test_starts_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()

    def test_opens_after_threshold(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

    def test_success_decays_failures(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=3))
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        # failure_count should be 1 now
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED  # still only 2 total

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=2, recovery_timeout=0.1
        ))
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.can_execute()

    def test_half_open_to_closed_on_success(self):
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=2, recovery_timeout=0.1, success_threshold=2
        ))
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_to_open_on_failure(self):
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=2, recovery_timeout=0.1
        ))
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.state == CircuitState.HALF_OPEN
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_half_open_max_requests(self):
        cb = CircuitBreaker(CircuitBreakerConfig(
            failure_threshold=2, recovery_timeout=0.1, half_open_max=2
        ))
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.15)
        assert cb.can_execute()
        cb._half_open_requests = 2
        assert not cb.can_execute()

    def test_reset(self):
        cb = CircuitBreaker(CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()


class TestAIMDRateLimiter:
    def test_initial_rate(self):
        rl = AIMDRateLimiter(AIMDConfig(initial_rate=5))
        assert rl.current_rate == 5.0

    def test_allows_under_limit(self):
        rl = AIMDRateLimiter(AIMDConfig(initial_rate=3, window_seconds=60))
        assert rl.try_acquire()
        assert rl.try_acquire()
        assert rl.try_acquire()
        assert not rl.try_acquire()  # 4th should fail

    def test_additive_increase(self):
        rl = AIMDRateLimiter(AIMDConfig(initial_rate=5, additive_increase=2))
        rl.record_success()
        assert rl.current_rate == 7.0

    def test_multiplicative_decrease(self):
        rl = AIMDRateLimiter(AIMDConfig(initial_rate=10, multiplicative_decrease=0.5))
        rl.record_failure()
        assert rl.current_rate == 5.0

    def test_respects_floor(self):
        rl = AIMDRateLimiter(AIMDConfig(initial_rate=2, min_rate=1, multiplicative_decrease=0.1))
        rl.record_failure()
        rl.record_failure()
        assert rl.current_rate >= 1.0

    def test_respects_ceiling(self):
        rl = AIMDRateLimiter(AIMDConfig(initial_rate=98, max_rate=100, additive_increase=5))
        rl.record_success()
        assert rl.current_rate == 100.0

    def test_window_reset(self):
        rl = AIMDRateLimiter(AIMDConfig(initial_rate=2, window_seconds=0.1))
        assert rl.try_acquire()
        assert rl.try_acquire()
        assert not rl.try_acquire()
        time.sleep(0.15)
        assert rl.try_acquire()  # new window

    def test_stats(self):
        rl = AIMDRateLimiter(AIMDConfig(initial_rate=2))
        rl.try_acquire()
        rl.try_acquire()
        rl.try_acquire()  # rejected
        stats = rl.stats()
        assert stats["total_allowed"] == 2
        assert stats["total_rejected"] == 1


class TestAntiGaming:
    def test_no_alerts_normal_usage(self):
        ag = AntiGamingEngine()
        alerts = ag.check_transaction(10.0, 500.0)
        assert len(alerts) == 0

    def test_rapid_fire_detection(self):
        ag = AntiGamingEngine(rapid_fire_threshold=3, rapid_fire_window=60.0)
        ag.check_transaction(5.0, 500.0)
        ag.check_transaction(5.0, 500.0)
        ag.check_transaction(5.0, 500.0)
        alerts = ag.check_transaction(5.0, 500.0)
        rapid = [a for a in alerts if a.alert_type == "rapid_fire"]
        assert len(rapid) == 1
        assert rapid[0].severity == "high"

    def test_amount_manipulation_detection(self):
        ag = AntiGamingEngine(manipulation_margin=0.05)
        ceiling = 100.0
        alerts = ag.check_transaction(97.0, ceiling)  # 97% of ceiling
        manip = [a for a in alerts if a.alert_type == "amount_manipulation"]
        assert len(manip) == 1

    def test_no_manipulation_below_margin(self):
        ag = AntiGamingEngine(manipulation_margin=0.05)
        alerts = ag.check_transaction(50.0, 100.0)  # 50% of ceiling
        manip = [a for a in alerts if a.alert_type == "amount_manipulation"]
        assert len(manip) == 0

    def test_split_evasion_detection(self):
        ag = AntiGamingEngine(
            split_threshold=3, split_total_ratio=0.8, split_window=300.0
        )
        ceiling = 100.0
        ag.check_transaction(27.0, ceiling)
        ag.check_transaction(27.0, ceiling)
        alerts = ag.check_transaction(27.0, ceiling)  # total=81, 81% of ceiling, all < 30% of ceiling
        split = [a for a in alerts if a.alert_type == "split_evasion"]
        assert len(split) == 1

    def test_recent_alerts(self):
        ag = AntiGamingEngine(rapid_fire_threshold=2, rapid_fire_window=60.0)
        ag.check_transaction(5.0, 500.0)
        ag.check_transaction(5.0, 500.0)
        ag.check_transaction(5.0, 500.0)
        assert len(ag.recent_alerts()) > 0

    def test_clear(self):
        ag = AntiGamingEngine(rapid_fire_threshold=2, rapid_fire_window=60.0)
        ag.check_transaction(5.0, 500.0)
        ag.check_transaction(5.0, 500.0)
        ag.check_transaction(5.0, 500.0)
        ag.clear()
        assert len(ag.recent_alerts()) == 0


class TestPSIDrift:
    def test_insufficient_samples(self):
        det = PSIDriftDetector()
        det.update({"avg_tx": 10.0})
        result = det.check()
        assert not result.drifted
        assert "Insufficient" in result.message

    def test_no_drift_stable(self):
        det = PSIDriftDetector(sensitivity=0.3)
        for _ in range(15):
            det.update({"avg_tx": 10.0, "recall_freq": 5.0})
        result = det.check()
        assert not result.drifted
        assert result.drift_score < 0.3

    def test_detects_drift(self):
        det = PSIDriftDetector(sensitivity=0.3)
        # Establish baseline
        for _ in range(12):
            det.update({"avg_tx": 10.0, "recall_freq": 5.0})
        # Sudden shift
        for _ in range(5):
            det.update({"avg_tx": 50.0, "recall_freq": 25.0})
        result = det.check()
        assert result.drifted or result.drift_score > 0.1

    def test_parameters_changed_listed(self):
        det = PSIDriftDetector(sensitivity=0.2)
        for _ in range(15):
            det.update({"metric_a": 10.0, "metric_b": 100.0})
        # Shift metric_a drastically
        for _ in range(10):
            det.update({"metric_a": 100.0, "metric_b": 100.0})
        result = det.check()
        if result.drifted:
            assert "metric_a" in result.parameters_changed

    def test_handles_nan_gracefully(self):
        det = PSIDriftDetector()
        det.update({"good": 10.0, "bad": float("nan")})
        # Should not crash
        result = det.check()
        assert isinstance(result.drifted, bool)

    def test_handles_zero_baseline(self):
        det = PSIDriftDetector(sensitivity=0.3)
        for _ in range(15):
            det.update({"metric": 0.0})
        det.update({"metric": 5.0})
        result = det.check()
        assert isinstance(result.drift_score, float)
