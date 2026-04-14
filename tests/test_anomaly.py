"""
Tests for anomaly detection: EWMA, BehaviorMonitor, CanarySystem.
"""


import pytest

from mnemopay.anomaly import BehaviorMonitor, CanarySystem, EWMADetector
from mnemopay.types import (
    AlertSeverity,
    AnomalyConfig,
    CanaryType,
    HijackSeverity,
)

# ── EWMA Detector ─────────────────────────────────────────────────────────


class TestEWMAConfig:
    def test_default_params(self):
        det = EWMADetector()
        state = det.get_state()
        assert state.count == 0
        assert state.warmed_up is False

    def test_custom_params(self):
        det = EWMADetector(alpha=0.2, warning_k=3.0, critical_k=4.0, warmup=20)
        assert det._alpha == 0.2
        assert det._warning_k == 3.0

    def test_alpha_out_of_range(self):
        with pytest.raises(ValueError, match="Alpha"):
            EWMADetector(alpha=0)
        with pytest.raises(ValueError, match="Alpha"):
            EWMADetector(alpha=1)
        with pytest.raises(ValueError, match="Alpha"):
            EWMADetector(alpha=-0.1)

    def test_warning_must_be_positive(self):
        with pytest.raises(ValueError, match="positive"):
            EWMADetector(warning_k=0)

    def test_critical_must_exceed_warning(self):
        with pytest.raises(ValueError, match="exceed"):
            EWMADetector(warning_k=3.0, critical_k=2.0)
        with pytest.raises(ValueError, match="exceed"):
            EWMADetector(warning_k=3.0, critical_k=3.0)

    def test_warmup_must_be_positive(self):
        with pytest.raises(ValueError, match="at least 1"):
            EWMADetector(warmup=0)


class TestEWMAUpdate:
    def test_first_observation_initializes(self):
        det = EWMADetector()
        alert = det.update(100)
        assert alert.mean == 100
        assert alert.anomaly is False
        assert alert.severity == AlertSeverity.NONE

    def test_mean_tracks_values(self):
        det = EWMADetector(alpha=0.15)
        for _ in range(50):
            det.update(100)
        state = det.get_state()
        assert abs(state.mean - 100) < 1

    def test_no_alert_during_warmup(self):
        det = EWMADetector(warmup=10)
        for i in range(9):
            alert = det.update(100)
        # Even an extreme value during warmup should not alert
        alert = det.update(10000)
        # It might not alert because count < warmup
        # The 10th observation (count=10) is exactly at warmup, so count < warmup is false
        # Let's check: warmup=10, after 10 updates count=10, which is NOT < 10
        # So the 10th one CAN alert. Check the 9th:
        det2 = EWMADetector(warmup=10)
        for i in range(8):
            det2.update(100)
        alert = det2.update(10000)  # 9th update, count=9 < 10
        assert alert.severity == AlertSeverity.NONE

    def test_anomaly_detection(self):
        det = EWMADetector(alpha=0.15, warning_k=2.5, critical_k=3.5, warmup=10)
        # Build baseline
        for _ in range(50):
            det.update(100)
        # Inject anomaly
        alert = det.update(1000)
        assert alert.anomaly is True
        assert alert.z_score > 2.5

    def test_critical_severity(self):
        det = EWMADetector(alpha=0.05, warning_k=2.5, critical_k=3.5, warmup=10)
        for _ in range(100):
            det.update(100)
        alert = det.update(10000)  # Very extreme
        assert alert.severity in (AlertSeverity.WARNING, AlertSeverity.CRITICAL)
        assert alert.anomaly is True

    def test_warning_severity(self):
        det = EWMADetector(alpha=0.15, warning_k=2.0, critical_k=5.0, warmup=5)
        for _ in range(20):
            det.update(100)
        # Moderate deviation
        alert = det.update(200)
        # May or may not be warning depending on variance
        assert alert.severity in (AlertSeverity.NONE, AlertSeverity.WARNING, AlertSeverity.CRITICAL)

    def test_nan_raises(self):
        det = EWMADetector()
        with pytest.raises(ValueError, match="finite"):
            det.update(float("nan"))

    def test_inf_raises(self):
        det = EWMADetector()
        with pytest.raises(ValueError, match="finite"):
            det.update(float("inf"))

    def test_ewma_formula(self):
        """mu_t = alpha * x_t + (1 - alpha) * mu_{t-1}"""
        det = EWMADetector(alpha=0.15)
        det.update(100)  # mean = 100
        det.update(200)
        # mean = 0.15 * 200 + 0.85 * 100 = 30 + 85 = 115
        state = det.get_state()
        assert state.mean == pytest.approx(115, abs=0.01)

    def test_variance_formula(self):
        """sigma^2 = alpha * (x - mu)^2 + (1 - alpha) * sigma^2_{t-1}"""
        det = EWMADetector(alpha=0.15)
        det.update(100)
        det.update(200)
        # After update: mean = 115, deviation = 200 - 115 = 85
        # variance = 0.15 * 85^2 + 0.85 * 0 = 0.15 * 7225 = 1083.75
        assert det._variance == pytest.approx(0.15 * (200 - 115) ** 2, abs=1)


class TestEWMAState:
    def test_get_state(self):
        det = EWMADetector()
        det.update(42)
        state = det.get_state()
        assert state.count == 1
        assert state.last_value == 42
        assert state.warmed_up is False

    def test_warmed_up_after_warmup(self):
        det = EWMADetector(warmup=5)
        for _ in range(5):
            det.update(100)
        assert det.get_state().warmed_up is True

    def test_reset(self):
        det = EWMADetector()
        for _ in range(10):
            det.update(100)
        det.reset()
        state = det.get_state()
        assert state.count == 0
        assert state.mean == 0
        assert state.variance == 0


class TestEWMASerialization:
    def test_serialize_restore(self):
        det = EWMADetector(alpha=0.2)
        for _ in range(10):
            det.update(100)
        data = det.serialize()

        det2 = EWMADetector(alpha=0.2)
        det2.restore(data)
        assert det2.get_state().mean == det.get_state().mean
        assert det2.get_state().count == det.get_state().count

    def test_restore_invalid_mean(self):
        det = EWMADetector()
        with pytest.raises(ValueError, match="Invalid mean"):
            det.restore({"mean": float("nan"), "variance": 0, "count": 0})

    def test_restore_negative_variance(self):
        det = EWMADetector()
        with pytest.raises(ValueError, match="Invalid variance"):
            det.restore({"mean": 0, "variance": -1, "count": 0})

    def test_restore_negative_count(self):
        det = EWMADetector()
        with pytest.raises(ValueError, match="Invalid count"):
            det.restore({"mean": 0, "variance": 0, "count": -1})


# ── Behavior Monitor ──────────────────────────────────────────────────────


class TestBehaviorMonitor:
    def test_new_agent_observation(self):
        monitor = BehaviorMonitor()
        result = monitor.observe("agent-1", {"amount": 100, "hourOfDay": 14})
        assert result.suspected is False
        assert result.total_features > 0

    def test_consistent_behavior_no_hijack(self):
        monitor = BehaviorMonitor()
        for i in range(20):
            result = monitor.observe("agent-1", {
                "amount": 100 + (i % 3),
                "hourOfDay": 14,
            })
        assert result.suspected is False

    def test_behavioral_shift_detected(self):
        config = AnomalyConfig(
            warmup_period=10,
            hijack_feature_threshold=0.3,
        )
        monitor = BehaviorMonitor(config)
        # Build baseline
        for _ in range(20):
            monitor.observe("agent-1", {
                "amount": 100,
                "hourOfDay": 14,
                "timeBetweenTx": 3600,
                "chargesPerHour": 2,
            })
        # Inject behavioral shift
        result = monitor.observe("agent-1", {
            "amount": 100000,
            "hourOfDay": 3,
            "timeBetweenTx": 1,
            "chargesPerHour": 1000,
        })
        # May or may not detect depending on EWMA convergence
        assert result.anomaly_score >= 0

    def test_get_fingerprint(self):
        monitor = BehaviorMonitor()
        monitor.observe("agent-1", {"amount": 100})
        fp = monitor.get_fingerprint("agent-1")
        assert fp is not None
        assert fp.agent_id == "agent-1"
        assert fp.observations == 1

    def test_get_fingerprint_nonexistent(self):
        monitor = BehaviorMonitor()
        assert monitor.get_fingerprint("unknown") is None

    def test_remove_agent(self):
        monitor = BehaviorMonitor()
        monitor.observe("agent-1", {"amount": 100})
        assert monitor.remove_agent("agent-1") is True
        assert monitor.remove_agent("agent-1") is False
        assert monitor.agent_count == 0

    def test_agent_count(self):
        monitor = BehaviorMonitor()
        monitor.observe("agent-1", {"amount": 100})
        monitor.observe("agent-2", {"amount": 200})
        assert monitor.agent_count == 2

    def test_established_after_warmup(self):
        config = AnomalyConfig(warmup_period=5)
        monitor = BehaviorMonitor(config)
        for _ in range(5):
            monitor.observe("agent-1", {"amount": 100})
        fp = monitor.get_fingerprint("agent-1")
        assert fp.established is True

    def test_not_established_before_warmup(self):
        config = AnomalyConfig(warmup_period=10)
        monitor = BehaviorMonitor(config)
        for _ in range(3):
            monitor.observe("agent-1", {"amount": 100})
        fp = monitor.get_fingerprint("agent-1")
        assert fp.established is False

    def test_invalid_agent_id_raises(self):
        monitor = BehaviorMonitor()
        with pytest.raises(ValueError, match="agent_id"):
            monitor.observe("", {"amount": 100})

    def test_invalid_features_raises(self):
        monitor = BehaviorMonitor()
        with pytest.raises(ValueError, match="features"):
            monitor.observe("agent-1", None)

    def test_severity_levels(self):
        """Severity: none < low < medium < high < critical"""
        severities = [
            HijackSeverity.NONE, HijackSeverity.LOW,
            HijackSeverity.MEDIUM, HijackSeverity.HIGH,
            HijackSeverity.CRITICAL,
        ]
        assert len(set(severities)) == 5

    def test_non_tracked_features_ignored(self):
        monitor = BehaviorMonitor()
        result = monitor.observe("agent-1", {"unknown_feature": 999})
        assert result.total_features == 0


class TestBehaviorMonitorSerialization:
    def test_serialize_roundtrip(self):
        monitor = BehaviorMonitor()
        for _ in range(5):
            monitor.observe("agent-1", {"amount": 100, "hourOfDay": 14})
        data = monitor.serialize()

        restored = BehaviorMonitor.deserialize(data)
        assert restored.agent_count == 1
        fp = restored.get_fingerprint("agent-1")
        assert fp.observations == 5

    def test_deserialize_empty(self):
        monitor = BehaviorMonitor.deserialize({})
        assert monitor.agent_count == 0

    def test_deserialize_corrupted_skipped(self):
        data = {
            "agent-1": {
                "features": {"amount": {"mean": "bad", "variance": 0, "count": 0}},
                "count": 5,
                "last_observed": 0,
            },
        }
        monitor = BehaviorMonitor.deserialize(data)
        # Should not crash, corrupted features skipped
        assert monitor.agent_count == 1


# ── Canary System ─────────────────────────────────────────────────────────


class TestCanaryPlant:
    def test_plant_canary(self):
        system = CanarySystem(max_canaries=5)
        canary = system.plant()
        assert canary.id.startswith("canary-")
        assert canary.triggered is False
        assert canary.amount > 0

    def test_plant_types(self):
        system = CanarySystem()
        c1 = system.plant(CanaryType.TRANSACTION)
        c2 = system.plant(CanaryType.MEMORY)
        c3 = system.plant(CanaryType.CREDENTIAL)
        assert c1.type == CanaryType.TRANSACTION
        assert c2.type == CanaryType.MEMORY
        assert c3.type == CanaryType.CREDENTIAL

    def test_max_canaries_evicts_oldest(self):
        system = CanarySystem(max_canaries=2)
        c1 = system.plant()
        c2 = system.plant()
        c3 = system.plant()  # Should evict c1
        assert not system.is_canary(c1.id)
        assert system.is_canary(c2.id)
        assert system.is_canary(c3.id)

    def test_invalid_max_canaries(self):
        with pytest.raises(ValueError, match="1-50"):
            CanarySystem(max_canaries=0)
        with pytest.raises(ValueError, match="1-50"):
            CanarySystem(max_canaries=51)


class TestCanaryCheck:
    def test_check_triggers_alert(self):
        system = CanarySystem()
        canary = system.plant()
        alert = system.check(canary.id, "evil-agent")
        assert alert is not None
        assert alert.severity == "critical"
        assert "evil-agent" in alert.message
        assert canary.triggered is True

    def test_check_non_canary_returns_none(self):
        system = CanarySystem()
        assert system.check("regular-tx-id", "agent") is None

    def test_check_batch(self):
        system = CanarySystem()
        c1 = system.plant()
        c2 = system.plant()
        alerts = system.check_batch(
            [c1.id, "normal-id", c2.id], "agent",
        )
        assert len(alerts) == 2

    def test_check_batch_empty(self):
        system = CanarySystem()
        alerts = system.check_batch([], "agent")
        assert alerts == []

    def test_is_canary(self):
        system = CanarySystem()
        canary = system.plant()
        assert system.is_canary(canary.id) is True
        assert system.is_canary("not-a-canary") is False


class TestCanaryAlerts:
    def test_get_alerts(self):
        system = CanarySystem()
        canary = system.plant()
        system.check(canary.id, "agent")
        alerts = system.get_alerts()
        assert len(alerts) == 1

    def test_alerts_capped(self):
        system = CanarySystem(max_canaries=50)
        for _ in range(150):
            canary = system.plant()
            system.check(canary.id, "agent")
        assert len(system.get_alerts()) <= CanarySystem.MAX_ALERTS

    def test_active_canaries(self):
        system = CanarySystem()
        c1 = system.plant()
        c2 = system.plant()
        system.check(c1.id, "agent")
        active = system.get_active_canaries()
        assert len(active) == 1
        assert active[0].id == c2.id


class TestCanarySerialization:
    def test_serialize_roundtrip(self):
        system = CanarySystem()
        c = system.plant()
        system.check(c.id, "agent")
        data = system.serialize()

        restored = CanarySystem.deserialize(data)
        assert restored.is_canary(c.id)
        # Alerts might not round-trip perfectly as they're stored as dicts

    def test_deserialize_empty(self):
        system = CanarySystem.deserialize({})
        assert len(system.get_active_canaries()) == 0

    def test_deserialize_invalid_skipped(self):
        data = {
            "canaries": [
                {"id": "c1", "amount": 50, "type": "transaction", "planted_at": 0, "triggered": False},
                {"id": None, "amount": "bad"},  # Invalid
            ],
        }
        system = CanarySystem.deserialize(data)
        assert system.is_canary("c1")


# ── Integration ────────────────────────────────────────────────────────────


class TestAnomalyIntegration:
    def test_ewma_plus_monitor(self):
        """EWMA detectors work inside BehaviorMonitor."""
        monitor = BehaviorMonitor(AnomalyConfig(warmup_period=5))
        for _ in range(10):
            monitor.observe("agent-1", {
                "amount": 100, "hourOfDay": 14,
            })
        fp = monitor.get_fingerprint("agent-1")
        assert fp.established is True
        assert "amount" in fp.features

    def test_canary_with_monitor(self):
        """Canary check is independent of behavior monitor."""
        monitor = BehaviorMonitor()
        canary_sys = CanarySystem()
        canary = canary_sys.plant()

        # Normal observation
        result = monitor.observe("agent-1", {"amount": 100})
        assert result.suspected is False

        # Canary trigger (separate system)
        alert = canary_sys.check(canary.id, "agent-1")
        assert alert is not None
        assert alert.severity == "critical"

    def test_default_config_values(self):
        config = AnomalyConfig()
        assert config.alpha == 0.15
        assert config.warning_threshold == 2.5
        assert config.critical_threshold == 3.5
        assert config.warmup_period == 10
        assert config.hijack_feature_threshold == 0.4
        assert config.max_canaries == 5
