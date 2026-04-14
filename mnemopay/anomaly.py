"""
Streaming Anomaly Detection -- Real-Time Agent Behavior Monitoring

Three complementary systems:
  1. EWMA Detector -- O(1) per update, no windowing, no buffer overflow
  2. Behavioral Fingerprint Monitor -- multi-dimensional behavioral profiling
  3. Canary Transaction System -- honeypot transactions for compromise detection

References:
  - Roberts, S.W. (1959). "Control Chart Tests Based on EWMA"
  - Lucas & Saccucci (1990). "EWMA Control Chart Properties"
  - Mahalanobis (1936). "On the Generalized Distance in Statistics"
"""

from __future__ import annotations

import math
import random
import time
from typing import Any, Dict, List, Optional

from .types import (
    AlertSeverity,
    AnomalyConfig,
    BehaviorFingerprint,
    CanaryAlert,
    CanaryTransaction,
    CanaryType,
    EWMAAlert,
    EWMAState,
    HijackDetection,
    HijackSeverity,
)


class EWMADetector:
    """Exponentially Weighted Moving Average anomaly detector.

    EWMA formulas:
      mu_t = alpha * x_t + (1 - alpha) * mu_{t-1}
      sigma_t^2 = alpha * (x_t - mu_t)^2 + (1 - alpha) * sigma_{t-1}^2
      Alert when: |x_t - mu_t| > k * sigma_t
    """

    def __init__(
        self,
        alpha: float = 0.15,
        warning_k: float = 2.5,
        critical_k: float = 3.5,
        warmup: int = 10,
    ) -> None:
        if alpha <= 0 or alpha >= 1:
            raise ValueError("Alpha must be in (0, 1)")
        if warning_k <= 0:
            raise ValueError("Warning threshold must be positive")
        if critical_k <= warning_k:
            raise ValueError("Critical threshold must exceed warning threshold")
        if warmup < 1:
            raise ValueError("Warmup period must be at least 1")

        self._alpha = alpha
        self._warning_k = warning_k
        self._critical_k = critical_k
        self._warmup = warmup

        self._mean: float = 0.0
        self._variance: float = 0.0
        self._count: int = 0
        self._last_value: float = 0.0

    def update(self, value: float) -> EWMAAlert:
        """Update the detector with a new observation. Returns anomaly assessment."""
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("EWMA value must be a finite number")

        self._count += 1
        self._last_value = value

        if self._count == 1:
            self._mean = value
            self._variance = 0.0
            return self._make_alert(value, 0.0, AlertSeverity.NONE)

        # Update mean (EWMA)
        self._mean = self._alpha * value + (1 - self._alpha) * self._mean

        # Update variance (EWMA of squared deviations)
        deviation = value - self._mean
        self._variance = (
            self._alpha * (deviation * deviation)
            + (1 - self._alpha) * self._variance
        )

        # Z-score
        std_dev = math.sqrt(self._variance)
        z_score = abs(value - self._mean) / std_dev if std_dev > 1e-10 else 0.0

        # Don't alert during warmup
        if self._count < self._warmup:
            return self._make_alert(value, z_score, AlertSeverity.NONE)

        # Classify severity
        if z_score >= self._critical_k:
            severity = AlertSeverity.CRITICAL
        elif z_score >= self._warning_k:
            severity = AlertSeverity.WARNING
        else:
            severity = AlertSeverity.NONE

        return self._make_alert(value, z_score, severity)

    def _make_alert(
        self, value: float, z_score: float, severity: AlertSeverity,
    ) -> EWMAAlert:
        return EWMAAlert(
            anomaly=severity != AlertSeverity.NONE,
            value=value,
            mean=round(self._mean * 10000) / 10000,
            std_dev=round(math.sqrt(self._variance) * 10000) / 10000,
            z_score=round(z_score * 1000) / 1000,
            severity=severity,
            timestamp=time.time() * 1000,
        )

    def get_state(self) -> EWMAState:
        """Get current detector state."""
        return EWMAState(
            mean=self._mean,
            variance=self._variance,
            std_dev=math.sqrt(self._variance),
            count=self._count,
            last_value=self._last_value,
            warmed_up=self._count >= self._warmup,
        )

    def reset(self) -> None:
        """Reset the detector."""
        self._mean = 0.0
        self._variance = 0.0
        self._count = 0
        self._last_value = 0.0

    def serialize(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "mean": self._mean,
            "variance": self._variance,
            "count": self._count,
            "last_value": self._last_value,
        }

    def restore(self, state: Dict[str, Any]) -> None:
        """Restore from serialized state."""
        if not isinstance(state.get("mean"), (int, float)) or not math.isfinite(state["mean"]):
            raise ValueError("Invalid mean")
        if not isinstance(state.get("variance"), (int, float)) or not math.isfinite(state["variance"]) or state["variance"] < 0:
            raise ValueError("Invalid variance")
        if not isinstance(state.get("count"), (int, float)) or state["count"] < 0:
            raise ValueError("Invalid count")
        self._mean = state["mean"]
        self._variance = state["variance"]
        self._count = int(state["count"])
        self._last_value = state.get("last_value", 0.0)


class BehaviorMonitor:
    """Behavioral fingerprint monitor for multi-agent systems."""

    def __init__(self, config: Optional[AnomalyConfig] = None) -> None:
        self.config = config or AnomalyConfig()
        self._fingerprints: Dict[str, Dict[str, EWMADetector]] = {}
        self._observation_counts: Dict[str, int] = {}
        self._last_observed: Dict[str, float] = {}

    def observe(
        self, agent_id: str, features: Dict[str, float],
    ) -> HijackDetection:
        """Observe agent behavior and return hijack detection result."""
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("agent_id is required")
        if not features or not isinstance(features, dict):
            raise ValueError("features must be a dict")

        # Initialize fingerprint for new agents
        if agent_id not in self._fingerprints:
            self._fingerprints[agent_id] = {}
            self._observation_counts[agent_id] = 0

        detectors = self._fingerprints[agent_id]
        count = self._observation_counts.get(agent_id, 0) + 1
        self._observation_counts[agent_id] = count
        self._last_observed[agent_id] = time.time() * 1000

        details: Dict[str, Dict[str, Any]] = {}
        anomalous_count = 0
        total_features = 0

        for feature in self.config.tracked_features:
            value = features.get(feature)
            if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
                continue

            total_features += 1

            if feature not in detectors:
                detectors[feature] = EWMADetector(
                    alpha=self.config.alpha,
                    warning_k=self.config.warning_threshold,
                    critical_k=self.config.critical_threshold,
                    warmup=self.config.warmup_period,
                )

            alert = detectors[feature].update(value)
            details[feature] = {"z_score": alert.z_score, "anomalous": alert.anomaly}
            if alert.anomaly:
                anomalous_count += 1

        anomaly_score = anomalous_count / total_features if total_features > 0 else 0.0
        suspected = (
            anomaly_score >= self.config.hijack_feature_threshold
            and count >= self.config.warmup_period
        )

        if suspected and anomaly_score >= 0.8:
            severity = HijackSeverity.CRITICAL
        elif suspected and anomaly_score >= 0.6:
            severity = HijackSeverity.HIGH
        elif suspected:
            severity = HijackSeverity.MEDIUM
        elif anomalous_count > 0:
            severity = HijackSeverity.LOW
        else:
            severity = HijackSeverity.NONE

        if severity == HijackSeverity.CRITICAL:
            recommendation = (
                f"CRITICAL: {anomalous_count}/{total_features} behavioral features "
                f"deviant. Agent {agent_id} may be compromised. Recommended: "
                f"suspend transactions, require re-authentication."
            )
        elif severity == HijackSeverity.HIGH:
            recommendation = (
                f"HIGH RISK: {anomalous_count}/{total_features} features anomalous. "
                f"Increase monitoring. Consider HITL approval for large transactions."
            )
        elif severity == HijackSeverity.MEDIUM:
            recommendation = "ELEVATED: Some behavioral deviation detected. Monitor closely."
        elif severity == HijackSeverity.LOW:
            recommendation = "Minor deviations detected. Within acceptable range."
        else:
            recommendation = "Behavior consistent with established profile."

        return HijackDetection(
            suspected=suspected,
            anomaly_score=round(anomaly_score * 1000) / 1000,
            anomalous_features=anomalous_count,
            total_features=total_features,
            details=details,
            severity=severity,
            recommendation=recommendation,
        )

    def get_fingerprint(self, agent_id: str) -> Optional[BehaviorFingerprint]:
        """Get the behavioral fingerprint for an agent."""
        detectors = self._fingerprints.get(agent_id)
        if detectors is None:
            return None

        features = {name: detector.get_state() for name, detector in detectors.items()}
        observations = self._observation_counts.get(agent_id, 0)

        return BehaviorFingerprint(
            agent_id=agent_id,
            features=features,
            observations=observations,
            established=observations >= self.config.warmup_period,
            last_observed=self._last_observed.get(agent_id, 0),
        )

    def remove_agent(self, agent_id: str) -> bool:
        """Remove an agent's fingerprint."""
        existed = agent_id in self._fingerprints
        self._fingerprints.pop(agent_id, None)
        self._observation_counts.pop(agent_id, None)
        self._last_observed.pop(agent_id, None)
        return existed

    @property
    def agent_count(self) -> int:
        """Number of agents being monitored."""
        return len(self._fingerprints)

    def serialize(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        result: Dict[str, Any] = {}
        for agent_id, detectors in self._fingerprints.items():
            features = {name: det.serialize() for name, det in detectors.items()}
            result[agent_id] = {
                "features": features,
                "count": self._observation_counts.get(agent_id, 0),
                "last_observed": self._last_observed.get(agent_id, 0),
            }
        return result

    @classmethod
    def deserialize(
        cls,
        data: Dict[str, Any],
        config: Optional[AnomalyConfig] = None,
    ) -> BehaviorMonitor:
        """Deserialize with validation."""
        monitor = cls(config)
        if not data or not isinstance(data, dict):
            return monitor

        for agent_id, agent_data in data.items():
            if not agent_data or not isinstance(agent_data, dict):
                continue
            detectors: Dict[str, EWMADetector] = {}

            features_data = agent_data.get("features", {})
            if isinstance(features_data, dict):
                for feature, state in features_data.items():
                    try:
                        detector = EWMADetector(
                            alpha=monitor.config.alpha,
                            warning_k=monitor.config.warning_threshold,
                            critical_k=monitor.config.critical_threshold,
                            warmup=monitor.config.warmup_period,
                        )
                        detector.restore(state)
                        detectors[feature] = detector
                    except (ValueError, TypeError, KeyError):
                        pass  # Skip corrupted feature data

            monitor._fingerprints[agent_id] = detectors
            monitor._observation_counts[agent_id] = (
                agent_data["count"] if isinstance(agent_data.get("count"), (int, float)) else 0
            )
            monitor._last_observed[agent_id] = (
                agent_data["last_observed"]
                if isinstance(agent_data.get("last_observed"), (int, float)) else 0
            )

        return monitor


class CanarySystem:
    """Canary transaction system -- honeypots for compromise detection."""

    MAX_ALERTS = 100

    def __init__(self, max_canaries: int = 5) -> None:
        if max_canaries < 1 or max_canaries > 50:
            raise ValueError("max_canaries must be 1-50")
        self._max_canaries = max_canaries
        self._canaries: Dict[str, CanaryTransaction] = {}
        self._alerts: List[CanaryAlert] = []

    def plant(
        self, canary_type: CanaryType = CanaryType.TRANSACTION,
    ) -> CanaryTransaction:
        """Plant a canary transaction. Returns the canary."""
        if len(self._canaries) >= self._max_canaries:
            # Remove oldest untriggered canary
            oldest = None
            for c in self._canaries.values():
                if not c.triggered and (oldest is None or c.planted_at < oldest.planted_at):
                    oldest = c
            if oldest:
                del self._canaries[oldest.id]

        now = time.time() * 1000
        rand_suffix = "".join(
            random.choices("abcdefghijklmnopqrstuvwxyz0123456789", k=6)
        )
        canary = CanaryTransaction(
            id=f"canary-{int(now)}-{rand_suffix}",
            amount=round(random.uniform(1, 100) * 100) / 100,
            type=canary_type,
            planted_at=now,
            triggered=False,
        )

        self._canaries[canary.id] = canary
        return canary

    def check(self, id_: str, agent_id: str) -> Optional[CanaryAlert]:
        """Check if an ID matches a canary. If yes, the agent is compromised."""
        canary = self._canaries.get(id_)
        if canary is None:
            return None

        # TRIGGERED
        canary.triggered = True
        canary.triggered_at = time.time() * 1000
        canary.triggered_by = agent_id

        minutes_ago = round((time.time() * 1000 - canary.planted_at) / 60000)
        alert = CanaryAlert(
            canary=CanaryTransaction(
                id=canary.id,
                amount=canary.amount,
                type=canary.type,
                planted_at=canary.planted_at,
                triggered=canary.triggered,
                triggered_at=canary.triggered_at,
                triggered_by=canary.triggered_by,
            ),
            agent_id=agent_id,
            severity="critical",
            message=(
                f"CANARY TRIGGERED: Agent {agent_id} accessed {canary.type.value} "
                f'canary "{canary.id}". This agent may be compromised. '
                f"Canary was planted {minutes_ago} minutes ago."
            ),
            timestamp=time.time() * 1000,
        )

        self._alerts.append(alert)
        if len(self._alerts) > self.MAX_ALERTS:
            self._alerts = self._alerts[-self.MAX_ALERTS:]

        return alert

    def check_batch(self, ids: List[str], agent_id: str) -> List[CanaryAlert]:
        """Check if any ID in a list matches a canary."""
        alerts = []
        for id_ in ids:
            alert = self.check(id_, agent_id)
            if alert:
                alerts.append(alert)
        return alerts

    def get_alerts(self) -> List[CanaryAlert]:
        """Get all alerts."""
        return list(self._alerts)

    def get_active_canaries(self) -> List[CanaryTransaction]:
        """Get all active (untriggered) canaries."""
        return [c for c in self._canaries.values() if not c.triggered]

    def is_canary(self, id_: str) -> bool:
        """Check if a specific ID is a canary."""
        return id_ in self._canaries

    def serialize(self) -> Dict[str, Any]:
        """Serialize for persistence."""
        return {
            "canaries": [
                {
                    "id": c.id,
                    "amount": c.amount,
                    "type": c.type.value,
                    "planted_at": c.planted_at,
                    "triggered": c.triggered,
                    "triggered_at": c.triggered_at,
                    "triggered_by": c.triggered_by,
                }
                for c in self._canaries.values()
            ],
            "alerts": [
                {
                    "canary": {
                        "id": a.canary.id,
                        "amount": a.canary.amount,
                        "type": a.canary.type.value if isinstance(a.canary.type, CanaryType) else a.canary.type,
                        "planted_at": a.canary.planted_at,
                        "triggered": a.canary.triggered,
                    },
                    "agent_id": a.agent_id,
                    "severity": a.severity,
                    "message": a.message,
                    "timestamp": a.timestamp,
                }
                for a in self._alerts
            ],
        }

    @classmethod
    def deserialize(
        cls,
        data: Dict[str, Any],
        max_canaries: int = 5,
    ) -> CanarySystem:
        """Deserialize with validation."""
        system = cls(max_canaries)
        if isinstance(data.get("canaries"), list):
            for c in data["canaries"]:
                if c.get("id") and isinstance(c.get("amount"), (int, float)) and math.isfinite(c["amount"]):
                    system._canaries[c["id"]] = CanaryTransaction(
                        id=c["id"],
                        amount=c["amount"],
                        type=CanaryType(c.get("type", "transaction")),
                        planted_at=c.get("planted_at", 0),
                        triggered=c.get("triggered", False),
                        triggered_at=c.get("triggered_at"),
                        triggered_by=c.get("triggered_by"),
                    )
        if isinstance(data.get("alerts"), list):
            system._alerts = data["alerts"][-cls.MAX_ALERTS:]  # type: ignore
        return system
