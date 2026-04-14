"""
Shared types and dataclasses for the MnemoPay Python SDK.

All types mirror the TypeScript SDK exactly — same field names, same semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

# ── Enums ──────────────────────────────────────────────────────────────────


class TransactionStatus(str, Enum):
    PENDING = "pending"
    COMPLETED = "completed"
    REFUNDED = "refunded"
    DISPUTED = "disputed"
    EXPIRED = "expired"


class ReputationTier(str, Enum):
    UNTRUSTED = "untrusted"
    NEWCOMER = "newcomer"
    ESTABLISHED = "established"
    TRUSTED = "trusted"
    EXEMPLARY = "exemplary"


class FICORating(str, Enum):
    EXCEPTIONAL = "exceptional"
    VERY_GOOD = "very_good"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"


class TrustLevel(str, Enum):
    FULL = "full"
    HIGH = "high"
    STANDARD = "standard"
    REDUCED = "reduced"
    MINIMAL = "minimal"


class AlertSeverity(str, Enum):
    NONE = "none"
    WARNING = "warning"
    CRITICAL = "critical"


class HijackSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    EXTREME = "extreme"


class HerdSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CanaryType(str, Enum):
    TRANSACTION = "transaction"
    MEMORY = "memory"
    CREDENTIAL = "credential"


# ── Core Dataclasses ───────────────────────────────────────────────────────


@dataclass
class Memory:
    id: str
    agent_id: str
    content: str
    importance: float
    score: float
    created_at: datetime
    last_accessed: datetime
    access_count: int
    tags: List[str] = field(default_factory=list)


@dataclass
class Transaction:
    id: str
    agent_id: str
    amount: float
    reason: str
    status: TransactionStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    platform_fee: Optional[float] = None
    net_amount: Optional[float] = None
    risk_score: Optional[float] = None
    counterparty_id: Optional[str] = None


@dataclass
class BalanceInfo:
    wallet: float
    reputation: float


@dataclass
class AgentProfile:
    id: str
    reputation: float
    wallet: float
    memories_count: int
    transactions_count: int


@dataclass
class Dispute:
    id: str
    tx_id: str
    agent_id: str
    reason: str
    status: Literal["open", "resolved"]
    outcome: Optional[Literal["refund", "uphold"]] = None
    filed_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None


@dataclass
class AuditEntry:
    id: str
    agent_id: str
    action: str
    details: Dict[str, Any]
    created_at: datetime


@dataclass
class ReputationReport:
    agent_id: str
    score: float
    tier: ReputationTier
    settled_count: int
    refund_count: int
    settlement_rate: float
    total_value_settled: float
    memories_count: int
    avg_memory_importance: float
    age_hours: float
    generated_at: datetime


# ── FICO Types ─────────────────────────────────────────────────────────────


@dataclass
class FICOTransaction:
    id: str
    amount: float
    status: TransactionStatus
    created_at: datetime
    reason: str
    completed_at: Optional[datetime] = None
    counterparty_id: Optional[str] = None
    risk_score: Optional[float] = None


@dataclass
class FICOInput:
    transactions: List[FICOTransaction]
    created_at: datetime
    fraud_flags: int
    dispute_count: int
    disputes_lost: int
    warnings: int
    budget_cap: float = 5000.0
    budget_period_days: int = 30
    memories_count: int = 0


@dataclass
class FICOComponent:
    score: float
    weight: float
    weighted: float
    factors: List[str]


@dataclass
class FICOResult:
    score: int
    rating: FICORating
    trust_level: TrustLevel
    fee_rate: float
    requires_hitl: bool
    components: Dict[str, FICOComponent]
    transaction_count: int
    stable: bool
    confidence: float
    generated_at: str


@dataclass
class FICOConfig:
    w1: float = 0.35
    w2: float = 0.20
    w3: float = 0.15
    w4: float = 0.15
    w5: float = 0.15
    min_stable_transactions: int = 50
    full_history_days: int = 365
    recency_half_life_days: int = 90
    max_expected_counterparties: int = 20
    max_expected_categories: int = 10


# ── Behavioral Types ───────────────────────────────────────────────────────


@dataclass
class ProspectValue:
    value: float
    amount: float
    domain: Literal["gain", "loss"]
    loss_aversion: float
    explanation: str


@dataclass
class CoolingOffResult:
    recommended: bool
    hours: float
    reason: str
    risk_level: RiskLevel
    regret_probability: float


@dataclass
class CommitmentResult:
    projected_rates: List[float]
    projected_savings: List[float]
    final_rate: float
    explanation: str


@dataclass
class LossFrame:
    message: str
    gain_message: str
    effectiveness_multiplier: float
    goal_delay_days: int


@dataclass
class ReframedExpense:
    original: Dict[str, Any]
    daily: float
    weekly: float
    monthly: float
    annual: float
    impact_frame: str
    opportunity_cost: str


@dataclass
class RegretEntry:
    amount: float
    category: str
    regret_score: float
    timestamp: str = ""


@dataclass
class RegretPrediction:
    probability: float
    confidence: float
    historical_regret_rate: float
    category_regret_rate: float
    recommendation: str
    trigger_cooling_off: bool


@dataclass
class TradeEntry:
    timestamp: float
    amount: float
    direction: Literal["buy", "sell"]
    asset: str
    realized_pl: Optional[float] = None


@dataclass
class OverconfidenceResult:
    detected: bool
    frequency: int
    optimal_frequency: int
    performance_drag: float
    disposition_effect: bool
    recommendation: str


@dataclass
class AssetMetrics:
    pe_ratio: float
    historical_mean_pe: float
    historical_std_pe: float
    recent_return_30d: float
    volume_ratio: float


@dataclass
class HerdAlert:
    detected: bool
    severity: HerdSeverity
    valuation_sigma: float
    message: str
    recommendation: str


@dataclass
class FinancialGoal:
    name: str
    target: float
    current: float
    monthly_savings: float


@dataclass
class EndowedProgress:
    percent_complete: float
    message: str
    expected_completion_rate: float


@dataclass
class BehavioralConfig:
    lambda_: float = 2.25
    alpha: float = 0.88
    beta_pt: float = 0.88
    beta_discount: float = 0.70
    delta: float = 0.96
    base_cooling_hours: float = 2.0
    cooling_threshold: float = 100.0
    max_trades_per_month: int = 15
    excess_trade_cost_bps: float = 50.0


# ── Anomaly Types ──────────────────────────────────────────────────────────


@dataclass
class EWMAState:
    mean: float
    variance: float
    std_dev: float
    count: int
    last_value: float
    warmed_up: bool


@dataclass
class EWMAAlert:
    anomaly: bool
    value: float
    mean: float
    std_dev: float
    z_score: float
    severity: AlertSeverity
    timestamp: float


@dataclass
class BehaviorFingerprint:
    agent_id: str
    features: Dict[str, EWMAState]
    observations: int
    established: bool
    last_observed: float


@dataclass
class HijackDetection:
    suspected: bool
    anomaly_score: float
    anomalous_features: int
    total_features: int
    details: Dict[str, Dict[str, Any]]
    severity: HijackSeverity
    recommendation: str


@dataclass
class CanaryTransaction:
    id: str
    amount: float
    type: CanaryType
    planted_at: float
    triggered: bool
    triggered_at: Optional[float] = None
    triggered_by: Optional[str] = None


@dataclass
class CanaryAlert:
    canary: CanaryTransaction
    agent_id: str
    severity: Literal["critical"]
    message: str
    timestamp: float


@dataclass
class AnomalyConfig:
    alpha: float = 0.15
    warning_threshold: float = 2.5
    critical_threshold: float = 3.5
    warmup_period: int = 10
    tracked_features: List[str] = field(default_factory=lambda: [
        "amount", "hourOfDay", "timeBetweenTx", "chargesPerHour",
        "avgAmount", "amountVariance", "counterpartyCount", "memoryOpsPerTx",
    ])
    hijack_feature_threshold: float = 0.4
    max_canaries: int = 5


# ── Integrity Types ────────────────────────────────────────────────────────


@dataclass
class MerkleLeaf:
    hash: str
    memory_id: str
    index: int


@dataclass
class MerkleProofStep:
    hash: str
    direction: Literal["left", "right"]


@dataclass
class MerkleProof:
    leaf_hash: str
    memory_id: str
    path: List[MerkleProofStep]
    root_hash: str


@dataclass
class IntegritySnapshot:
    root_hash: str
    leaf_count: int
    timestamp: str
    snapshot_hash: str


@dataclass
class TamperResult:
    tampered: bool
    failed_memories: List[str]
    current_root: str
    expected_root: str
    summary: str
