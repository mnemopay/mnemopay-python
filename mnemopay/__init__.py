"""
MnemoPay Python SDK -- Memory + Wallet for AI Agents

Give any AI agent memory and a wallet in 5 lines:

    from mnemopay import MnemoPay

    agent = MnemoPay("my-agent")
    agent.remember("user prefers Python")
    tx = agent.charge(10.00, "API call")
    agent.settle(tx.id)
"""

from .anomaly import BehaviorMonitor, CanarySystem, EWMADetector
from .behavioral import BehavioralEngine
from .circuit_breaker import (
    AIMDConfig,
    AIMDRateLimiter,
    AntiGamingAlert,
    AntiGamingEngine,
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitState,
    PSIDriftDetector,
    PSIDriftResult,
)
from .commerce import (
    CommerceEngine,
    CommerceProvider,
    Mandate,
    MockProvider,
    Order,
    OrderStatus,
    Product,
)
from .agent_credit_score import (
    AgentCreditRating,
    AgentCreditScore,
    AgentCreditScoreComponent,
    AgentCreditScoreConfig,
    AgentCreditScoreInput,
    AgentCreditScoreResult,
    AgentCreditScoreTransaction,
)
from .core import MnemoPay, auto_score, compute_score
from .fico import AgentFICO
from .integrity import MerkleTree
from .rails import (
    HoldOptions,
    MockRail,
    PaymentRail,
    PaymentRailResult,
    StripeRail,
    PaystackRail,
    LightningRail,
)
from .types import (
    AgentProfile,
    AlertSeverity,
    # Anomaly types
    AnomalyConfig,
    AssetMetrics,
    AuditEntry,
    BalanceInfo,
    # Behavioral types
    BehavioralConfig,
    BehaviorFingerprint,
    CanaryAlert,
    CanaryTransaction,
    CanaryType,
    CommitmentResult,
    CoolingOffResult,
    Dispute,
    EndowedProgress,
    EWMAAlert,
    EWMAState,
    FICOComponent,
    # FICO types
    FICOConfig,
    FICOInput,
    FICORating,
    FICOResult,
    FICOTransaction,
    FinancialGoal,
    HerdAlert,
    HerdSeverity,
    HijackDetection,
    HijackSeverity,
    IntegritySnapshot,
    LossFrame,
    # Core types
    Memory,
    # Integrity types
    MerkleLeaf,
    MerkleProof,
    MerkleProofStep,
    OverconfidenceResult,
    ProspectValue,
    ReframedExpense,
    RegretEntry,
    RegretPrediction,
    ReputationReport,
    ReputationTier,
    RiskLevel,
    TamperResult,
    TradeEntry,
    Transaction,
    TransactionStatus,
    TrustLevel,
)

__version__ = "1.0.1"
__all__ = [
    # Core
    "MnemoPay",
    "auto_score",
    "compute_score",
    # Modules
    "AgentCreditScore",
    "AgentFICO",  # deprecated alias of AgentCreditScore; removal v2.0.0
    "BehavioralEngine",
    "MerkleTree",
    "EWMADetector",
    "BehaviorMonitor",
    "CanarySystem",
    # Rails (v1.0.0b4 — parity with TS @mnemopay/sdk v1.6.x)
    "PaymentRail",
    "PaymentRailResult",
    "HoldOptions",
    "MockRail",
    "StripeRail",
    "PaystackRail",
    "LightningRail",
    # Commerce
    "CommerceEngine",
    "CommerceProvider",
    "MockProvider",
    "Product",
    "Order",
    "OrderStatus",
    "Mandate",
    # Circuit Breaker & Rate Limiting
    "CircuitBreaker",
    "CircuitBreakerConfig",
    "CircuitState",
    "AIMDRateLimiter",
    "AIMDConfig",
    "AntiGamingEngine",
    "AntiGamingAlert",
    "PSIDriftDetector",
    "PSIDriftResult",
    # Types
    "Memory",
    "Transaction",
    "TransactionStatus",
    "BalanceInfo",
    "AgentProfile",
    "Dispute",
    "AuditEntry",
    "ReputationReport",
    "ReputationTier",
    # Agent Credit Score types (canonical)
    "AgentCreditScoreConfig",
    "AgentCreditScoreInput",
    "AgentCreditScoreTransaction",
    "AgentCreditScoreResult",
    "AgentCreditScoreComponent",
    "AgentCreditRating",
    # FICO* type aliases — deprecated, removed v2.0.0
    "FICOConfig",
    "FICOInput",
    "FICOTransaction",
    "FICOResult",
    "FICOComponent",
    "FICORating",
    "TrustLevel",
    "BehavioralConfig",
    "ProspectValue",
    "CoolingOffResult",
    "CommitmentResult",
    "LossFrame",
    "ReframedExpense",
    "RegretEntry",
    "RegretPrediction",
    "TradeEntry",
    "OverconfidenceResult",
    "AssetMetrics",
    "HerdAlert",
    "FinancialGoal",
    "EndowedProgress",
    "RiskLevel",
    "HerdSeverity",
    "MerkleLeaf",
    "MerkleProof",
    "MerkleProofStep",
    "IntegritySnapshot",
    "TamperResult",
    "AnomalyConfig",
    "EWMAState",
    "EWMAAlert",
    "AlertSeverity",
    "BehaviorFingerprint",
    "HijackDetection",
    "HijackSeverity",
    "CanaryTransaction",
    "CanaryAlert",
    "CanaryType",
]
