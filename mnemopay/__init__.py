"""
MnemoPay Python SDK -- Memory + Wallet for AI Agents

Give any AI agent memory and a wallet in 5 lines:

    from mnemopay import MnemoPay

    agent = MnemoPay("my-agent")
    agent.remember("user prefers Python")
    tx = agent.charge(10.00, "API call")
    agent.settle(tx.id)
"""

from .core import MnemoPay, auto_score, compute_score
from .fico import AgentFICO
from .behavioral import BehavioralEngine
from .integrity import MerkleTree
from .anomaly import EWMADetector, BehaviorMonitor, CanarySystem
from .types import (
    # Core types
    Memory,
    Transaction,
    TransactionStatus,
    BalanceInfo,
    AgentProfile,
    Dispute,
    AuditEntry,
    ReputationReport,
    ReputationTier,
    # FICO types
    FICOConfig,
    FICOInput,
    FICOTransaction,
    FICOResult,
    FICOComponent,
    FICORating,
    TrustLevel,
    # Behavioral types
    BehavioralConfig,
    ProspectValue,
    CoolingOffResult,
    CommitmentResult,
    LossFrame,
    ReframedExpense,
    RegretEntry,
    RegretPrediction,
    TradeEntry,
    OverconfidenceResult,
    AssetMetrics,
    HerdAlert,
    FinancialGoal,
    EndowedProgress,
    RiskLevel,
    HerdSeverity,
    # Integrity types
    MerkleLeaf,
    MerkleProof,
    MerkleProofStep,
    IntegritySnapshot,
    TamperResult,
    # Anomaly types
    AnomalyConfig,
    EWMAState,
    EWMAAlert,
    AlertSeverity,
    BehaviorFingerprint,
    HijackDetection,
    HijackSeverity,
    CanaryTransaction,
    CanaryAlert,
    CanaryType,
)

__version__ = "1.0.0b1"
__all__ = [
    # Core
    "MnemoPay",
    "auto_score",
    "compute_score",
    # Modules
    "AgentFICO",
    "BehavioralEngine",
    "MerkleTree",
    "EWMADetector",
    "BehaviorMonitor",
    "CanarySystem",
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
