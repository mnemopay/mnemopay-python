"""
Agent Credit Score -- FICO-style behavioral credit scoring for AI agents.

This is the canonical module name for MnemoPay's behavioral credit-scoring
engine. The legacy `mnemopay.fico` module + `AgentFICO` class remain as
deprecated aliases for back-compat and will be removed in v2.0.0.

Trademark notice
----------------
FICO is a registered trademark of Fair Isaac Corporation. MnemoPay's
Agent Credit Score is not affiliated with or endorsed by Fair Isaac
Corporation. The 300-850 range and five-component methodology are used
in the agent-credit-score sense; consumer FICO scores are regulated under
the FCRA and produced by Fair Isaac Corporation.

The five components, weights, and tier thresholds are unchanged from
`mnemopay.fico`:

  1. Payment History    (35%)
  2. Credit Utilization (20%)
  3. History Length      (15%)
  4. Behavior Diversity  (15%)
  5. Fraud Record        (15%)

Score is deterministic given inputs. No randomness, no hidden state.

Usage
-----
    from mnemopay import AgentCreditScore, AgentCreditScoreInput

    engine = AgentCreditScore()
    result = engine.compute(my_input)
    print(result.score, result.rating)
"""

from __future__ import annotations

from typing import Optional

from .fico import AgentFICO as _AgentFICO
from .types import (
    FICOComponent as AgentCreditScoreComponent,
    FICOConfig as AgentCreditScoreConfig,
    FICOInput as AgentCreditScoreInput,
    FICORating as AgentCreditRating,
    FICOResult as AgentCreditScoreResult,
    FICOTransaction as AgentCreditScoreTransaction,
)


class AgentCreditScore(_AgentFICO):
    """FICO-style agent credit scoring engine (300-850).

    Canonical class for MnemoPay's behavioral credit-scoring engine.
    Trademark notice: FICO is a registered trademark of Fair Isaac
    Corporation. MnemoPay's Agent Credit Score is not affiliated with
    or endorsed by Fair Isaac Corporation.

    Subclasses :class:`mnemopay.fico.AgentFICO` so the implementation and
    output are identical; this subclass exists solely to provide the
    canonical class name (the parent emits a DeprecationWarning on
    instantiation, this subclass does not).
    """

    def __init__(self, config: Optional[AgentCreditScoreConfig] = None) -> None:
        # Set the attribute directly to bypass the deprecation warning the
        # parent class emits on __init__; we intentionally duplicate the
        # validation logic by calling object.__init__ then re-running the
        # parent's validation manually.
        self.config = config or AgentCreditScoreConfig()
        weight_sum = (
            self.config.w1 + self.config.w2 + self.config.w3
            + self.config.w4 + self.config.w5
        )
        if abs(weight_sum - 1.0) > 0.001:
            raise ValueError(
                f"AgentCreditScore weights must sum to 1.0, got {weight_sum:.4f}"
            )
        if any(w <= 0 for w in [
            self.config.w1, self.config.w2, self.config.w3,
            self.config.w4, self.config.w5,
        ]):
            raise ValueError("All AgentCreditScore weights must be positive")


__all__ = [
    "AgentCreditScore",
    "AgentCreditScoreComponent",
    "AgentCreditScoreConfig",
    "AgentCreditScoreInput",
    "AgentCreditRating",
    "AgentCreditScoreResult",
    "AgentCreditScoreTransaction",
]
