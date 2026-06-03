"""
Agent Credit Score -- FICO-style behavioral credit scoring for AI agents.

**Deprecated module name.** This module is preserved for backwards
compatibility under the legacy `mnemopay.agent_credit_score` import path.
New code should import from `mnemopay.agent_reputation_scoring` (or the top-level
`mnemopay` package), which re-exports the same engine.
"""

from __future__ import annotations

import warnings
from typing import Optional

from .agent_reputation_scoring import AgentReputationScoring
from .types import (
    AgentCreditScoreComponent,
    AgentCreditScoreConfig,
    AgentCreditScoreInput,
    AgentCreditRating,
    AgentCreditScoreResult,
    AgentCreditScoreTransaction,
)

_warned_agent_credit_score_deprecated = False


def _maybe_warn_agent_credit_score_deprecated() -> None:
    """Emit a one-process DeprecationWarning the first time AgentCreditScore is instantiated."""
    global _warned_agent_credit_score_deprecated
    if _warned_agent_credit_score_deprecated:
        return
    _warned_agent_credit_score_deprecated = True
    warnings.warn(
        "mnemopay.agent_credit_score.AgentCreditScore is deprecated; use "
        "mnemopay.AgentReputationScoring (alias of the same engine) instead. "
        "The legacy name will be removed in v2.0.0.",
        DeprecationWarning,
        stacklevel=3,
    )


class AgentCreditScore(AgentReputationScoring):
    """FICO-style agent credit scoring engine (300-850).

    .. deprecated::
        Use :class:`mnemopay.AgentReputationScoring` instead. This class name will be
        removed in v2.0.0.
    """

    def __init__(self, config: Optional[AgentCreditScoreConfig] = None) -> None:
        _maybe_warn_agent_credit_score_deprecated()
        super().__init__(config)


__all__ = [
    "AgentCreditScore",
    "AgentCreditScoreComponent",
    "AgentCreditScoreConfig",
    "AgentCreditScoreInput",
    "AgentCreditRating",
    "AgentCreditScoreResult",
    "AgentCreditScoreTransaction",
]
