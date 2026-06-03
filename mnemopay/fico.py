"""
Agent Credit Score -- FICO-style behavioral credit scoring for AI agents.

**Deprecated module name.** This module is preserved for backwards
compatibility under the legacy `mnemopay.fico` import path. New code
should import from `mnemopay.agent_reputation_scoring` (or the top-level
`mnemopay` package), which re-exports the same engine under canonical
names:

  AgentFICO              -> AgentReputationScoring
  FICOConfig             -> AgentReputationConfig
  FICOInput              -> AgentReputationInput
  FICOResult             -> AgentReputationResult
  FICOComponent          -> AgentReputationComponent
  FICOTransaction        -> AgentReputationTransaction
  FICORating             -> AgentReputationRating

The legacy names continue to work and produce identical output.
Removal is targeted for v2.0.0.
"""

from __future__ import annotations

import warnings
from typing import Optional

from .agent_reputation_scoring import AgentReputationScoring, _clamp, _extract_category
from .types import (
    FICOComponent,
    FICOConfig,
    FICOInput,
    FICORating,
    FICOResult,
    FICOTransaction,
    TransactionStatus,
    TrustLevel,
)

_warned_agent_fico_deprecated = False


def _maybe_warn_agent_fico_deprecated() -> None:
    """Emit a one-process DeprecationWarning the first time AgentFICO is instantiated."""
    global _warned_agent_fico_deprecated
    if _warned_agent_fico_deprecated:
        return
    _warned_agent_fico_deprecated = True
    warnings.warn(
        "mnemopay.fico.AgentFICO is deprecated; use "
        "mnemopay.AgentReputationScoring (alias of the same engine) instead. "
        "The legacy name will be removed in v2.0.0.",
        DeprecationWarning,
        stacklevel=3,
    )


class AgentFICO(AgentReputationScoring):
    """FICO-style agent credit scoring engine.

    .. deprecated::
        Use :class:`mnemopay.AgentReputationScoring` instead. This class name will be
        removed in v2.0.0.
    """

    def __init__(self, config: Optional[FICOConfig] = None) -> None:
        _maybe_warn_agent_fico_deprecated()
        super().__init__(config)
