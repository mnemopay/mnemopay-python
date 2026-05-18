# Changelog

## 1.0.1 — 2026-05-16 — Trademark scrub: FICO → Agent Credit Score

Removes "Agent FICO" branding from the canonical Python surface to remove any implied association with Fair Isaac Corporation. FICO is a registered trademark of Fair Isaac Corporation; this SDK is not affiliated with or endorsed by them. The 300–850 range, five-component methodology, weights, and tier thresholds are unchanged.

Added (new public surface):
- `mnemopay.agent_credit_score` module — canonical import path.
- `mnemopay.AgentCreditScore` class — canonical class name. Subclasses `AgentFICO` so the implementation and output are identical; exists to provide the canonical class name (instantiation does not emit a deprecation warning, unlike the parent).
- Canonical type aliases on the top-level `mnemopay` package: `AgentCreditScoreConfig`, `AgentCreditScoreInput`, `AgentCreditScoreTransaction`, `AgentCreditScoreResult`, `AgentCreditScoreComponent`, `AgentCreditRating`. Each is an alias of the underlying `FICO*` dataclass / enum so `isinstance` checks across both names work.

Deprecated (kept; removal v2.0.0):
- `mnemopay.fico` module — still imports cleanly.
- `mnemopay.AgentFICO` class — instantiation now emits a one-process `DeprecationWarning` (silent by default under CPython; visible under `-W default`, pytest, and most logging setups).
- `FICOConfig`, `FICOInput`, `FICOTransaction`, `FICOResult`, `FICOComponent`, `FICORating` type names — still exported.

Other changes:
- `pyproject.toml` keyword `fico` → `agent-credit-score`.
- `README.md` — updated module table; added a short "Trademark notice" section.
- `mnemopay/fico.py` docstring rewritten with deprecation + trademark notice.
- `__version__` bumped to `1.0.1` (was stale at `1.0.0b4` in `__init__.py` while `pyproject.toml` already said `1.0.0`).

Migration:
- Replace `from mnemopay import AgentFICO` with `from mnemopay import AgentCreditScore`.
- Replace `from mnemopay.fico import AgentFICO` with `from mnemopay.agent_credit_score import AgentCreditScore`.
- Replace `FICOConfig` / `FICOInput` / `FICOResult` etc. with their `AgentCreditScore*` aliases. No behavior change.
- The `NOTICE` file already documented this rename ahead of code; the code now matches.

Tests: 422/422 pass. The one `DeprecationWarning` you'll see during `pytest` is the legacy-name warning firing once in `tests/test_fico.py` — expected and confirms the warning path works.
