# MnemoPay Python SDK

The governance layer for AI agents that handle money — Python edition. Mirrors the TypeScript [`@mnemopay/sdk`](https://www.npmjs.com/package/@mnemopay/sdk) shape so the same agent identity, score, charter, and audit chain work across both runtimes.

```python
from mnemopay import MnemoPay

agent = MnemoPay("my-agent")
agent.remember("user prefers Python")
tx = agent.charge(10.00, "API call")
# Settle after the escrow hold window expires.
agent.settle(tx.id)
```

## Install

```bash
pip install mnemopay                  # stdlib-only core
pip install "mnemopay[stripe]"        # + StripeRail (peer-loaded `stripe>=12.0`)
```

> **What MnemoPay is NOT:** not a bank, not a money transmitter, not a Stripe replacement, not an agent framework. It sits **above** the rail and **below** the runtime — declares the rules, enforces the budget, produces the evidence.

## Features

- **Payment rails** (TS parity): `MockRail`, `StripeRail`, `PaystackRail`, `LightningRail` — manual-capture two-phase commit, threading-safe capture race-protection, idempotency-key forwarding
- **Cognitive Memory**: Ebbinghaus decay, Hebbian reinforcement, auto-scoring, Layer 2 semantic recall + RL feedback
- **Micropayments**: escrow-based charges with volume-tiered fees (1.9% / 1.5% / 1.0%)
- **Agent Credit Score**: FICO-style agent behavioral scoring (300-850 range; not consumer FICO, not FCRA-regulated, not affiliated with Fair Isaac Corporation). Importable as `AgentCreditScore` (preferred) or the deprecated `AgentFICO` alias.
- **Behavioral Finance**: prospect theory, cooling-off periods, regret prediction
- **Merkle Integrity**: SHA-256 tamper detection for agent memory
- **Anomaly Detection**: EWMA streaming detector, behavioral fingerprinting, canary honeypots
- **Commerce**: autonomous shopping engine with mandates + escrow
- **Circuit Breaker**: AIMD rate limiting, anti-gaming, PSI drift detection
- **Stdlib core**: zero required dependencies — peer-deps only when you opt in (e.g. `stripe`)
- **Python 3.9+**: full type hints, dataclasses, sync API (matches the rest of the SDK)

## Modules

| Module | Class | Description |
|--------|-------|-------------|
| `mnemopay.core` | `MnemoPay` | Memory + payments + reputation |
| `mnemopay.rails` | `PaymentRail`, `MockRail`, `StripeRail`, `PaystackRail`, `LightningRail` | Payment rail abstraction |
| `mnemopay.agent_credit_score` | `AgentCreditScore` | Credit scoring (300-850, FICO-style; not affiliated with Fair Isaac Corporation). The `mnemopay.fico` module + `AgentFICO` class remain as deprecated aliases. |
| `mnemopay.behavioral` | `BehavioralEngine` | Behavioral finance tools |
| `mnemopay.integrity` | `MerkleTree` | SHA-256 tamper detection |
| `mnemopay.anomaly` | `EWMADetector`, `BehaviorMonitor`, `CanarySystem` | Anomaly detection |
| `mnemopay.commerce` | `CommerceEngine`, `CommerceProvider`, `Mandate` | Autonomous shopping |
| `mnemopay.circuit_breaker` | `CircuitBreaker`, `AIMDRateLimiter`, `AntiGamingEngine`, `PSIDriftDetector` | Adaptive defense |

## Payment rails

Mirrors the TypeScript `PaymentRail` interface. Same shape (`create_hold` / `capture_payment` / `reverse_payment`), same drop-in-swap semantics. Sync API to match the rest of the Python SDK — no asyncio.

```python
from mnemopay.rails import StripeRail, MockRail, HoldOptions

# Default — no infra, in-memory ledger, used in tests + dev
rail = MockRail()

# Production — real Stripe PaymentIntents with manual-capture two-phase commit
rail = StripeRail(secret_key="sk_test_...")

# Two-phase commit (hold → capture)
hold = rail.create_hold(
    amount=25.00,
    reason="Monthly access",
    agent_id="agent-1",
    opts=HoldOptions(
        customer_id="cus_real",
        payment_method_id="pm_real",
        off_session=True,
        metadata={"idempotencyKey": "req_abc"},
    ),
)
capture = rail.capture_payment(hold.external_id, 25.00)
# capture.status == "succeeded"
# capture.receipt_id == "ch_..."  # Stripe charge id

# Or reverse (cancel the hold) instead of capturing
rail.reverse_payment(hold.external_id, 25.00)
```

`StripeRail` includes onboarding helpers for off-session charges:

```python
result = rail.create_customer("user@example.com", name="Ada O")
# {"customer_id": "cus_..."}

result = rail.create_setup_intent(customer_id="cus_...")
# {"setup_intent_id": "seti_...", "client_secret": "..."}
```

For tests, inject a mock client:

```python
from unittest.mock import MagicMock
rail = StripeRail.from_client(MagicMock(), currency="usd")
```

## Compatibility with `@mnemopay/sdk` (TypeScript)

| Feature | TypeScript v1.14.1 | Python v1.2.2 |
|---|---|---|
| `MockRail` | yes | yes |
| `StripeRail` | yes | yes |
| `PaystackRail` | yes | yes |
| `LightningRail` | yes | yes |
| `StripeMPPRail` (alpha) | yes | not yet |
| `X402Rail` (alpha) | yes | not yet |
| `GoogleAP2Rail` (alpha) | yes | not yet |
| Governance (charter / FiscalGate / Article 12 / risk taxonomy / action ledger / governed skills) | yes | not yet |
| Agent Credit Score | yes | yes |
| Behavioral Finance | yes | yes |
| Merkle integrity | yes | yes |
| Anomaly detection (EWMA + canary) | yes | yes |
| Commerce engine | yes | yes |
| Circuit breaker | yes | yes |

The Python SDK ships behind the TypeScript SDK — port priorities track agent-developer demand. Open an issue if you need a specific rail or governance primitive in Python.

## Tests

```bash
pip install -e ".[dev,stripe]"
pytest                    # 456 tests across 10 modules
pytest tests/test_rails.py -v
```

## License

**v1.0.0+ is Apache License 2.0** — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

The pre-release betas (`1.0.0b1` through `1.0.0b4`) shipped under the MIT
License and remain available under those terms on PyPI in perpetuity. If
you depend on the MIT terms specifically, pin to `mnemopay==1.0.0b4`.

Why Apache 2.0 at the stable cut: alignment with the canonical TypeScript
SDK at [`@mnemopay/sdk`](https://www.npmjs.com/package/@mnemopay/sdk) plus
the patent grant + retaliation clause for enterprise embedders.

Copyright 2026 J&B Enterprise LLC.

## Trademark notice

FICO is a registered trademark of Fair Isaac Corporation. MnemoPay's Agent Credit Score is FICO-style behavioral scoring (300-850 range, five-component methodology) and is not affiliated with or endorsed by Fair Isaac Corporation. Consumer FICO scores are regulated under the FCRA and produced by Fair Isaac Corporation; this SDK produces neither.
