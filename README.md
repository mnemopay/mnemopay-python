# MnemoPay Python SDK

Memory + Wallet for AI Agents. Give any agent persistent memory and micropayment capabilities in 5 lines.

```python
from mnemopay import MnemoPay

agent = MnemoPay("my-agent")
agent.remember("user prefers Python")
tx = agent.charge(10.00, "API call")
agent.settle(tx.id)
```

## Install

```bash
pip install mnemopay
```

## Features

- **Cognitive Memory**: Ebbinghaus decay, Hebbian reinforcement, auto-scoring
- **Micropayments**: Escrow-based charges with volume-tiered fees (1.9% / 1.5% / 1.0%)
- **Agent FICO**: Credit scoring (300-850) adapted for AI agents
- **Behavioral Finance**: Prospect theory, cooling-off periods, regret prediction
- **Merkle Integrity**: SHA-256 tamper detection for agent memory
- **Anomaly Detection**: EWMA streaming detector, behavioral fingerprinting, canary honeypots
- **Zero Dependencies**: stdlib only (hashlib, math, uuid, datetime, json, dataclasses)
- **Python 3.9+**: Full type hints, dataclasses, modern Python idioms

## Modules

| Module | Class | Description |
|--------|-------|-------------|
| `mnemopay.core` | `MnemoPay` | Memory + payments + reputation |
| `mnemopay.fico` | `AgentFICO` | Credit scoring (300-850) |
| `mnemopay.behavioral` | `BehavioralEngine` | Behavioral finance tools |
| `mnemopay.integrity` | `MerkleTree` | SHA-256 tamper detection |
| `mnemopay.anomaly` | `EWMADetector`, `BehaviorMonitor`, `CanarySystem` | Anomaly detection |

## License

MIT
