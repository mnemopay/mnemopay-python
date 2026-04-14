"""
MnemoPay Core -- Memory + Wallet for AI Agents

Port of the TypeScript MnemoPayLite class. Provides:
  - Cognitive memory with Ebbinghaus decay
  - Micropayment escrow with double-entry ledger
  - Hebbian reinforcement (settled payments boost recent memories)
  - Volume-tiered platform fees
  - Reputation tracking

Zero external dependencies (stdlib only).
"""

from __future__ import annotations

import math
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from .types import (
    AgentProfile,
    AuditEntry,
    BalanceInfo,
    Dispute,
    Memory,
    ReputationReport,
    ReputationTier,
    Transaction,
    TransactionStatus,
)

# ── Constants ──────────────────────────────────────────────────────────────

MAX_WALLET_BALANCE = 1_000_000
MAX_MEMORIES = 50_000
MAX_TRANSACTIONS = 100_000
BASE_IMPORTANCE = 0.50
LONG_CONTENT_THRESHOLD = 200
LONG_CONTENT_BOOST = 0.10

IMPORTANCE_PATTERNS: List[Dict[str, Any]] = [
    {"pattern": re.compile(r"\b(error|fail|crash|critical|broken|bug)\b", re.I), "boost": 0.20},
    {"pattern": re.compile(r"\b(success|complete|paid|delivered|resolved)\b", re.I), "boost": 0.15},
    {"pattern": re.compile(r"\b(prefer|always|never|important|must|require)\b", re.I), "boost": 0.15},
]

INJECTION_PATTERNS = [
    re.compile(
        r"\b(ignore|disregard|forget)\b.{0,30}\b(previous|prior|above|all)\b"
        r".{0,30}\b(instructions?|rules?|constraints?)\b", re.I,
    ),
    re.compile(
        r"\b(you are|act as|pretend|roleplay|simulate)\b"
        r".{0,30}\b(admin|root|system|god|superuser)\b", re.I,
    ),
    re.compile(r"\bsystem\s*:\s*", re.I),
    re.compile(r"\bassistant\s*:\s*", re.I),
    re.compile(
        r"\b(transfer|send|move)\b.{0,20}\b(all|every|maximum)\b"
        r".{0,20}\b(funds?|money|balance|wallet)\b", re.I,
    ),
    re.compile(
        r"\b(set|change|update|override)\b.{0,20}\b(wallet|balance|reputation|role|permission)\b"
        r".{0,10}\b(to|=)\b", re.I,
    ),
]

# ── Layer 2: Local Embedding (pure stdlib, zero deps) ─────────────────────
# Mirrors the TypeScript localEmbed() in recall/engine.ts.
# Hash-projection TF-IDF with bigrams, trigrams, and preference synonym expansion.

_STOP_WORDS = {
    'the','a','an','is','are','was','were','be','been','being','have','has','had',
    'do','does','did','will','would','could','should','may','might','shall','can',
    'to','of','in','for','on','with','at','by','from','as','into','through',
    'and','but','or','nor','not','so','yet','both','each','few','more','most',
    'other','some','such','no','only','same','than','too','very','just','about',
    'up','it','its','this','that','these','those','i','me','my','we','our',
    'you','your','he','him','his','she','her','they','them','their','what',
    'which','who','because','out','off','over','under','then','once','again',
}

_PREFERENCE_SYNONYMS: Dict[str, List[str]] = {
    'favorite': ['prefer','love','enjoy','like'],
    'prefer':   ['favorite','love','enjoy','like'],
    'love':     ['favorite','prefer','enjoy','like'],
    'enjoy':    ['favorite','prefer','love','like'],
    'like':     ['favorite','prefer','love','enjoy'],
    'hate':     ['dislike','avoid','detest'],
    'dislike':  ['hate','avoid'],
    'allergic': ['allergy','intolerant','avoid','restrict'],
    'restriction': ['restrict','allergic','avoid','diet'],
}

_EMBED_DIM = 384


def _fnv1a(s: str) -> int:
    h = 2166136261
    for c in s.encode('utf-8'):
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def _tokenize_embed(text: str) -> List[str]:
    cleaned = re.sub(r'[^a-z0-9\s]', ' ', text.lower())
    return [w for w in cleaned.split() if len(w) > 1 and w not in _STOP_WORDS]


def _local_embed(text: str, dimensions: int = _EMBED_DIM) -> List[float]:
    """Convert text to a hash-projected TF-IDF vector (pure Python, zero deps)."""
    base_tokens = _tokenize_embed(text)
    expanded: set = set(base_tokens)
    for tok in base_tokens:
        for syn in _PREFERENCE_SYNONYMS.get(tok, []):
            expanded.add(syn)

    vec = [0.0] * dimensions

    def add_token(token: str, weight: float) -> None:
        h = _fnv1a(token)
        w = weight * (1 + math.log(max(len(token), 1)))
        vec[abs(h) % dimensions] += w
        vec[abs(h * 31) % dimensions] += w * 0.5
        vec[abs(h * 97) % dimensions] += w * 0.3
        if len(token) > 4:
            ph = _fnv1a(token[:4])
            vec[abs(ph) % dimensions] += w * 0.3

    for tok in expanded:
        add_token(tok, 1.0)
    for i in range(len(base_tokens) - 1):
        add_token(f'{base_tokens[i]}_{base_tokens[i+1]}', 0.7)
    for i in range(len(base_tokens) - 2):
        add_token(f'{base_tokens[i]}_{base_tokens[i+1]}_{base_tokens[i+2]}', 0.4)

    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm > 1e-10 else vec


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na * nb > 1e-10 else 0.0


# ── Volume-tiered fees ─────────────────────────────────────────────────────

_FEE_TIERS = [
    (100_000, 0.010),  # > $100K: 1.0%
    (10_000,  0.015),  # > $10K:  1.5%
    (0,       0.019),  # default: 1.9%
]


def _get_fee_rate(volume: float) -> float:
    for threshold, rate in _FEE_TIERS:
        if volume > threshold:
            return rate
    return 0.019


# ── Helpers ────────────────────────────────────────────────────────────────


def _sanitize_content(content: str) -> str:
    sanitized = content
    for pattern in INJECTION_PATTERNS:
        sanitized = pattern.sub("[FILTERED]", sanitized)
    return sanitized


def _validate_tags(tags: List[str]) -> List[str]:
    result = []
    for t in tags:
        if not isinstance(t, str) or len(t) > 50:
            continue
        cleaned = re.sub(r"[^a-zA-Z0-9_\-:.]", "", t)
        if cleaned:
            result.append(cleaned)
        if len(result) >= 20:
            break
    return result


def auto_score(content: str) -> float:
    """Compute automatic importance score from content keywords."""
    score = BASE_IMPORTANCE
    if len(content) > LONG_CONTENT_THRESHOLD:
        score += LONG_CONTENT_BOOST
    for entry in IMPORTANCE_PATTERNS:
        if entry["pattern"].search(content):
            score += entry["boost"]
    return min(score, 1.0)


def compute_score(
    importance: float,
    last_accessed: datetime,
    access_count: int,
    decay: float,
) -> float:
    """Compute composite memory score using Ebbinghaus decay.

    score = importance * e^(-decay * hours) * (1 + ln(1 + access_count))
    """
    hours_since = (
        datetime.now(timezone.utc).timestamp() - last_accessed.timestamp()
    ) / 3600
    recency = math.exp(-decay * hours_since)
    frequency = 1 + math.log(1 + access_count)
    return importance * recency * frequency


def _reputation_tier(score: float) -> ReputationTier:
    if score >= 0.9:
        return ReputationTier.EXEMPLARY
    if score >= 0.7:
        return ReputationTier.TRUSTED
    if score >= 0.4:
        return ReputationTier.ESTABLISHED
    if score >= 0.2:
        return ReputationTier.NEWCOMER
    return ReputationTier.UNTRUSTED


class MnemoPay:
    """MnemoPay core agent -- memory + wallet + reputation.

    Usage:
        agent = MnemoPay("my-agent")
        mem_id = agent.remember("user prefers Python")
        memories = agent.recall(5)
        tx = agent.charge(10.00, "API call")
        agent.settle(tx.id)
    """

    def __init__(
        self,
        agent_id: str,
        decay: float = 0.05,
        debug: bool = False,
    ) -> None:
        self.agent_id = agent_id
        self._decay = decay
        self._debug = debug
        self._memories: Dict[str, Memory] = {}
        self._embeddings: Dict[str, List[float]] = {}   # Layer 2 vector cache
        self._transactions: Dict[str, Transaction] = {}
        self._audit_log: List[AuditEntry] = []
        self._disputes: Dict[str, Dispute] = {}
        self._wallet: float = 0.0
        self._reputation: float = 0.5
        self._created_at: datetime = datetime.now(timezone.utc)
        self._total_volume: float = 0.0
        self._listeners: Dict[str, List[Callable]] = {}

        self._log("MnemoPay initialized (in-memory)")

    # ── Event System ──────────────────────────────────────────────────────

    def on(self, event: str, callback: Callable) -> None:
        """Register an event listener."""
        self._listeners.setdefault(event, []).append(callback)

    def _emit(self, event: str, data: Any = None) -> None:
        for cb in self._listeners.get(event, []):
            cb(data)

    # ── Memory Methods ────────────────────────────────────────────────────

    def remember(
        self,
        content: str,
        importance: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> str:
        """Store a memory. Returns the memory ID.

        Args:
            content: The memory content string.
            importance: Override importance score (0-1). Auto-scored if not provided.
            tags: Optional list of string tags.

        Returns:
            The UUID of the stored memory.

        Raises:
            ValueError: If content is empty or exceeds 100KB.
        """
        if not content or not isinstance(content, str):
            raise ValueError("Memory content is required")
        if len(content) > 100_000:
            raise ValueError("Memory content exceeds 100KB limit")
        if len(self._memories) >= MAX_MEMORIES:
            raise ValueError(
                f"Memory limit reached ({MAX_MEMORIES}). "
                f"Consolidate or forget old memories first."
            )

        safe_content = _sanitize_content(content)
        safe_tags = _validate_tags(tags or [])
        imp = importance if importance is not None else auto_score(safe_content)
        imp = min(max(imp, 0.0), 1.0)

        now = datetime.now(timezone.utc)
        mem = Memory(
            id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            content=safe_content,
            importance=imp,
            score=imp,
            created_at=now,
            last_accessed=now,
            access_count=0,
            tags=safe_tags,
        )
        self._memories[mem.id] = mem
        self._embeddings[mem.id] = _local_embed(safe_content)  # Layer 2: encode at write time
        self._audit("memory:stored", {"id": mem.id, "tags": safe_tags, "importance": mem.importance})
        self._emit("memory:stored", {"id": mem.id, "importance": mem.importance})
        self._log(
            f"Stored memory: id={mem.id} "
            f"(importance: {mem.importance:.2f}, tags: {','.join(safe_tags) or 'none'})"
        )
        return mem.id

    def recall(self, query_or_limit: Any = 5, limit: int = 5) -> List[Memory]:
        """Recall memories. Layer 2 semantic search when a string query is provided.

        Args:
            query_or_limit: str query for semantic recall, or int for score-only recall.
            limit: Max results when query_or_limit is a string (default 5).

        Returns:
            List of Memory objects sorted by relevance (highest first).
        """
        if isinstance(query_or_limit, str):
            query: Optional[str] = query_or_limit
            actual_limit = limit
        else:
            query = None
            actual_limit = int(query_or_limit) if query_or_limit else 5

        all_mems = list(self._memories.values())
        for m in all_mems:
            m.score = compute_score(m.importance, m.last_accessed, m.access_count, self._decay)

        if query:
            # Layer 2: hybrid score + cosine similarity
            query_vec = _local_embed(query)
            scored: List[tuple] = []
            for m in all_mems:
                vec = self._embeddings.get(m.id) or _local_embed(m.content)
                if m.id not in self._embeddings:
                    self._embeddings[m.id] = vec
                vec_sim = _cosine_similarity(query_vec, vec)
                # Normalize decay score to [0,1] range for blending
                norm_score = m.score / max(m.score, 1.0)
                combined = 0.5 * norm_score + 0.5 * vec_sim
                scored.append((combined, m))
            scored.sort(key=lambda x: x[0], reverse=True)
            results = [m for _, m in scored[:actual_limit]]
        else:
            all_mems.sort(key=lambda m: m.score, reverse=True)
            results = all_mems[:actual_limit]

        now = datetime.now(timezone.utc)
        for m in results:
            m.last_accessed = now
            m.access_count += 1

        self._emit("memory:recalled", {"count": len(results)})
        self._log(f"Recalled {len(results)} memories (semantic={query is not None})")
        return results

    def forget(self, memory_id: str) -> bool:
        """Delete a memory by ID.

        Returns:
            True if the memory existed and was deleted, False otherwise.
        """
        existed = memory_id in self._memories
        if existed:
            del self._memories[memory_id]
            self._audit("memory:deleted", {"id": memory_id})
            self._log(f"Forgot memory: {memory_id}")
        return existed

    def reinforce(self, memory_id: str, boost: float = 0.1) -> None:
        """Reinforce a memory by boosting its importance.

        Args:
            memory_id: The memory to reinforce.
            boost: Importance adjustment (-0.5 to 0.5).

        Raises:
            ValueError: If the memory doesn't exist or boost is out of range.
        """
        if not memory_id or not isinstance(memory_id, str):
            raise ValueError("Memory ID is required")
        if not isinstance(boost, (int, float)) or not math.isfinite(boost):
            raise ValueError("Boost must be a finite number")
        if boost < -0.5 or boost > 0.5:
            raise ValueError("Boost must be between -0.5 and 0.5")
        mem = self._memories.get(memory_id)
        if mem is None:
            raise ValueError(f"Memory {memory_id} not found")
        mem.importance = min(max(mem.importance + boost, 0.0), 1.0)
        mem.last_accessed = datetime.now(timezone.utc)
        self._audit("memory:reinforced", {"id": memory_id, "boost": boost, "new_importance": mem.importance})
        self._log(f"Reinforced memory {memory_id} by +{boost} -> {mem.importance:.2f}")

    def rl_feedback(self, memory_ids: List[str], reward: float, alpha: float = 0.1) -> None:
        """RL feedback loop — EWMA importance update for recalled memories.

        Call after an agent action to signal whether the recalled memories
        were useful. Updates importance via:
            new_importance = α × ((reward+1)/2) + (1−α) × old_importance

        Args:
            memory_ids: IDs from the preceding recall.
            reward:     Signal in [-1, 1]: +1=useful, 0=neutral, -1=useless.
            alpha:      Learning rate in (0.01, 0.5], default 0.1.
        """
        clamped = max(-1.0, min(1.0, float(reward)))
        normalized = (clamped + 1.0) / 2.0
        alpha = max(0.01, min(0.5, float(alpha)))
        updated = 0
        for mid in memory_ids:
            mem = self._memories.get(mid)
            if not mem:
                continue
            mem.importance = max(0.0, min(1.0,
                alpha * normalized + (1.0 - alpha) * mem.importance
            ))
            updated += 1
        self._audit("memory:rl_feedback", {"ids": memory_ids, "reward": reward, "updated": updated})
        self._log(f"rl_feedback: reward={reward:.2f}, updated {updated} memories")

    def consolidate(self) -> int:
        """Prune memories with scores below threshold (0.01).

        Returns:
            Number of memories pruned.
        """
        threshold = 0.01
        pruned = 0
        to_remove = []
        for mid, mem in self._memories.items():
            score = compute_score(mem.importance, mem.last_accessed, mem.access_count, self._decay)
            if score < threshold:
                to_remove.append(mid)
                pruned += 1
        for mid in to_remove:
            del self._memories[mid]
        self._audit("memory:consolidated", {"pruned": pruned})
        self._log(f"Consolidated: pruned {pruned} stale memories")
        return pruned

    # ── Payment Methods ───────────────────────────────────────────────────

    def charge(
        self,
        amount: float,
        reason: str,
        counterparty: Optional[str] = None,
    ) -> Transaction:
        """Create a pending payment charge.

        Args:
            amount: Payment amount (must be positive).
            reason: Description of the charge.
            counterparty: Optional counterparty agent ID.

        Returns:
            The created Transaction in "pending" status.

        Raises:
            ValueError: If amount exceeds reputation ceiling or limits are hit.
        """
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("Amount must be a positive finite number")
        amount = round(amount * 100) / 100
        if not reason or not isinstance(reason, str):
            raise ValueError("Reason is required")
        if len(reason) > 1000:
            raise ValueError("Reason exceeds 1000 character limit")
        if len(self._transactions) >= MAX_TRANSACTIONS:
            raise ValueError(
                f"Transaction limit reached ({MAX_TRANSACTIONS}). "
                f"Archive old transactions."
            )

        max_charge = 500 * self._reputation
        if amount > max_charge:
            raise ValueError(
                f"Amount ${amount:.2f} exceeds reputation ceiling "
                f"${max_charge:.2f} (reputation: {self._reputation:.2f})"
            )

        tx = Transaction(
            id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            amount=amount,
            reason=reason,
            status=TransactionStatus.PENDING,
            created_at=datetime.now(timezone.utc),
            counterparty_id=counterparty,
        )
        self._transactions[tx.id] = tx
        self._audit("payment:pending", {"id": tx.id, "amount": amount, "reason": reason})
        self._emit("payment:pending", {"id": tx.id, "amount": amount})
        self._log(f'Charge created: ${amount:.2f} for "{reason}" (pending)')
        return Transaction(
            id=tx.id, agent_id=tx.agent_id, amount=tx.amount,
            reason=tx.reason, status=tx.status, created_at=tx.created_at,
            counterparty_id=tx.counterparty_id,
        )

    def settle(self, tx_id: str) -> Transaction:
        """Settle a pending transaction.

        Applies volume-tiered platform fees, updates wallet and reputation,
        and reinforces recently-accessed memories (Hebbian learning).

        Returns:
            The settled Transaction with fee details.

        Raises:
            ValueError: If transaction not found or not pending.
        """
        if not tx_id or not isinstance(tx_id, str):
            raise ValueError("Transaction ID is required")
        tx = self._transactions.get(tx_id)
        if tx is None:
            raise ValueError(f"Transaction {tx_id} not found")
        if tx.status != TransactionStatus.PENDING:
            raise ValueError(f"Transaction {tx_id} is {tx.status.value}, not pending")

        # Apply platform fee (volume-tiered)
        fee_rate = _get_fee_rate(self._total_volume)
        fee_amount = round(tx.amount * fee_rate * 100) / 100
        net_amount = round((tx.amount - fee_amount) * 100) / 100

        tx.platform_fee = fee_amount
        tx.net_amount = net_amount
        tx.status = TransactionStatus.COMPLETED
        tx.completed_at = datetime.now(timezone.utc)

        # Update wallet (with overflow guard)
        new_balance = self._wallet + net_amount
        if new_balance > MAX_WALLET_BALANCE:
            raise ValueError(
                f"Wallet overflow: balance would exceed ${MAX_WALLET_BALANCE:,}"
            )
        self._wallet = round(new_balance * 100) / 100
        self._total_volume += tx.amount

        # Boost reputation
        self._reputation = min(self._reputation + 0.01, 1.0)

        # Hebbian reinforcement: boost recently-accessed memories
        one_hour_ago = datetime.now(timezone.utc).timestamp() - 3600
        reinforced = 0
        for mem in self._memories.values():
            if mem.last_accessed.timestamp() > one_hour_ago:
                mem.importance = min(mem.importance + 0.05, 1.0)
                reinforced += 1

        self._audit("payment:completed", {
            "id": tx.id, "gross_amount": tx.amount,
            "platform_fee": fee_amount, "net_amount": net_amount,
            "fee_rate": fee_rate, "reinforced_memories": reinforced,
        })
        self._emit("payment:completed", {"id": tx.id, "amount": net_amount, "fee": fee_amount})
        self._log(
            f"Settled ${tx.amount:.2f} (fee: ${fee_amount:.2f}, net: ${net_amount:.2f}) -> "
            f"wallet: ${self._wallet:.2f}, reputation: {self._reputation:.2f}, "
            f"reinforced: {reinforced} memories"
        )
        return Transaction(
            id=tx.id, agent_id=tx.agent_id, amount=tx.amount,
            reason=tx.reason, status=tx.status, created_at=tx.created_at,
            completed_at=tx.completed_at, platform_fee=tx.platform_fee,
            net_amount=tx.net_amount, counterparty_id=tx.counterparty_id,
        )

    def refund(self, tx_id: str, reason: Optional[str] = None) -> Transaction:
        """Refund a transaction.

        For completed transactions: deducts net amount from wallet, reduces reputation.
        For pending transactions: cancels the escrow.

        Returns:
            The refunded Transaction.

        Raises:
            ValueError: If transaction not found or already refunded/expired.
        """
        if not tx_id or not isinstance(tx_id, str):
            raise ValueError("Transaction ID is required")
        tx = self._transactions.get(tx_id)
        if tx is None:
            raise ValueError(f"Transaction {tx_id} not found")
        if tx.status == TransactionStatus.REFUNDED:
            raise ValueError(f"Transaction {tx_id} already refunded")
        if tx.status == TransactionStatus.EXPIRED:
            raise ValueError(f"Transaction {tx_id} has expired and cannot be refunded")

        if tx.status == TransactionStatus.COMPLETED:
            refund_amount = tx.net_amount if tx.net_amount is not None else tx.amount
            self._wallet = max(self._wallet - refund_amount, 0.0)
            self._reputation = max(self._reputation - 0.05, 0.0)

        tx.status = TransactionStatus.REFUNDED

        self._audit("payment:refunded", {
            "id": tx.id, "amount": tx.amount,
            "net_refunded": tx.net_amount or tx.amount,
        })
        self._emit("payment:refunded", {"id": tx.id})
        self._log(f"Refunded ${tx.amount:.2f} -> reputation: {self._reputation:.2f}")
        return Transaction(
            id=tx.id, agent_id=tx.agent_id, amount=tx.amount,
            reason=tx.reason, status=tx.status, created_at=tx.created_at,
            completed_at=tx.completed_at, platform_fee=tx.platform_fee,
            net_amount=tx.net_amount, counterparty_id=tx.counterparty_id,
        )

    def dispute(self, tx_id: str, reason: str) -> Dispute:
        """File a dispute on a completed transaction.

        Returns:
            The created Dispute.

        Raises:
            ValueError: If transaction not found or not completed.
        """
        tx = self._transactions.get(tx_id)
        if tx is None:
            raise ValueError(f"Transaction {tx_id} not found")
        if tx.status != TransactionStatus.COMPLETED:
            raise ValueError(
                f"Can only dispute completed transactions (current: {tx.status.value})"
            )

        d = Dispute(
            id=str(uuid.uuid4()),
            tx_id=tx_id,
            agent_id=self.agent_id,
            reason=reason,
            status="open",
            filed_at=datetime.now(timezone.utc),
        )
        self._disputes[d.id] = d
        tx.status = TransactionStatus.DISPUTED

        self._audit("payment:disputed", {"id": tx.id, "dispute_id": d.id, "reason": reason})
        self._emit("payment:disputed", {"tx_id": tx_id, "dispute_id": d.id})
        self._log(f"Dispute filed for tx {tx_id}: {reason}")
        return d

    def resolve_dispute(
        self, dispute_id: str, outcome: str,
    ) -> Dispute:
        """Resolve a dispute.

        Args:
            dispute_id: The dispute to resolve.
            outcome: "refund" or "uphold".

        Returns:
            The resolved Dispute.
        """
        if outcome not in ("refund", "uphold"):
            raise ValueError("Outcome must be 'refund' or 'uphold'")
        d = self._disputes.get(dispute_id)
        if d is None:
            raise ValueError(f"Dispute {dispute_id} not found")
        if d.status != "open":
            raise ValueError(f"Dispute {dispute_id} is already resolved")

        tx = self._transactions.get(d.tx_id)
        if tx is None:
            raise ValueError("Dispute references unknown transaction")

        d.status = "resolved"
        d.outcome = outcome  # type: ignore
        d.resolved_at = datetime.now(timezone.utc)

        if outcome == "refund" and tx.status == TransactionStatus.DISPUTED:
            refund_amount = tx.net_amount if tx.net_amount is not None else tx.amount
            self._wallet = max(self._wallet - refund_amount, 0.0)
            self._reputation = max(self._reputation - 0.05, 0.0)
            tx.status = TransactionStatus.REFUNDED
        elif outcome == "uphold" and tx.status == TransactionStatus.DISPUTED:
            tx.status = TransactionStatus.COMPLETED

        self._audit("dispute:resolved", {"dispute_id": dispute_id, "outcome": outcome})
        self._emit("dispute:resolved", {"dispute_id": dispute_id, "outcome": outcome})
        return d

    # ── Query Methods ─────────────────────────────────────────────────────

    def balance(self) -> BalanceInfo:
        """Get current wallet balance and reputation."""
        return BalanceInfo(
            wallet=round(self._wallet * 100) / 100,
            reputation=self._reputation,
        )

    def profile(self) -> AgentProfile:
        """Get full agent profile."""
        return AgentProfile(
            id=self.agent_id,
            reputation=self._reputation,
            wallet=self._wallet,
            memories_count=len(self._memories),
            transactions_count=len(self._transactions),
        )

    def history(self, limit: int = 50) -> List[Transaction]:
        """Get transaction history (most recent first).

        Args:
            limit: Maximum number of transactions to return.

        Returns:
            List of Transactions, newest first.
        """
        all_txs = list(self._transactions.values())
        all_txs.reverse()
        return [
            Transaction(
                id=tx.id, agent_id=tx.agent_id, amount=tx.amount,
                reason=tx.reason, status=tx.status, created_at=tx.created_at,
                completed_at=tx.completed_at, platform_fee=tx.platform_fee,
                net_amount=tx.net_amount, counterparty_id=tx.counterparty_id,
            )
            for tx in all_txs[:limit]
        ]

    def reputation(self) -> ReputationReport:
        """Generate a full reputation report.

        Returns:
            ReputationReport with settled/refund counts, settlement rate, etc.
        """
        txs = list(self._transactions.values())
        settled = [t for t in txs if t.status == TransactionStatus.COMPLETED]
        refunded = [t for t in txs if t.status == TransactionStatus.REFUNDED]
        total_completed = len(settled) + len(refunded)
        settlement_rate = len(settled) / total_completed if total_completed > 0 else 0.0
        total_value_settled = sum(t.amount for t in settled)

        mems = list(self._memories.values())
        avg_importance = (
            sum(m.importance for m in mems) / len(mems) if mems else 0.0
        )

        age_hours = (
            datetime.now(timezone.utc).timestamp() - self._created_at.timestamp()
        ) / 3600

        report = ReputationReport(
            agent_id=self.agent_id,
            score=self._reputation,
            tier=_reputation_tier(self._reputation),
            settled_count=len(settled),
            refund_count=len(refunded),
            settlement_rate=settlement_rate,
            total_value_settled=total_value_settled,
            memories_count=len(self._memories),
            avg_memory_importance=round(avg_importance * 100) / 100,
            age_hours=round(age_hours * 10) / 10,
            generated_at=datetime.now(timezone.utc),
        )
        self._log(
            f"Reputation: {report.tier.value} ({report.score:.2f}), "
            f"{report.settled_count} settled, "
            f"rate: {report.settlement_rate * 100:.0f}%"
        )
        return report

    def logs(self, limit: int = 50) -> List[AuditEntry]:
        """Get recent audit log entries."""
        return list(self._audit_log[-limit:])

    # ── Internal ──────────────────────────────────────────────────────────

    def _log(self, msg: str) -> None:
        if self._debug:
            print(f"[mnemopay:{self.agent_id}] {msg}")

    def _audit(self, action: str, details: Dict[str, Any]) -> None:
        entry = AuditEntry(
            id=str(uuid.uuid4()),
            agent_id=self.agent_id,
            action=action,
            details=details,
            created_at=datetime.now(timezone.utc),
        )
        self._audit_log.append(entry)
        if len(self._audit_log) > 1000:
            self._audit_log = self._audit_log[-500:]
