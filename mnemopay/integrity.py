"""
Merkle Tree Memory Integrity -- Tamper-Evident Agent Memory

Protects against MemoryGraft attacks by maintaining a SHA-256 Merkle hash tree
over all memory writes. Any modification to any memory changes the root hash.

Attack vectors defended:
  - MemoryGraft: Injecting fabricated memories
  - Memory tampering: Modifying existing content
  - Memory deletion: Removing memories silently
  - Replay attacks: Re-inserting old memories
  - Reordering attacks: Changing memory chronology

References:
  - Merkle, R. (1987). "A Digital Signature Based on a Conventional Encryption Function"
  - OWASP Agentic AI Top 10 2026: A03 -- Memory Poisoning
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .types import (
    IntegritySnapshot,
    MerkleLeaf,
    MerkleProof,
    MerkleProofStep,
    TamperResult,
)


def _sha256(data: str) -> str:
    """Compute SHA-256 hex digest of a string."""
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


class MerkleTree:
    """SHA-256 Merkle tree for tamper-evident memory integrity."""

    MAX_AUDIT_LOG = 1000
    MAX_LEAVES = 100_000

    def __init__(self) -> None:
        self._leaves: List[Optional[MerkleLeaf]] = []
        self._leaf_map: Dict[str, int] = {}  # memory_id -> index
        self._hash_set: set[str] = set()
        self._audit_log: List[Dict[str, Any]] = []

    def add_leaf(
        self,
        memory_id: str,
        content: str,
        timestamp: Optional[str] = None,
    ) -> str:
        """Add a memory entry to the tree. Returns the leaf hash."""
        if not memory_id or not isinstance(memory_id, str):
            raise ValueError("memory_id is required")
        if not content or not isinstance(content, str):
            raise ValueError("content is required")
        if len([l for l in self._leaves if l is not None]) >= self.MAX_LEAVES:
            raise ValueError(
                f"Merkle tree leaf limit reached ({self.MAX_LEAVES}). "
                f"Compact old memories first."
            )

        ts = timestamp or datetime.now(timezone.utc).isoformat()
        index = len(self._leaves)
        leaf_data = f"{index}:{memory_id}:{content}:{ts}"
        hash_val = _sha256(leaf_data)

        # Duplicate detection
        if hash_val in self._hash_set:
            raise ValueError(f"Duplicate leaf detected for memory {memory_id}")

        # If this memoryId was already added, remove old leaf
        if memory_id in self._leaf_map:
            self._remove_leaf_by_memory_id(memory_id)

        leaf = MerkleLeaf(hash=hash_val, memory_id=memory_id, index=index)
        self._leaves.append(leaf)
        self._leaf_map[memory_id] = len(self._leaves) - 1
        self._hash_set.add(hash_val)

        self._audit("leaf_added", memory_id)
        return hash_val

    def remove_leaf(self, memory_id: str) -> bool:
        """Remove a memory from the tree."""
        if memory_id not in self._leaf_map:
            return False
        self._remove_leaf_by_memory_id(memory_id)
        self._audit("leaf_removed", memory_id)
        return True

    def _remove_leaf_by_memory_id(self, memory_id: str) -> None:
        idx = self._leaf_map.get(memory_id)
        if idx is None:
            return

        leaf = self._leaves[idx]
        if leaf is not None:
            self._hash_set.discard(leaf.hash)

        self._leaves[idx] = None
        del self._leaf_map[memory_id]

    def get_root(self) -> str:
        """Compute the Merkle root from all current leaves."""
        active_leaves = [l for l in self._leaves if l is not None]
        if not active_leaves:
            return _sha256("empty")

        level = [l.hash for l in active_leaves]

        while len(level) > 1:
            next_level: List[str] = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else left
                next_level.append(_sha256(left + right))
            level = next_level

        return level[0]

    def get_proof(self, memory_id: str) -> MerkleProof:
        """Generate a Merkle proof for a specific memory."""
        idx = self._leaf_map.get(memory_id)
        if idx is None:
            raise ValueError(f"Memory {memory_id} not in tree")

        leaf = self._leaves[idx]
        if leaf is None:
            raise ValueError(f"Memory {memory_id} has been removed")

        active_leaves = [l for l in self._leaves if l is not None]
        active_index = next(
            (i for i, l in enumerate(active_leaves) if l.memory_id == memory_id),
            -1,
        )
        if active_index == -1:
            raise ValueError(f"Memory {memory_id} not found in active leaves")

        level = [l.hash for l in active_leaves]
        path: List[MerkleProofStep] = []
        current_idx = active_index

        while len(level) > 1:
            next_level: List[str] = []
            for i in range(0, len(level), 2):
                left = level[i]
                right = level[i + 1] if i + 1 < len(level) else left

                if i == current_idx or i + 1 == current_idx:
                    if current_idx % 2 == 0:
                        sibling = level[i + 1] if i + 1 < len(level) else level[i]
                        path.append(MerkleProofStep(hash=sibling, direction="right"))
                    else:
                        path.append(MerkleProofStep(hash=level[i], direction="left"))

                next_level.append(_sha256(left + right))
            current_idx = current_idx // 2
            level = next_level

        return MerkleProof(
            leaf_hash=leaf.hash,
            memory_id=memory_id,
            path=path,
            root_hash=self.get_root(),
        )

    @staticmethod
    def verify_proof(proof: MerkleProof) -> bool:
        """Verify a Merkle proof links the leaf to the root."""
        if not proof or not proof.leaf_hash or not proof.root_hash or not isinstance(proof.path, list):
            return False

        current_hash = proof.leaf_hash

        for step in proof.path:
            if step.direction == "left":
                current_hash = _sha256(step.hash + current_hash)
            else:
                current_hash = _sha256(current_hash + step.hash)

        return current_hash == proof.root_hash

    def verify_memory(self, memory_id: str, content: str, timestamp: str) -> bool:
        """Verify a specific memory's content against the tree."""
        idx = self._leaf_map.get(memory_id)
        if idx is None:
            return False

        leaf = self._leaves[idx]
        if leaf is None:
            return False

        leaf_data = f"{leaf.index}:{memory_id}:{content}:{timestamp}"
        expected_hash = _sha256(leaf_data)

        match = expected_hash == leaf.hash
        self._audit("verification_pass" if match else "verification_fail", memory_id)
        return match

    def snapshot(self) -> IntegritySnapshot:
        """Take a snapshot of current tree state."""
        root_hash = self.get_root()
        leaf_count = len([l for l in self._leaves if l is not None])
        timestamp = datetime.now(timezone.utc).isoformat()
        snapshot_data = f"{root_hash}:{leaf_count}:{timestamp}"
        snapshot_hash = _sha256(snapshot_data)

        self._audit("snapshot_created")

        return IntegritySnapshot(
            root_hash=root_hash,
            leaf_count=leaf_count,
            timestamp=timestamp,
            snapshot_hash=snapshot_hash,
        )

    def detect_tampering(self, previous_snapshot: IntegritySnapshot) -> TamperResult:
        """Detect tampering by comparing current root to a previous snapshot."""
        # Verify the snapshot itself
        expected_snapshot_hash = _sha256(
            f"{previous_snapshot.root_hash}:{previous_snapshot.leaf_count}:"
            f"{previous_snapshot.timestamp}"
        )
        if expected_snapshot_hash != previous_snapshot.snapshot_hash:
            return TamperResult(
                tampered=True,
                failed_memories=[],
                current_root=self.get_root(),
                expected_root=previous_snapshot.root_hash,
                summary=(
                    "CRITICAL: The snapshot itself has been tampered with. "
                    "Cannot trust any comparison."
                ),
            )

        current_root = self.get_root()
        current_leaf_count = len([l for l in self._leaves if l is not None])

        if current_root == previous_snapshot.root_hash:
            return TamperResult(
                tampered=False,
                failed_memories=[],
                current_root=current_root,
                expected_root=previous_snapshot.root_hash,
                summary=(
                    f"Integrity verified. {current_leaf_count} memories, "
                    f"root matches snapshot."
                ),
            )

        leaf_diff = current_leaf_count - previous_snapshot.leaf_count
        sign = "+" if leaf_diff >= 0 else ""
        summary = (
            f"Root hash mismatch. Leaves: {previous_snapshot.leaf_count} -> "
            f"{current_leaf_count} ({sign}{leaf_diff})."
        )

        if leaf_diff > 0:
            summary += f" {leaf_diff} new memories added since snapshot."
        elif leaf_diff < 0:
            summary += f" {abs(leaf_diff)} memories removed since snapshot."
        else:
            summary += " Same leaf count but content changed -- possible memory modification."
            self._audit("tamper_detected")

        return TamperResult(
            tampered=True,
            failed_memories=[],
            current_root=current_root,
            expected_root=previous_snapshot.root_hash,
            summary=summary,
        )

    def verify_tree_integrity(self) -> Dict[str, Any]:
        """Full tree verification."""
        active_leaves = [l for l in self._leaves if l is not None]
        computed_root = self.get_root()

        # Verify no hash collisions
        unique_hashes = set(l.hash for l in active_leaves)
        if len(unique_hashes) != len(active_leaves):
            return {"valid": False, "leaf_count": len(active_leaves), "root_hash": computed_root}

        # Verify leaf map consistency
        for mem_id, idx in self._leaf_map.items():
            leaf = self._leaves[idx]
            if leaf is None or leaf.memory_id != mem_id:
                return {"valid": False, "leaf_count": len(active_leaves), "root_hash": computed_root}

        return {"valid": True, "leaf_count": len(active_leaves), "root_hash": computed_root}

    @property
    def size(self) -> int:
        """Number of active (non-removed) leaves."""
        return len([l for l in self._leaves if l is not None])

    def get_audit_log(self) -> List[Dict[str, Any]]:
        """Get a copy of the audit log."""
        return list(self._audit_log)

    def compact(self) -> Dict[str, int]:
        """Compact the tree by removing null entries. Invalidates existing proofs."""
        before = len(self._leaves)
        active_leaves = [l for l in self._leaves if l is not None]

        self._leaves = []
        self._leaf_map.clear()
        self._hash_set.clear()

        for i, leaf in enumerate(active_leaves):
            new_leaf = MerkleLeaf(hash=leaf.hash, memory_id=leaf.memory_id, index=i)
            self._leaves.append(new_leaf)
            self._leaf_map[new_leaf.memory_id] = i
            self._hash_set.add(new_leaf.hash)

        return {"removed": before - len(active_leaves), "remaining": len(active_leaves)}

    def serialize(self) -> Dict[str, Any]:
        """Serialize tree state for persistence."""
        active_leaves = [l for l in self._leaves if l is not None]
        return {
            "leaves": [
                {"hash": l.hash, "memory_id": l.memory_id, "index": l.index}
                for l in active_leaves
            ],
            "root_hash": self.get_root(),
        }

    @classmethod
    def deserialize(cls, data: Dict[str, Any]) -> MerkleTree:
        """Deserialize from persisted state with validation."""
        if not data or not isinstance(data.get("leaves"), list):
            raise ValueError("Invalid MerkleTree data")

        tree = cls()
        seen_ids: set[str] = set()
        seen_hashes: set[str] = set()

        for leaf_data in data["leaves"]:
            hash_val = leaf_data.get("hash", "")
            memory_id = leaf_data.get("memory_id", "")

            if not hash_val or not isinstance(hash_val, str) or len(hash_val) != 64:
                raise ValueError(f"Invalid leaf hash for memory {memory_id}")
            if not memory_id or not isinstance(memory_id, str):
                raise ValueError("Invalid leaf memoryId")
            if memory_id in seen_ids:
                raise ValueError(f"Duplicate memoryId in tree: {memory_id}")
            if hash_val in seen_hashes:
                raise ValueError(f"Duplicate hash in tree: {hash_val}")

            seen_ids.add(memory_id)
            seen_hashes.add(hash_val)

            index = len(tree._leaves)
            leaf = MerkleLeaf(hash=hash_val, memory_id=memory_id, index=index)
            tree._leaves.append(leaf)
            tree._leaf_map[memory_id] = index
            tree._hash_set.add(hash_val)

        # Verify root matches if provided
        if data.get("root_hash"):
            computed_root = tree.get_root()
            if computed_root != data["root_hash"]:
                raise ValueError(
                    f"Root hash mismatch on deserialize: computed {computed_root}, "
                    f"expected {data['root_hash']}. Tree may be corrupted."
                )

        return tree

    def _audit(
        self,
        event: str,
        memory_id: Optional[str] = None,
    ) -> None:
        needs_root = event in (
            "verification_pass", "verification_fail",
            "tamper_detected", "snapshot_created",
        )
        self._audit_log.append({
            "event": event,
            "memory_id": memory_id,
            "root_hash": self.get_root() if needs_root else "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        if len(self._audit_log) > self.MAX_AUDIT_LOG:
            self._audit_log = self._audit_log[-(self.MAX_AUDIT_LOG // 2):]
