"""
Tests for MerkleTree memory integrity.
"""

import hashlib

import pytest

from mnemopay.integrity import MerkleTree, _sha256
from mnemopay.types import IntegritySnapshot, MerkleLeaf, MerkleProof


class TestMerkleBasics:
    def test_add_leaf(self):
        tree = MerkleTree()
        h = tree.add_leaf("mem-1", "Hello world", "2026-01-01T00:00:00Z")
        assert isinstance(h, str)
        assert len(h) == 64  # SHA-256 hex

    def test_add_multiple_leaves(self):
        tree = MerkleTree()
        h1 = tree.add_leaf("mem-1", "First", "2026-01-01T00:00:00Z")
        h2 = tree.add_leaf("mem-2", "Second", "2026-01-01T00:00:01Z")
        assert h1 != h2
        assert tree.size == 2

    def test_size_property(self):
        tree = MerkleTree()
        assert tree.size == 0
        tree.add_leaf("mem-1", "Content")
        assert tree.size == 1

    def test_empty_memory_id_raises(self):
        tree = MerkleTree()
        with pytest.raises(ValueError, match="memory_id"):
            tree.add_leaf("", "Content")

    def test_empty_content_raises(self):
        tree = MerkleTree()
        with pytest.raises(ValueError, match="content"):
            tree.add_leaf("mem-1", "")

    def test_duplicate_hash_impossible_different_index(self):
        """Different memory IDs at different indices produce different hashes
        even with identical content, because index is part of the hash input."""
        tree = MerkleTree()
        h1 = tree.add_leaf("mem-1", "Same content", "2026-01-01T00:00:00Z")
        h2 = tree.add_leaf("mem-2", "Same content", "2026-01-01T00:00:00Z")
        assert h1 != h2  # Index prevents collisions

    def test_update_memory_replaces(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Original", "2026-01-01T00:00:00Z")
        tree.add_leaf("mem-1", "Updated", "2026-01-01T00:00:01Z")
        assert tree.size == 1  # Updated, not duplicated


class TestMerkleRemove:
    def test_remove_existing(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Content")
        assert tree.remove_leaf("mem-1") is True
        assert tree.size == 0

    def test_remove_nonexistent(self):
        tree = MerkleTree()
        assert tree.remove_leaf("nonexistent") is False

    def test_remove_changes_root(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "First", "2026-01-01T00:00:00Z")
        tree.add_leaf("mem-2", "Second", "2026-01-01T00:00:01Z")
        root_before = tree.get_root()
        tree.remove_leaf("mem-1")
        root_after = tree.get_root()
        assert root_before != root_after


class TestMerkleRoot:
    def test_empty_tree_root(self):
        tree = MerkleTree()
        root = tree.get_root()
        assert root == _sha256("empty")

    def test_single_leaf_root(self):
        """Single leaf: root IS the leaf hash (loop doesn't execute when len==1)."""
        tree = MerkleTree()
        h = tree.add_leaf("mem-1", "Content", "2026-01-01T00:00:00Z")
        root = tree.get_root()
        # With 1 leaf, while loop condition (len > 1) is false, so root = leaf hash
        assert root == h

    def test_root_changes_on_add(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "First", "2026-01-01T00:00:00Z")
        root1 = tree.get_root()
        tree.add_leaf("mem-2", "Second", "2026-01-01T00:00:01Z")
        root2 = tree.get_root()
        assert root1 != root2

    def test_root_deterministic(self):
        tree1 = MerkleTree()
        tree1.add_leaf("mem-1", "Content", "2026-01-01T00:00:00Z")
        tree2 = MerkleTree()
        tree2.add_leaf("mem-1", "Content", "2026-01-01T00:00:00Z")
        assert tree1.get_root() == tree2.get_root()

    def test_two_leaf_root(self):
        tree = MerkleTree()
        h1 = tree.add_leaf("mem-1", "A", "t1")
        h2 = tree.add_leaf("mem-2", "B", "t2")
        root = tree.get_root()
        expected = _sha256(h1 + h2)
        assert root == expected


class TestMerkleProof:
    def test_generate_and_verify_proof(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "First", "2026-01-01T00:00:00Z")
        tree.add_leaf("mem-2", "Second", "2026-01-01T00:00:01Z")
        tree.add_leaf("mem-3", "Third", "2026-01-01T00:00:02Z")
        proof = tree.get_proof("mem-2")
        assert MerkleTree.verify_proof(proof) is True

    def test_proof_for_first_leaf(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "First", "2026-01-01T00:00:00Z")
        tree.add_leaf("mem-2", "Second", "2026-01-01T00:00:01Z")
        proof = tree.get_proof("mem-1")
        assert MerkleTree.verify_proof(proof) is True

    def test_proof_for_last_leaf(self):
        tree = MerkleTree()
        for i in range(5):
            tree.add_leaf(f"mem-{i}", f"Content {i}", f"2026-01-01T00:00:0{i}Z")
        proof = tree.get_proof("mem-4")
        assert MerkleTree.verify_proof(proof) is True

    def test_proof_single_leaf(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Only leaf", "2026-01-01T00:00:00Z")
        proof = tree.get_proof("mem-1")
        assert MerkleTree.verify_proof(proof) is True

    def test_tampered_proof_fails(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "First", "2026-01-01T00:00:00Z")
        tree.add_leaf("mem-2", "Second", "2026-01-01T00:00:01Z")
        proof = tree.get_proof("mem-1")
        # Tamper with the root
        proof.root_hash = "0" * 64
        assert MerkleTree.verify_proof(proof) is False

    def test_proof_nonexistent_raises(self):
        tree = MerkleTree()
        with pytest.raises(ValueError, match="not in tree"):
            tree.get_proof("nonexistent")

    def test_proof_removed_raises(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Content")
        tree.remove_leaf("mem-1")
        with pytest.raises(ValueError, match="not in tree"):
            tree.get_proof("mem-1")

    def test_invalid_proof_returns_false(self):
        assert MerkleTree.verify_proof(None) is False  # type: ignore

    def test_many_leaves_proof(self):
        tree = MerkleTree()
        for i in range(20):
            tree.add_leaf(f"mem-{i}", f"Content {i}", f"2026-01-01T00:{i:02d}:00Z")
        for i in range(20):
            proof = tree.get_proof(f"mem-{i}")
            assert MerkleTree.verify_proof(proof) is True


class TestVerifyMemory:
    def test_verify_correct_content(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Secret data", "2026-01-01T00:00:00Z")
        assert tree.verify_memory("mem-1", "Secret data", "2026-01-01T00:00:00Z") is True

    def test_verify_wrong_content(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Original", "2026-01-01T00:00:00Z")
        assert tree.verify_memory("mem-1", "Tampered", "2026-01-01T00:00:00Z") is False

    def test_verify_wrong_timestamp(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Content", "2026-01-01T00:00:00Z")
        assert tree.verify_memory("mem-1", "Content", "2026-12-31T00:00:00Z") is False

    def test_verify_nonexistent(self):
        tree = MerkleTree()
        assert tree.verify_memory("nonexistent", "Content", "ts") is False


class TestSnapshot:
    def test_snapshot_captures_state(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Content 1", "t1")
        tree.add_leaf("mem-2", "Content 2", "t2")
        snap = tree.snapshot()
        assert snap.leaf_count == 2
        assert snap.root_hash == tree.get_root()
        assert len(snap.snapshot_hash) == 64

    def test_no_tamper_detected(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Content", "t1")
        snap = tree.snapshot()
        result = tree.detect_tampering(snap)
        assert result.tampered is False

    def test_add_detected_as_tamper(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Content", "t1")
        snap = tree.snapshot()
        tree.add_leaf("mem-2", "New content", "t2")
        result = tree.detect_tampering(snap)
        assert result.tampered is True
        assert "added" in result.summary

    def test_remove_detected_as_tamper(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Content 1", "t1")
        tree.add_leaf("mem-2", "Content 2", "t2")
        snap = tree.snapshot()
        tree.remove_leaf("mem-1")
        result = tree.detect_tampering(snap)
        assert result.tampered is True
        assert "removed" in result.summary

    def test_snapshot_self_integrity(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Content", "t1")
        snap = tree.snapshot()
        # Tamper with the snapshot
        tampered_snap = IntegritySnapshot(
            root_hash=snap.root_hash,
            leaf_count=snap.leaf_count,
            timestamp=snap.timestamp,
            snapshot_hash="0" * 64,  # Bad hash
        )
        result = tree.detect_tampering(tampered_snap)
        assert result.tampered is True
        assert "snapshot itself" in result.summary


class TestTreeIntegrity:
    def test_valid_tree(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Content 1", "t1")
        tree.add_leaf("mem-2", "Content 2", "t2")
        result = tree.verify_tree_integrity()
        assert result["valid"] is True
        assert result["leaf_count"] == 2

    def test_empty_tree_valid(self):
        tree = MerkleTree()
        result = tree.verify_tree_integrity()
        assert result["valid"] is True
        assert result["leaf_count"] == 0


class TestCompact:
    def test_compact_removes_gaps(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Content 1", "t1")
        tree.add_leaf("mem-2", "Content 2", "t2")
        tree.add_leaf("mem-3", "Content 3", "t3")
        tree.remove_leaf("mem-2")
        result = tree.compact()
        assert result["removed"] == 1
        assert result["remaining"] == 2
        assert tree.size == 2

    def test_compact_preserves_content(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Content 1", "t1")
        tree.add_leaf("mem-2", "Content 2", "t2")
        tree.remove_leaf("mem-1")
        tree.compact()
        # mem-2 should still be accessible
        integrity = tree.verify_tree_integrity()
        assert integrity["valid"] is True


class TestSerialization:
    def test_serialize_roundtrip(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Content 1", "t1")
        tree.add_leaf("mem-2", "Content 2", "t2")
        data = tree.serialize()
        restored = MerkleTree.deserialize(data)
        assert restored.size == 2
        assert restored.get_root() == tree.get_root()

    def test_deserialize_empty(self):
        tree = MerkleTree.deserialize({"leaves": [], "root_hash": _sha256("empty")})
        assert tree.size == 0

    def test_deserialize_invalid_raises(self):
        with pytest.raises(ValueError, match="Invalid"):
            MerkleTree.deserialize({})

    def test_deserialize_bad_hash_raises(self):
        with pytest.raises(ValueError, match="leaf hash"):
            MerkleTree.deserialize({"leaves": [{"hash": "bad", "memory_id": "x", "index": 0}]})

    def test_deserialize_duplicate_id_raises(self):
        h = _sha256("test1")
        h2 = _sha256("test2")
        with pytest.raises(ValueError, match="Duplicate memoryId"):
            MerkleTree.deserialize({"leaves": [
                {"hash": h, "memory_id": "same", "index": 0},
                {"hash": h2, "memory_id": "same", "index": 1},
            ]})

    def test_deserialize_root_mismatch_raises(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Content", "t1")
        data = tree.serialize()
        data["root_hash"] = "0" * 64
        with pytest.raises(ValueError, match="Root hash mismatch"):
            MerkleTree.deserialize(data)


class TestAuditLog:
    def test_audit_log_captures_events(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Content", "t1")
        tree.remove_leaf("mem-1")
        log = tree.get_audit_log()
        events = [e["event"] for e in log]
        assert "leaf_added" in events
        assert "leaf_removed" in events

    def test_verification_events_logged(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Content", "2026-01-01T00:00:00Z")
        tree.verify_memory("mem-1", "Content", "2026-01-01T00:00:00Z")
        log = tree.get_audit_log()
        events = [e["event"] for e in log]
        assert "verification_pass" in events

    def test_snapshot_event_logged(self):
        tree = MerkleTree()
        tree.add_leaf("mem-1", "Content", "t1")
        tree.snapshot()
        log = tree.get_audit_log()
        events = [e["event"] for e in log]
        assert "snapshot_created" in events


class TestSHA256:
    def test_sha256_deterministic(self):
        assert _sha256("test") == _sha256("test")

    def test_sha256_matches_hashlib(self):
        expected = hashlib.sha256("test".encode()).hexdigest()
        assert _sha256("test") == expected

    def test_sha256_length(self):
        assert len(_sha256("any input")) == 64
