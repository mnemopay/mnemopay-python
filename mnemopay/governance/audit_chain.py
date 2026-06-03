"""
MnemoPay Audit Chain -- event-stream + tree-Merkle root audit chain.

Optimized for Article 12 compliance bundles, signed exports, and event-level signatures.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Union

# ─── Types ──────────────────────────────────────────────────────────────────


@dataclass
class ChainEvent:
    id: str
    sequence: int
    occurred_at: str
    kind: str
    payload: Dict[str, Any]
    parent_id: Optional[str] = None
    signature: Optional[str] = None


@dataclass
class ChainBundle:
    version: int
    built_at: str
    meta: Dict[str, Any]
    events: List[ChainEvent]
    merkle_root: str


# ─── Canonicalization & Hash ────────────────────────────────────────────────


def canonicalize(value: Any) -> str:
    """JCS-compliant canonical serialization (alphabetical keys, no whitespace)."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        # Convert float to integer if it has no fractional part to match JS float behavior
        if isinstance(value, float) and value.is_integer():
            return str(int(value))
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, str):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(canonicalize(x) for x in value) + "]"
    if isinstance(value, dict):
        keys = sorted(value.keys())
        return (
            "{"
            + ",".join(
                json.dumps(k, separators=(",", ":")) + ":" + canonicalize(value[k])
                for k in keys
            )
            + "}"
        )
    if isinstance(value, datetime):
        # TS dates serialize as ISO strings (e.g. YYYY-MM-DDTHH:mm:ss.sssZ)
        s = value.astimezone(timezone.utc).isoformat()
        if s.endswith("+00:00"):
            s = s[:-6] + "Z"
        # Match standard TS 3-digit millisecond resolution
        if "." in s:
            parts = s.split(".")
            tail = parts[1]
            z = "Z" if tail.endswith("Z") else ""
            num_tail = tail.rstrip("Z")
            if len(num_tail) > 3:
                num_tail = num_tail[:3]
            s = parts[0] + "." + num_tail + z
        return canonicalize(s)
    # Support dataclasses
    if hasattr(value, "__dataclass_fields__"):
        d = {k: v for k, v in value.__dict__.items() if v is not None}
        return canonicalize(d)
    return json.dumps(value, separators=(",", ":"))


def sha256_hex(input_data: str | bytes) -> str:
    if isinstance(input_data, str):
        data = input_data.encode("utf-8")
    else:
        data = input_data
    return hashlib.sha256(data).hexdigest()


# ─── AuditChain ─────────────────────────────────────────────────────────────


class AuditChain:
    """Chained JSONL and in-memory event-log audit chain using Merkle root calculation."""

    def __init__(
        self,
        signer: Optional[Callable[[str], str]] = None,
        sequence_start: int = 0,
        path: Optional[str] = None,
    ) -> None:
        self._events: List[ChainEvent] = []
        self._leaf_hashes: List[str] = []
        self.signer = signer
        self.next_sequence = sequence_start
        self.path = path

        if self.path:
            # Best effort directory creation
            dir_name = os.path.dirname(self.path)
            if dir_name:
                try:
                    os.makedirs(dir_name, exist_ok=True)
                except Exception as err:
                    print(
                        f"[mnemopay/governance/audit_chain] mkdir failed for {self.path}: {err}"
                    )

    def emit(
        self,
        kind: str,
        payload: Dict[str, Any],
        parent_id: Optional[str] = None,
    ) -> ChainEvent:
        draft = ChainEvent(
            id=str(uuid.uuid4()),
            sequence=self.next_sequence,
            occurred_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            kind=kind,
            payload=payload,
            parent_id=parent_id,
        )
        self.next_sequence += 1

        if self.signer:
            canonical = canonicalize(draft)
            draft.signature = self.signer(canonical)
            self._leaf_hashes.append(sha256_hex(canonicalize(draft)))
        else:
            self._leaf_hashes.append(sha256_hex(canonicalize(draft)))

        self._events.append(draft)

        if self.path:
            try:
                # Append to JSONL file
                with open(self.path, "a", encoding="utf-8") as f:
                    # Clean signature representation
                    event_dict = {
                        "id": draft.id,
                        "sequence": draft.sequence,
                        "occurred_at": draft.occurred_at,
                        "kind": draft.kind,
                        "payload": draft.payload,
                    }
                    if draft.parent_id:
                        event_dict["parent_id"] = draft.parent_id
                    if draft.signature:
                        event_dict["signature"] = draft.signature
                    f.write(json.dumps(event_dict, separators=(",", ":")) + "\n")
            except Exception as err:
                print(
                    f"[mnemopay/governance/audit_chain] disk append failed: {err}"
                )

        return draft

    def events(self) -> List[ChainEvent]:
        return self._events.copy()

    def roll_merkle_root(self) -> str:
        """Roll a tree-Merkle root over all emitted events."""
        if not self._events:
            return ""
        layer = self._leaf_hashes.copy()
        while len(layer) > 1:
            next_layer = []
            for i in range(0, len(layer), 2):
                a = layer[i]
                b = layer[i + 1] if i + 1 < len(layer) else a
                next_layer.append(sha256_hex(a + b))
            layer = next_layer
        return layer[0] if layer else ""

    def to_bundle(self, meta: Optional[Dict[str, Any]] = None) -> ChainBundle:
        return ChainBundle(
            version=1,
            built_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            meta=meta or {},
            events=self._events.copy(),
            merkle_root=self.roll_merkle_root(),
        )

    def roll_and_export(
        self, path_out: str, meta: Optional[Dict[str, Any]] = None
    ) -> ChainBundle:
        bundle = self.to_bundle(meta)
        dir_name = os.path.dirname(path_out)
        if dir_name:
            try:
                os.makedirs(dir_name, exist_ok=True)
            except Exception:
                pass
        try:
            # Custom encoder/serializer for bundle
            bundle_dict = {
                "version": bundle.version,
                "built_at": bundle.built_at,
                "meta": bundle.meta,
                "events": [
                    {
                        "id": e.id,
                        "sequence": e.sequence,
                        "occurred_at": e.occurred_at,
                        "kind": e.kind,
                        "payload": e.payload,
                        **({"parent_id": e.parent_id} if e.parent_id else {}),
                        **({"signature": e.signature} if e.signature else {}),
                    }
                    for e in bundle.events
                ],
                "merkle_root": bundle.merkle_root,
            }
            with open(path_out, "w", encoding="utf-8") as f:
                json.dump(bundle_dict, f, indent=2)
        except Exception as err:
            print(f"[mnemopay/governance/audit_chain] export failed: {err}")
        return bundle


# ─── Verification ──────────────────────────────────────────────────────────


def verify_bundle(
    bundle: ChainBundle,
    verify_event_signature_fn: Optional[Callable[[ChainEvent], bool]] = None,
) -> Dict[str, Any]:
    """Verify an external chain bundle by recalculating its Merkle root."""
    if not bundle.events:
        if bundle.merkle_root == "":
            return {"ok": True}
        return {"ok": False, "reason": "root_mismatch"}

    layer = [sha256_hex(canonicalize(e)) for e in bundle.events]
    while len(layer) > 1:
        next_layer = []
        for i in range(0, len(layer), 2):
            a = layer[i]
            b = layer[i + 1] if i + 1 < len(layer) else a
            next_layer.append(sha256_hex(a + b))
        layer = next_layer

    if layer[0] != bundle.merkle_root:
        return {"ok": False, "reason": "root_mismatch"}

    if verify_event_signature_fn:
        for idx, ev in enumerate(bundle.events):
            if ev.signature and not verify_event_signature_fn(ev):
                return {"ok": False, "reason": "bad_event_signature", "index": idx}

    return {"ok": True}
