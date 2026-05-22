"""
Paystack Payment Rail — Python port

Mirrors the TypeScript `@mnemopay/sdk` Paystack rail implementation
(`src/rails/paystack.ts`). Zero external dependencies, pure stdlib.
Supports NGN, GHS, ZAR, KES, USD. Implements:
  - two-phase commit escrow via reference verification (holds/captures)
  - recurring charge authorizations
  - transfer recipients and bank payout transfers
  - timing-safe HMAC-SHA512 webhook signature verification
  - LRU/TTL size-bounded idempotency guard
"""
from __future__ import annotations

import hashlib
import hmac
import json
import random
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import HoldOptions, PaymentRailResult

# ─── Types ──────────────────────────────────────────────────────────────────

PaystackCurrency = str  # "NGN" | "GHS" | "ZAR" | "KES" | "USD"


@dataclass
class PaystackHoldResult(PaymentRailResult):
    """Return shape from create_hold."""
    authorization_url: Optional[str] = None
    access_code: Optional[str] = None
    reference: str = ""


@dataclass
class PaystackVerifyResult(PaymentRailResult):
    """Detailed verify response containing customer and billing authorization."""
    amount: float = 0.0
    currency: PaystackCurrency = "NGN"
    customer_email: Optional[str] = None
    authorization: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    gateway_response: Optional[str] = None
    paid_at: Optional[str] = None


@dataclass
class PaystackTransferRecipient:
    """Paystack bank transfer recipient details."""
    recipient_code: str
    name: str
    bank_code: str
    account_number: str
    currency: PaystackCurrency


@dataclass
class PaystackTransferResult(PaymentRailResult):
    """Result from initiate_transfer payout."""
    reference: str = ""
    amount: float = 0.0
    transfer_status: str = "pending"


class PaystackRail:
    """Paystack payment rail conforming to PaymentRail protocol.

    Provides multi-rail transactions for emerging markets.
    """

    name: str = "paystack"
    MAX_PROCESSED_REFS = 10000
    REF_TTL_MS = 24 * 60 * 60 * 1000  # 24h

    def __init__(self, secret_key: str, config: Optional[Dict[str, Any]] = None) -> None:
        if not secret_key or not isinstance(secret_key, str):
            raise ValueError("Paystack secret key is required")
        if not secret_key.startswith("sk_"):
            raise ValueError("Invalid Paystack secret key format (must start with sk_)")

        config = config or {}
        self.secret_key = secret_key
        self.currency: PaystackCurrency = config.get("currency", "NGN")
        self.base_url = config.get("base_url", "https://api.paystack.co").rstrip("/")
        self.callback_url = config.get("callback_url")
        self.timeout_ms = config.get("timeout_ms", 30000)
        self.channels = config.get("channels")

        # Reference-based idempotency guard (ref -> timestamp)
        self._processed_refs: Dict[str, float] = {}
        # In-flight capture locks or tracking is handled synchronously in Python
        self._in_flight_captures: Dict[str, Any] = {}

    # ── PaymentRail Interface ───────────────────────────────────────────────

    def create_hold(
        self,
        amount: float,
        reason: str,
        agent_id: str,
        opts: Optional[HoldOptions] = None,
    ) -> PaystackHoldResult:
        if not isinstance(amount, (int, float)) or amount <= 0:
            raise ValueError("Amount must be a positive finite number")
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("agent_id is required")

        opts = opts or HoldOptions()
        # Deterministic reference
        idempotency_key = (
            opts.metadata.get("idempotencyKey")
            if opts.metadata
            else None
        ) or (opts.metadata.get("requestId") if opts.metadata else None)

        reference = (
            opts.metadata.get("reference")
            if opts.metadata
            else None
        ) or idempotency_key or f"mnemo_{agent_id}_{int(time.time() * 1000)}_{random.getrandbits(16):04x}"

        amount_in_minor = self.to_minor_units(amount)
        email = opts.email or f"{agent_id}@mnemopay.agent"

        if opts.authorization_code:
            # Charge saved card directly
            payload = {
                "authorization_code": opts.authorization_code,
                "email": email,
                "amount": amount_in_minor,
                "currency": self.currency,
                "reference": reference,
                "metadata": {
                    "agentId": agent_id,
                    "reason": (reason or "")[:500],
                    "source": "mnemopay",
                    **(opts.metadata or {}),
                },
            }
            res = self._request("POST", "/transaction/charge_authorization", payload)
            data = res.get("data") or {}
            return PaystackHoldResult(
                external_id=reference,
                status=data.get("status", "pending"),
                reference=reference,
            )

        # Initialize checkout session
        payload = {
            "email": email,
            "amount": amount_in_minor,
            "currency": self.currency,
            "reference": reference,
            "metadata": {
                "agentId": agent_id,
                "reason": (reason or "")[:500],
                "source": "mnemopay",
                **(opts.metadata or {}),
            },
        }

        if self.callback_url:
            payload["callback_url"] = self.callback_url
        if self.channels:
            payload["channels"] = self.channels

        res = self._request("POST", "/transaction/initialize", payload)
        data = res.get("data") or {}

        return PaystackHoldResult(
            external_id=reference,
            status="initialized",
            authorization_url=data.get("authorization_url"),
            access_code=data.get("access_code"),
            reference=reference,
        )

    def capture_payment(
        self,
        external_id: str,
        amount: float,  # noqa: ARG002
    ) -> PaystackVerifyResult:
        if not external_id or not isinstance(external_id, str):
            raise ValueError("Transaction reference is required")

        # Idempotency check
        self._evict_expired_refs()
        if external_id in self._processed_refs:
            res = self._request("GET", f"/transaction/verify/{urllib.parse.quote(external_id)}")
            return self._map_verify_response(res, external_id)

        res = self._request("GET", f"/transaction/verify/{urllib.parse.quote(external_id)}")
        data = res.get("data") or {}

        if data.get("status") != "success":
            return PaystackVerifyResult(
                external_id=external_id,
                status=data.get("status", "failed"),
                amount=self.from_minor_units(data.get("amount", 0)),
                currency=data.get("currency", self.currency),
                gateway_response=data.get("gateway_response"),
            )

        # Mark processed
        self._processed_refs[external_id] = time.time() * 1000
        return self._map_verify_response(res, external_id)

    def reverse_payment(
        self,
        external_id: str,
        amount: float,
    ) -> PaymentRailResult:
        if not external_id or not isinstance(external_id, str):
            raise ValueError("Transaction reference is required")
        if not isinstance(amount, (int, float)) or amount <= 0:
            raise ValueError("Refund amount must be a positive finite number")

        # Get transaction id by verifying reference first
        verify_res = self._request("GET", f"/transaction/verify/{urllib.parse.quote(external_id)}")
        data = verify_res.get("data") or {}
        tx_id = data.get("id")

        if not tx_id:
            raise RuntimeError(f"Transaction {external_id} not found on Paystack")

        payload = {
            "transaction": tx_id,
            "amount": self.to_minor_units(amount),
        }
        res = self._request("POST", "/refund", payload)
        res_data = res.get("data") or {}

        # Allow re-verification of the reference
        self._processed_refs.pop(external_id, None)

        return PaymentRailResult(
            external_id=external_id,
            status=res_data.get("status", "pending"),
            receipt_id=str(res_data.get("id")) if res_data.get("id") is not None else None,
        )

    # ── Paystack Specific Primitives ────────────────────────────────────────

    def create_transfer_recipient(
        self,
        name: str,
        account_number: str,
        bank_code: str,
        currency: Optional[PaystackCurrency] = None,
    ) -> PaystackTransferRecipient:
        if not name or not account_number or not bank_code:
            raise ValueError("Name, account number, and bank code are required")

        # Validate bank account resolution
        self.resolve_bank(account_number, bank_code)

        cur = currency or self.currency
        payload = {
            "type": "nuban",
            "name": name,
            "account_number": account_number,
            "bank_code": bank_code,
            "currency": cur,
        }

        res = self._request("POST", "/transferrecipient", payload)
        data = res.get("data") or {}

        return PaystackTransferRecipient(
            recipient_code=data.get("recipient_code", ""),
            name=data.get("details", {}).get("account_name") or name,
            bank_code=bank_code,
            account_number=account_number,
            currency=cur,
        )

    def initiate_transfer(
        self,
        recipient_code: str,
        amount: float,
        reason: str,
        agent_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
    ) -> PaystackTransferResult:
        if not recipient_code:
            raise ValueError("Recipient code is required")
        if not isinstance(amount, (int, float)) or amount <= 0:
            raise ValueError("Transfer amount must be a positive finite number")

        reference = idempotency_key or f"mnemo_xfer_{int(time.time() * 1000)}_{random.getrandbits(16):04x}"

        payload = {
            "source": "balance",
            "amount": self.to_minor_units(amount),
            "recipient": recipient_code,
            "reason": (reason or "")[:500],
            "reference": reference,
            "metadata": {
                "agentId": agent_id,
                "source": "mnemopay",
            },
        }

        res = self._request("POST", "/transfer", payload)
        data = res.get("data") or {}

        return PaystackTransferResult(
            external_id=data.get("transfer_code") or reference,
            status=data.get("status", "pending"),
            reference=reference,
            amount=amount,
            transfer_status=data.get("status", "pending"),
        )

    def verify_webhook(self, raw_body: str | bytes, signature: str) -> Dict[str, Any]:
        """Validate HMAC-SHA512 webhook signature from Paystack."""
        if not signature:
            raise ValueError("Webhook signature is required")
        if not raw_body:
            raise ValueError("Webhook body is required")

        body_bytes = raw_body if isinstance(raw_body, bytes) else raw_body.encode("utf-8")

        expected = hmac.new(
            self.secret_key.encode("utf-8"),
            body_bytes,
            hashlib.sha512,
        ).hexdigest()

        # Timing safe comparison
        if not hmac.compare_digest(signature.lower(), expected.lower()):
            raise ValueError("Invalid webhook signature")

        return json.loads(body_bytes.decode("utf-8"))

    def resolve_bank(self, account_number: str, bank_code: str) -> Dict[str, str]:
        """Verify the bank account details match."""
        path = f"/bank/resolve?account_number={urllib.parse.quote(account_number)}&bank_code={urllib.parse.quote(bank_code)}"
        res = self._request("GET", path)
        data = res.get("data") or {}
        return {
            "account_name": data.get("account_name", ""),
            "account_number": data.get("account_number", account_number),
        }

    def list_banks(self, country: str = "nigeria", per_page: int = 100) -> List[Dict[str, str]]:
        """Fetch list of banks supported by Paystack."""
        path = f"/bank?country={urllib.parse.quote(country)}&perPage={per_page}"
        res = self._request("GET", path)
        data = res.get("data") or []
        return [{"name": b.get("name", ""), "code": b.get("code", "")} for b in data]

    # ── Unit Helpers ────────────────────────────────────────────────────────

    def to_minor_units(self, amount: float) -> int:
        return round(amount * 100)

    def from_minor_units(self, minor_amount: float) -> float:
        return round(minor_amount) / 100.0

    # ── HTTP Requests ───────────────────────────────────────────────────────

    def _request(self, method: str, path: str, body: Optional[Any] = None) -> Any:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.secret_key}",
            "Content-Type": "application/json",
        }

        data_bytes = None
        if body is not None:
            data_bytes = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data_bytes,
            headers=headers,
            method=method,
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout_ms / 1000.0) as res:
                resp_bytes = res.read()
                if not resp_bytes:
                    return {}
                return json.loads(resp_bytes.decode("utf-8"))
        except urllib.error.HTTPError as e:
            try:
                err_resp = e.read().decode("utf-8")
                err_json = json.loads(err_resp)
            except Exception:
                err_json = {}
            msg = err_json.get("message") or err_json.get("error") or str(e)
            raise RuntimeError(f"Paystack error: {msg}") from e
        except Exception as e:
            raise RuntimeError(f"Request failed: {e}") from e

    def _map_verify_response(self, response: Dict[str, Any], external_id: str) -> PaystackVerifyResult:
        data = response.get("data") or {}
        auth = data.get("authorization")
        auth_mapped = None

        if auth:
            auth_mapped = {
                "authorizationCode": auth.get("authorization_code"),
                "cardType": auth.get("card_type"),
                "last4": auth.get("last4"),
                "bank": auth.get("bank"),
                "reusable": auth.get("reusable", False),
            }

        return PaystackVerifyResult(
            external_id=external_id,
            status=data.get("status", "unknown"),
            receipt_id=str(data.get("id")) if data.get("id") is not None else None,
            amount=self.from_minor_units(data.get("amount", 0)),
            currency=data.get("currency", self.currency),
            customer_email=data.get("customer", {}).get("email"),
            authorization=auth_mapped,
            metadata=data.get("metadata") or {},
            gateway_response=data.get("gateway_response"),
            paid_at=data.get("paid_at"),
        )

    def _evict_expired_refs(self) -> None:
        if len(self._processed_refs) <= self.MAX_PROCESSED_REFS / 2:
            return
        cutoff = (time.time() * 1000) - self.REF_TTL_MS
        to_del = [k for k, v in self._processed_refs.items() if v < cutoff]
        for k in to_del:
            self._processed_refs.pop(k, None)

        if len(self._processed_refs) > self.MAX_PROCESSED_REFS:
            # Evict oldest by timestamp
            sorted_refs = sorted(self._processed_refs.items(), key=lambda x: x[1])
            excess = len(self._processed_refs) - self.MAX_PROCESSED_REFS
            for k, _ in sorted_refs[:excess]:
                self._processed_refs.pop(k, None)


# ─── Nigerian Bank Codes ────────────────────────────────────────────────────

NIGERIAN_BANKS = {
    "access": "044",
    "citibank": "023",
    "ecobank": "050",
    "fidelity": "070",
    "firstbank": "011",
    "fcmb": "214",
    "gtbank": "058",
    "heritage": "030",
    "keystone": "082",
    "polaris": "076",
    "providus": "101",
    "stanbic": "221",
    "standard": "068",
    "sterling": "232",
    "uba": "033",
    "union": "032",
    "unity": "215",
    "wema": "035",
    "zenith": "057",
    "kuda": "50211",
    "opay": "999992",
    "palmpay": "999991",
    "moniepoint": "50515",
}
