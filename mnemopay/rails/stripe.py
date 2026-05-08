"""
Stripe Rail — Python port

Mirrors ``StripeRail`` from the TypeScript SDK
(``@mnemopay/sdk/src/rails/index.ts``). Uses the official ``stripe``
Python SDK (>=12.0.0) as a peer dependency loaded lazily so users
on other rails don't need it installed.

Two-phase commit via PaymentIntent ``capture_method='manual'``:
  - ``create_hold`` reserves funds (real escrow on Stripe's side)
  - ``capture_payment`` captures
  - ``reverse_payment`` cancels

Race-protection: parallel ``capture_payment`` calls on the same
intent are deduplicated via a per-intent threading lock.
"""
from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from . import HoldOptions, PaymentRailResult


class StripeRail:
    """Stripe rail using PaymentIntents with manual capture.

    Requires the ``stripe`` Python SDK. Install with::

        pip install stripe

    Usage::

        from mnemopay.rails import StripeRail
        rail = StripeRail("sk_test_...")

    For testing, inject a pre-built client::

        rail = StripeRail.from_client(mock_stripe_client, currency="usd")
    """

    name: str = "stripe"

    def __init__(self, secret_key: str, currency: str = "usd") -> None:
        if not secret_key or not isinstance(secret_key, str):
            raise ValueError("StripeRail: secret_key is required")
        self.currency = currency
        try:
            import stripe  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "Stripe package not installed. Run: pip install stripe"
            ) from e
        self._stripe = stripe
        self._stripe.api_key = secret_key
        # Per-intent locks to dedupe parallel capture calls. Mirrors the
        # TS in_flight_captures Map<intentId, Promise>.
        self._capture_locks: Dict[str, threading.Lock] = {}
        self._capture_lock_factory = threading.Lock()

    @classmethod
    def from_client(cls, client: Any, currency: str = "usd") -> "StripeRail":
        """Build a StripeRail from a pre-constructed Stripe client.

        Useful for tests (inject a mock) and for apps that want to
        share a Stripe client across multiple rails. Bypasses the
        ``import stripe`` path.
        """
        if client is None:
            raise ValueError("StripeRail.from_client: client is required")
        rail = cls.__new__(cls)
        rail.name = "stripe"
        rail.currency = currency
        rail._stripe = client
        rail._capture_locks = {}
        rail._capture_lock_factory = threading.Lock()
        return rail

    # ── Hold ─────────────────────────────────────────────────────────────

    def create_hold(
        self,
        amount: float,
        reason: str,
        agent_id: str,
        opts: Optional[HoldOptions] = None,
    ) -> PaymentRailResult:
        if not isinstance(amount, (int, float)) or amount <= 0:
            raise ValueError("StripeRail.create_hold: amount must be positive")
        if not agent_id or not isinstance(agent_id, str):
            raise ValueError("StripeRail.create_hold: agent_id is required")

        opts = opts or HoldOptions()
        params: Dict[str, Any] = {
            "amount": round(amount * 100),  # Stripe uses minor units
            "currency": self.currency,
            "capture_method": "manual",  # Hold funds, don't capture yet
            "metadata": {
                "agentId": agent_id,
                "reason": (reason or "")[:500],
                "source": "mnemopay",
                **(opts.metadata or {}),
            },
        }

        if opts.customer_id:
            params["customer"] = opts.customer_id
        if opts.payment_method_id:
            params["payment_method"] = opts.payment_method_id
            params["confirm"] = True
            if opts.off_session:
                params["off_session"] = True

        idempotency_key = (
            opts.metadata.get("idempotencyKey")
            if opts.metadata
            else None
        ) or (opts.metadata.get("requestId") if opts.metadata else None)

        kwargs: Dict[str, Any] = {}
        if idempotency_key:
            kwargs["idempotency_key"] = idempotency_key

        intent = self._stripe.PaymentIntent.create(**params, **kwargs)

        return PaymentRailResult(
            external_id=intent.id,
            status=intent.status,
        )

    # ── Capture ──────────────────────────────────────────────────────────

    def capture_payment(
        self,
        external_id: str,
        amount: float,
    ) -> PaymentRailResult:
        if not external_id:
            raise ValueError("StripeRail.capture_payment: external_id is required")

        # Race-protection: serialize captures of the same intent through
        # a per-intent lock. The first caller does the API call; later
        # callers wait + return the same shape.
        with self._capture_lock_factory:
            lock = self._capture_locks.get(external_id)
            if lock is None:
                lock = threading.Lock()
                self._capture_locks[external_id] = lock

        with lock:
            intent = self._stripe.PaymentIntent.capture(
                external_id,
                amount_to_capture=round(amount * 100),
                idempotency_key=f"cap_{external_id}",
            )

        # Cleanup lock — best-effort (other waiters that hold a ref
        # finish their wait + early-return cleanly because Stripe's
        # idempotency-key gives them the same intent back).
        with self._capture_lock_factory:
            self._capture_locks.pop(external_id, None)

        return PaymentRailResult(
            external_id=intent.id,
            status=intent.status,
            receipt_id=getattr(intent, "latest_charge", None),
        )

    # ── Reverse ──────────────────────────────────────────────────────────

    def reverse_payment(
        self,
        external_id: str,
        amount: float,  # noqa: ARG002  -- API parity with TS
    ) -> PaymentRailResult:
        if not external_id:
            raise ValueError("StripeRail.reverse_payment: external_id is required")
        intent = self._stripe.PaymentIntent.cancel(external_id)
        return PaymentRailResult(
            external_id=intent.id,
            status="reversed" if intent.status == "canceled" else intent.status,
        )

    # ── Customer onboarding helpers ──────────────────────────────────────

    def create_customer(
        self,
        email: str,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, str]:
        """Create a Stripe customer. Returns {'customer_id': 'cus_...'}."""
        if not email or not isinstance(email, str):
            raise ValueError("StripeRail.create_customer: email is required")
        params: Dict[str, Any] = {"email": email}
        if name:
            params["name"] = name
        if metadata:
            params["metadata"] = metadata
        customer = self._stripe.Customer.create(**params)
        return {"customer_id": customer.id}

    def create_setup_intent(
        self,
        customer_id: str,
    ) -> Dict[str, str]:
        """Create a SetupIntent for collecting a card via Stripe.js
        (off-session charges later). Returns
        {'setup_intent_id': 'seti_...', 'client_secret': '...'}.
        """
        if not customer_id or not isinstance(customer_id, str):
            raise ValueError("StripeRail.create_setup_intent: customer_id is required")
        si = self._stripe.SetupIntent.create(
            customer=customer_id,
            usage="off_session",
        )
        return {
            "setup_intent_id": si.id,
            "client_secret": si.client_secret,
        }
