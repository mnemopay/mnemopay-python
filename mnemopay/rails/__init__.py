"""
Payment Rail Abstraction — Python port

Mirrors the TypeScript ``@mnemopay/sdk`` rail interface (``src/rails/index.ts``).
Sync API to match the rest of the Python SDK (no asyncio).

The default ``MockRail`` keeps in-memory ledger behavior. Real rails
(``StripeRail``, ``PaystackRail`` future, ``LightningRail`` future) connect
to actual payment processors via lazy peer-dependency imports.

Usage::

    from mnemopay import MnemoPay
    from mnemopay.rails import StripeRail

    agent = MnemoPay("my-agent", payment_rail=StripeRail("sk_test_..."))

This module is part of the v1.0.0b4 rail port — added 2026-05-08
to bring the Python SDK to parity with TypeScript v1.6.0-alpha.0.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol, runtime_checkable


# ─── Result + Options ────────────────────────────────────────────────────


@dataclass
class PaymentRailResult:
    """Return shape from every rail method."""

    external_id: str
    status: str
    receipt_id: Optional[str] = None


@dataclass
class HoldOptions:
    """Optional rail-specific options for ``create_hold``.

    Rails ignore fields they don't support. Existing callers that pass
    no options keep the pre-rails behaviour (rail charges its own
    default source).
    """

    customer_id: Optional[str] = None
    payment_method_id: Optional[str] = None
    email: Optional[str] = None
    authorization_code: Optional[str] = None
    off_session: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class PaymentRail(Protocol):
    """Protocol every rail implements.

    Use ``isinstance(rail, PaymentRail)`` for runtime checks. Static
    type checkers see the structural conformance.
    """

    name: str

    def create_hold(
        self,
        amount: float,
        reason: str,
        agent_id: str,
        opts: Optional[HoldOptions] = None,
    ) -> PaymentRailResult:
        """Create a hold/escrow on the external payment system.

        Called during ``charge()``. The hold should NOT capture funds yet.
        """
        ...

    def capture_payment(
        self,
        external_id: str,
        amount: float,
    ) -> PaymentRailResult:
        """Capture / finalize the payment. Called during ``settle()``.
        Moves real money."""
        ...

    def reverse_payment(
        self,
        external_id: str,
        amount: float,
    ) -> PaymentRailResult:
        """Reverse / cancel the payment. Called during ``refund()``.
        Returns money to payer."""
        ...


# ─── Mock Rail (default — in-memory) ──────────────────────────────────────


class MockRail:
    """Default rail. Records hold/capture/reverse counters in memory.

    Useful for tests + dev without external infra. Conforms to the
    ``PaymentRail`` Protocol.
    """

    name: str = "mock"

    def __init__(self) -> None:
        self._counter: int = 0

    def create_hold(
        self,
        amount: float,
        reason: str,
        agent_id: str,
        opts: Optional[HoldOptions] = None,
    ) -> PaymentRailResult:
        self._counter += 1
        return PaymentRailResult(
            external_id=f"mock_hold_{self._counter}",
            status="held",
        )

    def capture_payment(
        self,
        external_id: str,
        amount: float,
    ) -> PaymentRailResult:
        return PaymentRailResult(
            external_id=external_id,
            status="captured",
            receipt_id=f"mock_receipt_{self._counter}",
        )

    def reverse_payment(
        self,
        external_id: str,
        amount: float,
    ) -> PaymentRailResult:
        return PaymentRailResult(
            external_id=external_id,
            status="reversed",
        )


# ─── Re-exports ──────────────────────────────────────────────────────────

# Rails are kept in separate modules so peer-dep imports / socket operations
# are only attempted when someone constructs a specific Rail.
from .stripe import StripeRail  # noqa: E402
from .paystack import PaystackRail, NIGERIAN_BANKS  # noqa: E402
from .lightning import LightningRail  # noqa: E402

__all__ = [
    "PaymentRail",
    "PaymentRailResult",
    "HoldOptions",
    "MockRail",
    "StripeRail",
    "PaystackRail",
    "LightningRail",
    "NIGERIAN_BANKS",
]

