"""
Tests for the Python SDK rail port (mnemopay.rails).

Mirrors the TypeScript ``tests/stripe-rail.test.ts`` shape. No real
Stripe — a mock client is injected via ``StripeRail.from_client()``.
"""
from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest

from mnemopay.rails import (
    HoldOptions,
    MockRail,
    PaymentRail,
    PaymentRailResult,
    StripeRail,
)


# ─── MockRail ────────────────────────────────────────────────────────────


class TestMockRail:
    def test_implements_payment_rail_protocol(self) -> None:
        rail = MockRail()
        assert isinstance(rail, PaymentRail)
        assert rail.name == "mock"

    def test_create_hold_returns_held_status(self) -> None:
        rail = MockRail()
        r = rail.create_hold(25.0, "test", "agent-1")
        assert isinstance(r, PaymentRailResult)
        assert r.status == "held"
        assert r.external_id.startswith("mock_hold_")

    def test_create_hold_increments_counter(self) -> None:
        rail = MockRail()
        r1 = rail.create_hold(1.0, "x", "agent-1")
        r2 = rail.create_hold(1.0, "x", "agent-1")
        assert r1.external_id != r2.external_id

    def test_capture_payment(self) -> None:
        rail = MockRail()
        h = rail.create_hold(10.0, "x", "agent-1")
        r = rail.capture_payment(h.external_id, 10.0)
        assert r.status == "captured"
        assert r.external_id == h.external_id
        assert r.receipt_id is not None

    def test_reverse_payment(self) -> None:
        rail = MockRail()
        h = rail.create_hold(10.0, "x", "agent-1")
        r = rail.reverse_payment(h.external_id, 10.0)
        assert r.status == "reversed"


# ─── HoldOptions ────────────────────────────────────────────────────────


class TestHoldOptions:
    def test_defaults(self) -> None:
        opts = HoldOptions()
        assert opts.customer_id is None
        assert opts.payment_method_id is None
        assert opts.off_session is False
        assert opts.metadata == {}

    def test_with_values(self) -> None:
        opts = HoldOptions(
            customer_id="cus_x",
            payment_method_id="pm_y",
            off_session=True,
            metadata={"k": "v"},
        )
        assert opts.customer_id == "cus_x"
        assert opts.metadata["k"] == "v"


# ─── StripeRail (mocked client) ──────────────────────────────────────────


def _make_mock_stripe() -> Any:
    """Build a minimal mock Stripe client matching the surface StripeRail uses."""
    client = MagicMock()
    client.PaymentIntent = MagicMock()
    client.Customer = MagicMock()
    client.SetupIntent = MagicMock()

    intent = MagicMock()
    intent.id = "pi_test_123"
    intent.status = "requires_capture"
    intent.latest_charge = "ch_test_456"
    client.PaymentIntent.create.return_value = intent
    client.PaymentIntent.capture.return_value = MagicMock(
        id="pi_test_123",
        status="succeeded",
        latest_charge="ch_test_456",
    )
    client.PaymentIntent.cancel.return_value = MagicMock(
        id="pi_test_123",
        status="canceled",
    )

    customer = MagicMock(id="cus_test_789")
    client.Customer.create.return_value = customer

    si = MagicMock(id="seti_abc", client_secret="seti_abc_secret")
    client.SetupIntent.create.return_value = si

    return client


class TestStripeRailConstruction:
    def test_constructor_validates_secret_key(self) -> None:
        with pytest.raises(ValueError, match="secret_key"):
            StripeRail("")
        with pytest.raises(ValueError, match="secret_key"):
            StripeRail(None)  # type: ignore[arg-type]

    def test_from_client_rejects_none(self) -> None:
        with pytest.raises(ValueError, match="client"):
            StripeRail.from_client(None)

    def test_from_client_currency_default(self) -> None:
        rail = StripeRail.from_client(_make_mock_stripe())
        assert rail.currency == "usd"
        assert rail.name == "stripe"

    def test_from_client_currency_override(self) -> None:
        rail = StripeRail.from_client(_make_mock_stripe(), currency="eur")
        assert rail.currency == "eur"


class TestStripeRailCreateHold:
    def test_creates_manual_capture_intent(self) -> None:
        client = _make_mock_stripe()
        rail = StripeRail.from_client(client)
        r = rail.create_hold(25.0, "Monthly access", "agent-1")

        assert r.external_id == "pi_test_123"
        # Stripe was called once with the right shape
        call = client.PaymentIntent.create.call_args
        kwargs = call.kwargs
        assert kwargs["amount"] == 2500  # cents
        assert kwargs["currency"] == "usd"
        assert kwargs["capture_method"] == "manual"
        assert kwargs["metadata"]["agentId"] == "agent-1"
        assert kwargs["metadata"]["reason"] == "Monthly access"
        assert kwargs["metadata"]["source"] == "mnemopay"
        # No customer / no confirm — legacy handoff flow
        assert "customer" not in kwargs
        assert "payment_method" not in kwargs
        assert "confirm" not in kwargs

    def test_off_session_with_customer_and_payment_method(self) -> None:
        client = _make_mock_stripe()
        rail = StripeRail.from_client(client)
        rail.create_hold(
            42.5,
            "paid task",
            "agent-9",
            HoldOptions(
                customer_id="cus_real",
                payment_method_id="pm_real",
                off_session=True,
            ),
        )
        kwargs = client.PaymentIntent.create.call_args.kwargs
        assert kwargs["customer"] == "cus_real"
        assert kwargs["payment_method"] == "pm_real"
        assert kwargs["confirm"] is True
        assert kwargs["off_session"] is True

    def test_currency_override_propagates(self) -> None:
        client = _make_mock_stripe()
        rail = StripeRail.from_client(client, currency="eur")
        rail.create_hold(10.0, "x", "agent-1")
        kwargs = client.PaymentIntent.create.call_args.kwargs
        assert kwargs["currency"] == "eur"

    def test_amount_in_cents_rounded(self) -> None:
        client = _make_mock_stripe()
        rail = StripeRail.from_client(client)
        rail.create_hold(0.029, "sub-cent", "agent-1")
        kwargs = client.PaymentIntent.create.call_args.kwargs
        assert kwargs["amount"] == 3  # 0.029 * 100 = 2.9 → round(2.9) = 3

    def test_idempotency_key_forwarded(self) -> None:
        client = _make_mock_stripe()
        rail = StripeRail.from_client(client)
        rail.create_hold(
            10.0,
            "x",
            "agent-1",
            HoldOptions(metadata={"idempotencyKey": "req_abc"}),
        )
        # idempotency_key is passed via separate kwarg, not body params
        call = client.PaymentIntent.create.call_args
        assert call.kwargs.get("idempotency_key") == "req_abc"

    def test_request_id_falls_through_to_idempotency(self) -> None:
        client = _make_mock_stripe()
        rail = StripeRail.from_client(client)
        rail.create_hold(
            10.0,
            "x",
            "agent-1",
            HoldOptions(metadata={"requestId": "req_xyz"}),
        )
        call = client.PaymentIntent.create.call_args
        assert call.kwargs.get("idempotency_key") == "req_xyz"

    def test_reason_truncated_to_500_chars(self) -> None:
        client = _make_mock_stripe()
        rail = StripeRail.from_client(client)
        rail.create_hold(1.0, "a" * 1000, "agent-1")
        kwargs = client.PaymentIntent.create.call_args.kwargs
        assert len(kwargs["metadata"]["reason"]) == 500

    def test_metadata_user_overrides_defaults(self) -> None:
        client = _make_mock_stripe()
        rail = StripeRail.from_client(client)
        rail.create_hold(
            1.0,
            "x",
            "agent-1",
            HoldOptions(metadata={"customField": "v", "source": "from_user"}),
        )
        kwargs = client.PaymentIntent.create.call_args.kwargs
        assert kwargs["metadata"]["customField"] == "v"
        # User-provided metadata wins over defaults (matches TS).
        assert kwargs["metadata"]["source"] == "from_user"

    def test_rejects_non_positive_amount(self) -> None:
        client = _make_mock_stripe()
        rail = StripeRail.from_client(client)
        with pytest.raises(ValueError, match="positive"):
            rail.create_hold(0, "x", "agent-1")
        with pytest.raises(ValueError, match="positive"):
            rail.create_hold(-5, "x", "agent-1")

    def test_rejects_empty_agent_id(self) -> None:
        client = _make_mock_stripe()
        rail = StripeRail.from_client(client)
        with pytest.raises(ValueError, match="agent_id"):
            rail.create_hold(1, "x", "")


class TestStripeRailCapturePayment:
    def test_captures_amount_in_cents(self) -> None:
        client = _make_mock_stripe()
        rail = StripeRail.from_client(client)
        r = rail.capture_payment("pi_test_123", 25.0)
        assert r.external_id == "pi_test_123"
        assert r.status == "succeeded"
        assert r.receipt_id == "ch_test_456"

        call = client.PaymentIntent.capture.call_args
        assert call.args[0] == "pi_test_123"
        assert call.kwargs["amount_to_capture"] == 2500
        assert call.kwargs["idempotency_key"] == "cap_pi_test_123"

    def test_rejects_empty_external_id(self) -> None:
        client = _make_mock_stripe()
        rail = StripeRail.from_client(client)
        with pytest.raises(ValueError, match="external_id"):
            rail.capture_payment("", 1.0)


class TestStripeRailReversePayment:
    def test_reverses_returns_reversed_status(self) -> None:
        client = _make_mock_stripe()
        rail = StripeRail.from_client(client)
        r = rail.reverse_payment("pi_test_123", 25.0)
        assert r.external_id == "pi_test_123"
        assert r.status == "reversed"
        client.PaymentIntent.cancel.assert_called_once_with("pi_test_123")

    def test_rejects_empty_external_id(self) -> None:
        client = _make_mock_stripe()
        rail = StripeRail.from_client(client)
        with pytest.raises(ValueError, match="external_id"):
            rail.reverse_payment("", 1.0)


class TestStripeRailCustomerHelpers:
    def test_create_customer(self) -> None:
        client = _make_mock_stripe()
        rail = StripeRail.from_client(client)
        result = rail.create_customer("user@example.com", name="Test User")
        assert result == {"customer_id": "cus_test_789"}
        kwargs = client.Customer.create.call_args.kwargs
        assert kwargs["email"] == "user@example.com"
        assert kwargs["name"] == "Test User"

    def test_create_customer_rejects_empty_email(self) -> None:
        client = _make_mock_stripe()
        rail = StripeRail.from_client(client)
        with pytest.raises(ValueError, match="email"):
            rail.create_customer("")

    def test_create_setup_intent(self) -> None:
        client = _make_mock_stripe()
        rail = StripeRail.from_client(client)
        result = rail.create_setup_intent("cus_test_789")
        assert result == {
            "setup_intent_id": "seti_abc",
            "client_secret": "seti_abc_secret",
        }
        kwargs = client.SetupIntent.create.call_args.kwargs
        assert kwargs["customer"] == "cus_test_789"
        assert kwargs["usage"] == "off_session"

    def test_create_setup_intent_rejects_empty(self) -> None:
        client = _make_mock_stripe()
        rail = StripeRail.from_client(client)
        with pytest.raises(ValueError, match="customer_id"):
            rail.create_setup_intent("")
