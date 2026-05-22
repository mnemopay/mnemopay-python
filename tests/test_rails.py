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


# ─── PaystackRail (mocked) ───────────────────────────────────────────────

from unittest.mock import patch
import json
import hmac
import hashlib
from mnemopay.rails import PaystackRail, NIGERIAN_BANKS


def _mock_response(data: dict, status: int = 200) -> MagicMock:
    res = MagicMock()
    res.read.return_value = json.dumps(data).encode("utf-8")
    res.status = status
    res.__enter__.return_value = res
    return res


class TestPaystackRail:
    def test_constructor_validates_key(self) -> None:
        with pytest.raises(ValueError, match="secret key"):
            PaystackRail("")
        with pytest.raises(ValueError, match="must start with sk_"):
            PaystackRail("invalid_key")

    @patch("urllib.request.urlopen")
    def test_create_hold_checkout(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({
            "status": True,
            "data": {
                "authorization_url": "https://checkout.paystack.com/auth",
                "access_code": "code123"
            }
        })
        rail = PaystackRail("sk_test_123")
        res = rail.create_hold(50.0, "Escrow Hold", "agent-1")

        assert res.status == "initialized"
        assert res.authorization_url == "https://checkout.paystack.com/auth"
        assert res.access_code == "code123"
        assert res.reference.startswith("mnemo_agent-1_")

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.method == "POST"
        assert req.full_url == "https://api.paystack.co/transaction/initialize"
        body = json.loads(req.data.decode("utf-8"))
        assert body["amount"] == 5000  # minor units
        assert body["email"] == "agent-1@mnemopay.agent"

    @patch("urllib.request.urlopen")
    def test_create_hold_charge_authorization(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({
            "status": True,
            "data": {"status": "success"}
        })
        rail = PaystackRail("sk_test_123")
        res = rail.create_hold(
            15.5,
            "recurrent charge",
            "agent-2",
            HoldOptions(authorization_code="AUTH_567")
        )

        assert res.status == "success"
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.method == "POST"
        assert req.full_url == "https://api.paystack.co/transaction/charge_authorization"
        body = json.loads(req.data.decode("utf-8"))
        assert body["authorization_code"] == "AUTH_567"
        assert body["amount"] == 1550

    @patch("urllib.request.urlopen")
    def test_capture_payment(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({
            "status": True,
            "data": {
                "status": "success",
                "id": 9999,
                "amount": 2500,
                "currency": "NGN",
                "customer": {"email": "agent@domain.com"},
                "authorization": {"authorization_code": "AUTH_XYZ", "reusable": True}
            }
        })
        rail = PaystackRail("sk_test_123")
        res = rail.capture_payment("ref_xyz", 25.0)

        assert res.status == "success"
        assert res.amount == 25.0
        assert res.currency == "NGN"
        assert res.customer_email == "agent@domain.com"
        assert res.authorization["authorizationCode"] == "AUTH_XYZ"

        # Assert cached idempotency
        mock_urlopen.reset_mock()
        res2 = rail.capture_payment("ref_xyz", 25.0)
        assert res2.status == "success"
        mock_urlopen.assert_called_once()  # Called only once via GET (cached bypass)

    @patch("urllib.request.urlopen")
    def test_reverse_payment(self, mock_urlopen: MagicMock) -> None:
        # 1. verify request mock, 2. refund request mock
        mock_urlopen.side_effect = [
            _mock_response({"status": True, "data": {"id": 12345}}),
            _mock_response({"status": True, "data": {"status": "reversed", "id": 678}})
        ]
        rail = PaystackRail("sk_test_123")
        res = rail.reverse_payment("ref_xyz", 10.0)

        assert res.status == "reversed"
        assert res.receipt_id == "678"

    @patch("urllib.request.urlopen")
    def test_create_transfer_recipient(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.side_effect = [
            _mock_response({"status": True, "data": {"account_name": "Agent Inc"}}),
            _mock_response({"status": True, "data": {"recipient_code": "RCP_999", "details": {"account_name": "Agent Inc"}}})
        ]
        rail = PaystackRail("sk_test_123")
        recipient = rail.create_transfer_recipient("Agent Inc", "1234567890", NIGERIAN_BANKS["gtbank"])

        assert recipient.recipient_code == "RCP_999"
        assert recipient.name == "Agent Inc"

    @patch("urllib.request.urlopen")
    def test_initiate_transfer(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({
            "status": True,
            "data": {"transfer_code": "TRF_abc", "status": "success"}
        })
        rail = PaystackRail("sk_test_123")
        res = rail.initiate_transfer("RCP_999", 100.0, "settle payout")

        assert res.external_id == "TRF_abc"
        assert res.status == "success"

    def test_verify_webhook(self) -> None:
        secret = "sk_test_123"
        rail = PaystackRail(secret)
        body = '{"event":"charge.success","data":{"id":12}}'
        sig = hmac.new(secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha512).hexdigest()

        event = rail.verify_webhook(body, sig)
        assert event["event"] == "charge.success"

        with pytest.raises(ValueError, match="Invalid webhook signature"):
            rail.verify_webhook(body, "wrong_sig")


# ─── LightningRail (mocked) ──────────────────────────────────────────────

from mnemopay.rails import LightningRail


class TestLightningRail:
    def test_ssrf_is_private_or_reserved(self) -> None:
        # Loopbacks
        assert LightningRail.is_private_or_reserved("localhost") is True
        assert LightningRail.is_private_or_reserved("127.0.0.1") is True
        assert LightningRail.is_private_or_reserved("[::1]") is True
        # Private classes
        assert LightningRail.is_private_or_reserved("10.0.0.1") is True
        assert LightningRail.is_private_or_reserved("192.168.1.100") is True
        assert LightningRail.is_private_or_reserved("172.16.5.2") is True
        # Octal IP checks
        assert LightningRail.is_private_or_reserved("0177.0000.0000.0001") is True
        # Hex IP checks
        assert LightningRail.is_private_or_reserved("0x7f000001") is True
        # Number conversion check
        assert LightningRail.is_private_or_reserved("2130706433") is True
        # Suffix internal
        assert LightningRail.is_private_or_reserved("host.internal") is True
        assert LightningRail.is_private_or_reserved("service.local") is True
        # Metadata class
        assert LightningRail.is_private_or_reserved("169.254.169.254") is True
        assert LightningRail.is_private_or_reserved("metadata.google.internal") is True
        # Valid domain
        assert LightningRail.is_private_or_reserved("api.lightning.network") is False

    def test_constructor_validates_ssrf(self) -> None:
        with pytest.raises(ValueError, match="protocol"):
            LightningRail("ftp://localhost:8080", "macaroon")
        with pytest.raises(ValueError, match="private/internal"):
            LightningRail("https://127.0.0.1:8080", "macaroon")
        with pytest.raises(ValueError, match="private/internal"):
            LightningRail("https://metadata.google.internal:80", "mac")

    @patch("urllib.request.urlopen")
    def test_create_hold(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({
            "r_hash": "aGFzaDEyMw==",  # base64 payment hash
            "payment_request": "lnbc500n1p..."
        })
        rail = LightningRail("https://api.lightning.network:8080", "macaroon_hex", btc_price_usd=50000)
        res = rail.create_hold(10.0, "API compute hold", "agent-x")

        assert res.external_id == "aGFzaDEyMw=="
        assert res.status == "invoice_created"
        assert res.receipt_id == "lnbc500n1p..."

        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.method == "POST"
        body = json.loads(req.data.decode("utf-8"))
        assert body["value"] == "20000"  # 10.0 USD -> 0.0002 BTC -> 20,000 sats

    @patch("urllib.request.urlopen")
    def test_capture_payment(self, mock_urlopen: MagicMock) -> None:
        mock_urlopen.return_value = _mock_response({
            "settled": True,
            "payment_request": "lnbc500n1p..."
        })
        rail = LightningRail("https://api.lightning.network:8080", "macaroon_hex")
        
        # Test standard external_id and URL-safe base64 path mappings (removes +, / and =)
        res = rail.capture_payment("a+b/c=", 10.0)

        assert res.status == "captured"
        assert res.receipt_id == "lnbc500n1p..."
        call_args = mock_urlopen.call_args
        req = call_args[0][0]
        assert req.full_url.endswith("/v1/invoice/a-b_c")  # Url safe conversion verify

    def test_reverse_payment(self) -> None:
        rail = LightningRail("https://api.lightning.network:8080", "macaroon_hex")
        res = rail.reverse_payment("ref_xyz", 10.0)
        assert res.status == "expired"

