"""Tests for CommerceEngine — autonomous shopping with escrow + approval flow."""

import asyncio
from datetime import timedelta

import pytest
from mnemopay import MnemoPay
from mnemopay.core import SETTLEMENT_HOLD_MINUTES
from mnemopay.commerce import (
    CommerceEngine,
    Mandate,
    MockProvider,
    OrderStatus,
    Product,
)


def run(coro):
    """Helper to run async code in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


def age_escrow_after_hold(agent: MnemoPay, order) -> None:
    agent._transactions[order.escrow_tx_id].created_at -= timedelta(
        minutes=SETTLEMENT_HOLD_MINUTES + 1
    )


class TestCommerceEngine:
    def setup_method(self):
        self.agent = MnemoPay("test-shopper")
        # Boost reputation so charges go through
        self.agent._reputation = 0.8
        self.engine = CommerceEngine(self.agent, MockProvider())

    def test_search_returns_products(self):
        results = run(self.engine.search("API"))
        assert len(results) > 0
        assert all(isinstance(p, Product) for p in results)

    def test_search_empty_query_raises(self):
        with pytest.raises(ValueError, match="query is required"):
            run(self.engine.search(""))

    def test_search_long_query_raises(self):
        with pytest.raises(ValueError, match="too long"):
            run(self.engine.search("x" * 501))

    def test_buy_no_mandate_needs_approval(self):
        products = run(self.engine.search("API"))
        order = run(self.engine.buy(products[0]))
        assert order.status == OrderStatus.PENDING_APPROVAL

    def test_buy_under_mandate_auto_approves(self):
        self.engine.set_mandate(Mandate(max_single=50.0, max_daily=200.0))
        products = run(self.engine.search("API"))
        order = run(self.engine.buy(products[0]))
        assert order.status in (OrderStatus.PURCHASED, OrderStatus.ESCROW_FUNDED)

    def test_buy_over_mandate_needs_approval(self):
        self.engine.set_mandate(Mandate(max_single=1.0, max_daily=5.0))
        products = run(self.engine.search("API"))
        order = run(self.engine.buy(products[0]))
        assert order.status == OrderStatus.PENDING_APPROVAL

    def test_approve_order(self):
        products = run(self.engine.search("API"))
        order = run(self.engine.buy(products[0]))
        assert order.status == OrderStatus.PENDING_APPROVAL
        approved = self.engine.approve(order.id)
        assert approved.status == OrderStatus.APPROVED

    def test_reject_order(self):
        products = run(self.engine.search("API"))
        order = run(self.engine.buy(products[0]))
        rejected = self.engine.reject(order.id, "Too expensive")
        assert rejected.status == OrderStatus.REJECTED
        assert rejected.rejection_reason == "Too expensive"

    def test_approve_nonexistent_raises(self):
        with pytest.raises(ValueError, match="No pending"):
            self.engine.approve("fake-id")

    def test_reject_nonexistent_raises(self):
        with pytest.raises(ValueError, match="No pending"):
            self.engine.reject("fake-id")

    def test_pending_approvals_list(self):
        products = run(self.engine.search("API"))
        run(self.engine.buy(products[0]))
        run(self.engine.buy(products[1] if len(products) > 1 else products[0]))
        pending = self.engine.pending_approvals()
        assert len(pending) == 2

    def test_orders_list(self):
        self.engine.set_mandate(Mandate(max_single=50.0, max_daily=200.0))
        products = run(self.engine.search("API"))
        run(self.engine.buy(products[0]))
        orders = self.engine.orders()
        assert len(orders) == 1

    def test_confirm_delivery(self):
        self.engine.set_mandate(Mandate(max_single=50.0, max_daily=200.0))
        products = run(self.engine.search("API"))
        order = run(self.engine.buy(products[0]))
        if order.status == OrderStatus.PURCHASED:
            age_escrow_after_hold(self.agent, order)
            delivered = self.engine.confirm_delivery(order.id)
            assert delivered.status == OrderStatus.DELIVERED

    def test_confirm_delivery_before_hold_expires_raises(self):
        self.engine.set_mandate(Mandate(max_single=50.0, max_daily=200.0))
        products = run(self.engine.search("API"))
        order = run(self.engine.buy(products[0]))
        if order.status == OrderStatus.PURCHASED:
            with pytest.raises(ValueError, match="Settlement hold"):
                self.engine.confirm_delivery(order.id)
            assert order.status == OrderStatus.PURCHASED

    def test_confirm_delivery_wrong_status_raises(self):
        products = run(self.engine.search("API"))
        order = run(self.engine.buy(products[0]))
        with pytest.raises(ValueError, match="not purchased"):
            self.engine.confirm_delivery(order.id)

    def test_invalid_quantity_raises(self):
        products = run(self.engine.search("API"))
        with pytest.raises(ValueError, match="Quantity"):
            run(self.engine.buy(products[0], quantity=0))

    def test_mandate_negative_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            self.engine.set_mandate(Mandate(max_single=-1.0))

    def test_mandate_daily_limit(self):
        self.engine.set_mandate(Mandate(max_single=50.0, max_daily=15.0))
        products = run(self.engine.search("API"))
        # First buy should auto-approve ($10)
        order1 = run(self.engine.buy(products[0]))
        assert order1.status != OrderStatus.PENDING_APPROVAL
        # Second buy should need approval (would exceed $15 daily)
        order2 = run(self.engine.buy(products[0]))
        assert order2.status == OrderStatus.PENDING_APPROVAL

    def test_require_approval_above(self):
        self.engine.set_mandate(
            Mandate(max_single=100.0, max_daily=500.0, require_approval_above=8.0)
        )
        products = run(self.engine.search("API"))
        order = run(self.engine.buy(products[0]))  # $10 > $8 threshold
        assert order.status == OrderStatus.PENDING_APPROVAL

    def test_unavailable_product_raises(self):
        p = Product(id="dead", name="Gone", price=5.0, available=False)
        with pytest.raises(ValueError, match="not available"):
            run(self.engine.buy(p))


class TestProduct:
    def test_product_creation(self):
        p = Product(id="p1", name="Test", price=9.99)
        assert p.id == "p1"
        assert p.price == 9.99
        assert p.available is True

    def test_product_with_all_fields(self):
        p = Product(
            id="p2", name="Full", price=19.99,
            currency="eur", description="A thing",
            url="https://example.com", merchant="TestCo",
        )
        assert p.merchant == "TestCo"
        assert p.currency == "eur"
