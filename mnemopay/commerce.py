"""
CommerceEngine — Autonomous shopping for AI agents.

Port of the TypeScript CommerceEngine. Provides:
  - Product search via configurable providers
  - Shopping cart with escrow-based checkout
  - Spending mandates with approval thresholds
  - Human-in-the-loop approval flow
  - Order tracking and delivery confirmation

Zero external dependencies (stdlib only).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


# ── Types ─────────────────────────────────────────────────────────────────


class OrderStatus(Enum):
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    ESCROW_FUNDED = "escrow_funded"
    PURCHASED = "purchased"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass
class Product:
    id: str
    name: str
    price: float
    currency: str = "usd"
    description: str = ""
    url: str = ""
    merchant: str = ""
    available: bool = True


@dataclass
class Order:
    id: str
    agent_id: str
    product: Product
    quantity: int
    total: float
    status: OrderStatus
    created_at: datetime
    approved_at: Optional[datetime] = None
    purchased_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    escrow_tx_id: Optional[str] = None
    rejection_reason: Optional[str] = None


@dataclass
class Mandate:
    """Spending mandate — defines what the agent can spend without approval."""
    max_single: float = 0.0       # Max single purchase without approval
    max_daily: float = 0.0        # Max daily spend without approval
    allowed_merchants: List[str] = field(default_factory=list)
    blocked_categories: List[str] = field(default_factory=list)
    require_approval_above: float = 0.0  # Always require approval above this


# ── Commerce Provider Interface ──────────────────────────────────────────


class CommerceProvider:
    """Base class for commerce providers (mock, Firecrawl, Shopify, etc.)."""

    async def search(self, query: str, limit: int = 10) -> List[Product]:
        raise NotImplementedError

    async def get_product(self, product_id: str) -> Optional[Product]:
        raise NotImplementedError

    async def execute_purchase(self, product: Product, quantity: int) -> bool:
        raise NotImplementedError


class MockProvider(CommerceProvider):
    """In-memory mock provider for testing."""

    def __init__(self) -> None:
        self._catalog: List[Product] = [
            Product(id="mock-1", name="API Credits (1000)", price=10.00, merchant="MockMart"),
            Product(id="mock-2", name="Cloud Storage (1TB/mo)", price=5.00, merchant="MockMart"),
            Product(id="mock-3", name="SSL Certificate", price=8.00, merchant="MockMart"),
            Product(id="mock-4", name="Domain Registration", price=12.00, merchant="MockMart"),
            Product(id="mock-5", name="Email Service (1000/mo)", price=3.00, merchant="MockMart"),
        ]

    async def search(self, query: str, limit: int = 10) -> List[Product]:
        q = query.lower()
        results = [p for p in self._catalog if q in p.name.lower() or q in p.description.lower()]
        if not results:
            results = self._catalog[:limit]
        return results[:limit]

    async def get_product(self, product_id: str) -> Optional[Product]:
        for p in self._catalog:
            if p.id == product_id:
                return p
        return None

    async def execute_purchase(self, product: Product, quantity: int) -> bool:
        return product.available


# ── Commerce Engine ──────────────────────────────────────────────────────


ApprovalCallback = Callable[[Order], bool]


class CommerceEngine:
    """Autonomous shopping engine with escrow and approval flow.

    Usage:
        from mnemopay import MnemoPay
        from mnemopay.commerce import CommerceEngine, MockProvider

        agent = MnemoPay("shop-agent")
        engine = CommerceEngine(agent, MockProvider())
        engine.set_mandate(Mandate(max_single=25.0, max_daily=100.0))

        products = await engine.search("API credits")
        order = await engine.buy(products[0])
        # If under mandate → auto-approved
        # If over mandate → pending_approval
    """

    MAX_ORDERS = 10_000
    MAX_DAILY_ORDERS = 100

    def __init__(
        self,
        agent: Any,  # MnemoPay instance
        provider: Optional[CommerceProvider] = None,
        approval_callback: Optional[ApprovalCallback] = None,
    ) -> None:
        self._agent = agent
        self._provider = provider or MockProvider()
        self._approval_callback = approval_callback
        self._mandate: Mandate = Mandate()
        self._orders: Dict[str, Order] = {}
        self._daily_spend: float = 0.0
        self._daily_reset: datetime = datetime.now(timezone.utc)
        self._pending_approvals: Dict[str, Order] = {}

    def set_mandate(self, mandate: Mandate) -> None:
        """Set spending mandate for autonomous purchases."""
        if mandate.max_single < 0 or mandate.max_daily < 0:
            raise ValueError("Mandate limits must be non-negative")
        if mandate.require_approval_above < 0:
            raise ValueError("Approval threshold must be non-negative")
        self._mandate = mandate

    async def search(self, query: str, limit: int = 10) -> List[Product]:
        """Search for products."""
        if not query or not isinstance(query, str):
            raise ValueError("Search query is required")
        if len(query) > 500:
            raise ValueError("Query too long (max 500 chars)")
        limit = min(max(limit, 1), 50)
        return await self._provider.search(query, limit)

    async def buy(
        self,
        product: Product,
        quantity: int = 1,
    ) -> Order:
        """Initiate a purchase. May require approval based on mandate.

        Returns:
            Order in either APPROVED or PENDING_APPROVAL status.
        """
        if quantity < 1 or quantity > 100:
            raise ValueError("Quantity must be 1-100")
        if not product.available:
            raise ValueError(f"Product {product.id} is not available")
        if len(self._orders) >= self.MAX_ORDERS:
            raise ValueError("Order limit reached")

        total = round(product.price * quantity * 100) / 100
        if total <= 0:
            raise ValueError("Order total must be positive")

        # Reset daily spend if new day
        self._check_daily_reset()

        order = Order(
            id=str(uuid.uuid4()),
            agent_id=self._agent.agent_id,
            product=product,
            quantity=quantity,
            total=total,
            status=OrderStatus.PENDING_APPROVAL,
            created_at=datetime.now(timezone.utc),
        )

        # Check mandate
        needs_approval = self._needs_approval(order)

        if not needs_approval:
            order.status = OrderStatus.APPROVED
            order.approved_at = datetime.now(timezone.utc)
            self._daily_spend += total
            # Auto-execute
            await self._execute_order(order)
        else:
            self._pending_approvals[order.id] = order

        self._orders[order.id] = order
        return order

    def approve(self, order_id: str) -> Order:
        """Approve a pending order."""
        order = self._pending_approvals.get(order_id)
        if order is None:
            raise ValueError(f"No pending approval for order {order_id}")
        if order.status != OrderStatus.PENDING_APPROVAL:
            raise ValueError(f"Order {order_id} is not pending approval")

        order.status = OrderStatus.APPROVED
        order.approved_at = datetime.now(timezone.utc)
        self._daily_spend += order.total
        del self._pending_approvals[order_id]
        return order

    def reject(self, order_id: str, reason: str = "") -> Order:
        """Reject a pending order."""
        order = self._pending_approvals.get(order_id)
        if order is None:
            raise ValueError(f"No pending approval for order {order_id}")

        order.status = OrderStatus.REJECTED
        order.rejection_reason = reason
        del self._pending_approvals[order_id]
        return order

    def confirm_delivery(self, order_id: str) -> Order:
        """Confirm delivery of a purchased order."""
        order = self._orders.get(order_id)
        if order is None:
            raise ValueError(f"Order {order_id} not found")
        if order.status != OrderStatus.PURCHASED:
            raise ValueError(f"Order {order_id} is {order.status.value}, not purchased")

        if order.escrow_tx_id:
            self._agent.settle(order.escrow_tx_id)

        order.status = OrderStatus.DELIVERED
        order.delivered_at = datetime.now(timezone.utc)

        return order

    def pending_approvals(self) -> List[Order]:
        """List orders waiting for approval."""
        return list(self._pending_approvals.values())

    def orders(self, limit: int = 50) -> List[Order]:
        """List recent orders."""
        all_orders = sorted(
            self._orders.values(),
            key=lambda o: o.created_at.timestamp(),
            reverse=True,
        )
        return all_orders[:limit]

    # ── Internal ─────────────────────────────────────────────────────────

    def _needs_approval(self, order: Order) -> bool:
        m = self._mandate

        # Always approve if mandate says zero (everything needs approval)
        if m.max_single <= 0 and m.max_daily <= 0:
            return True

        # Check approval threshold
        if m.require_approval_above > 0 and order.total > m.require_approval_above:
            return True

        # Check single purchase limit
        if order.total > m.max_single:
            return True

        # Check daily limit
        if self._daily_spend + order.total > m.max_daily:
            return True

        # Check blocked merchants
        merchant = order.product.merchant.lower()
        if m.blocked_categories and merchant in [c.lower() for c in m.blocked_categories]:
            return True

        return False

    async def _execute_order(self, order: Order) -> None:
        """Execute an approved order — create escrow charge + purchase."""
        # Create escrow charge via MnemoPay
        try:
            tx = self._agent.charge(
                order.total,
                f"Purchase: {order.product.name} x{order.quantity}",
            )
            order.escrow_tx_id = tx.id
            order.status = OrderStatus.ESCROW_FUNDED
        except ValueError:
            order.status = OrderStatus.CANCELLED
            order.rejection_reason = "Insufficient reputation for charge"
            return

        # Execute purchase via provider
        try:
            success = await self._provider.execute_purchase(order.product, order.quantity)
            if success:
                order.status = OrderStatus.PURCHASED
                order.purchased_at = datetime.now(timezone.utc)
            else:
                order.status = OrderStatus.CANCELLED
                order.rejection_reason = "Purchase failed at provider"
                # Refund escrow
                if order.escrow_tx_id:
                    try:
                        self._agent.refund(order.escrow_tx_id, "Purchase failed")
                    except Exception:
                        pass
        except Exception as e:
            order.status = OrderStatus.CANCELLED
            order.rejection_reason = f"Provider error: {str(e)[:200]}"

    def _check_daily_reset(self) -> None:
        now = datetime.now(timezone.utc)
        if now.date() > self._daily_reset.date():
            self._daily_spend = 0.0
            self._daily_reset = now
