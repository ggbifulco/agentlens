"""The support desk's in-memory "database": customers, orders and FAQ articles.

Everything is fixed, including "today" (:data:`TODAY`), so the tools give the
same answer on every run. :func:`new_database` returns a fresh deep copy for
each agent run: tools that write (refunds, address changes) never leak state
into another run, also when the suite runs cases concurrently.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import date
from typing import Any

__all__ = ["FAQ", "TODAY", "Database", "FaqArticle", "new_database"]

TODAY = date(2026, 11, 1)
"""The date the support desk lives in (keeps the refund window deterministic)."""


@dataclass(frozen=True)
class FaqArticle:
    """One help-centre article found by ``search_faq``."""

    id: str
    title: str
    keywords: tuple[str, ...]
    answer: str


FAQ: tuple[FaqArticle, ...] = (
    FaqArticle(
        "returns-policy",
        "Return and refund policy",
        ("return", "returns", "refund policy", "money back", "send back"),
        "You can return any delivered item within 30 days of delivery for a full refund. "
        "Refunds go back to the original payment method within 5 business days.",
    ),
    FaqArticle(
        "shipping-times",
        "Shipping times",
        ("shipping", "delivery time", "deliver", "ship"),
        "Standard shipping takes 3-5 business days; express shipping takes 1-2 business days.",
    ),
    FaqArticle(
        "international-shipping",
        "International shipping",
        ("international", "abroad", "canada", "usa", "uk", "outside", "country", "countries"),
        "We ship to the EU, the UK, the USA and Canada. International delivery takes "
        "7-12 business days and customs fees are paid by the recipient.",
    ),
    FaqArticle(
        "payment-methods",
        "Payment methods",
        ("payment", "pay", "paypal", "credit card", "card", "apple pay"),
        "We accept Visa, Mastercard, American Express, PayPal and Apple Pay.",
    ),
    FaqArticle(
        "warranty",
        "Warranty",
        ("warranty", "guarantee", "broken after", "defect"),
        "All electronics come with a 2-year warranty covering manufacturing defects.",
    ),
    FaqArticle(
        "cancel-order",
        "Cancelling an order",
        ("cancel", "cancellation"),
        "Orders can be cancelled free of charge until they ship. "
        "Shipped orders can be returned after delivery instead.",
    ),
    FaqArticle(
        "gift-cards",
        "Gift cards",
        ("gift card", "gift cards", "voucher"),
        "Gift cards are valid for 3 years and can be used for any product in the store.",
    ),
    FaqArticle(
        "contact-hours",
        "Contact and opening hours",
        ("hours", "opening", "open", "contact", "phone", "email"),
        "Our support team is available Monday to Friday, 9:00-18:00 CET, at support@example.com.",
    ),
)
"""The help-centre articles, searched by keyword overlap."""

_CUSTOMERS: dict[str, dict[str, Any]] = {
    "C001": {"id": "C001", "name": "Alice Rossi", "email": "alice@example.com", "tier": "gold"},
    "C002": {"id": "C002", "name": "Bob Smith", "email": "bob@example.com", "tier": "standard"},
    "C003": {
        "id": "C003",
        "name": "Chiara Bianchi",
        "email": "chiara@example.com",
        "tier": "standard",
    },
    "C004": {"id": "C004", "name": "Dan Lee", "email": "dan@example.com", "tier": "standard"},
}

_ORDERS: dict[str, dict[str, Any]] = {
    "1001": {
        "id": "1001",
        "customer_id": "C001",
        "items": [{"name": "Wireless mouse", "quantity": 1, "price": 49.99}],
        "total": 49.99,
        "status": "delivered",
        "delivered_on": "2026-10-20",
        "shipping_address": "8 Via Toledo, Napoli",
        "carrier": "BRT",
        "tracking_number": "BRT-55120",
        "refunded": False,
    },
    "1002": {
        "id": "1002",
        "customer_id": "C002",
        "items": [{"name": "Mechanical keyboard", "quantity": 1, "price": 129.00}],
        "total": 129.00,
        "status": "shipped",
        "eta": "2026-11-04",
        "shipping_address": "21 Baker Street, London",
        "carrier": "UPS",
        "tracking_number": "1Z999AA10123456784",
        "refunded": False,
    },
    "1003": {
        "id": "1003",
        "customer_id": "C003",
        "items": [{"name": "USB-C cable", "quantity": 2, "price": 17.75}],
        "total": 35.50,
        "status": "processing",
        "shipping_address": "5 Via Roma, Milano",
        "refunded": False,
    },
    "1004": {
        "id": "1004",
        "customer_id": "C001",
        "items": [{"name": "Laptop Pro 14", "quantity": 1, "price": 899.00}],
        "total": 899.00,
        "status": "delivered",
        "delivered_on": "2026-10-28",
        "shipping_address": "8 Via Toledo, Napoli",
        "carrier": "DHL",
        "tracking_number": "DHL-7731002",
        "refunded": False,
    },
    "1005": {
        "id": "1005",
        "customer_id": "C004",
        "items": [{"name": "Phone case", "quantity": 1, "price": 25.00}],
        "total": 25.00,
        "status": "delivered",
        "delivered_on": "2026-08-15",
        "shipping_address": "3 Rue Cler, Paris",
        "carrier": "La Poste",
        "tracking_number": "LP-88001",
        "refunded": False,
    },
    "1006": {
        "id": "1006",
        "customer_id": "C002",
        "items": [{"name": "Noise-cancelling headphones", "quantity": 1, "price": 189.00}],
        "total": 189.00,
        "status": "delayed",
        "eta": "2026-11-06",
        "shipping_address": "21 Baker Street, London",
        "carrier": "DHL",
        "tracking_number": "DHL-7731006",
        "refunded": False,
    },
    "1007": {
        "id": "1007",
        "customer_id": "C003",
        "items": [{"name": "Screen protector", "quantity": 1, "price": 15.00}],
        "total": 15.00,
        "status": "delivered",
        "delivered_on": "2026-10-25",
        "shipping_address": "5 Via Roma, Milano",
        "carrier": "BRT",
        "tracking_number": "BRT-55177",
        "refunded": True,
    },
    "1008": {
        "id": "1008",
        "customer_id": "C004",
        "items": [{"name": "Smart watch", "quantity": 1, "price": 249.00}],
        "total": 249.00,
        "status": "cancelled",
        "shipping_address": "3 Rue Cler, Paris",
        "refunded": True,
    },
    "1009": {
        "id": "1009",
        "customer_id": "C001",
        "items": [
            {"name": "Desk lamp", "quantity": 1, "price": 39.90},
            {"name": "LED bulb", "quantity": 2, "price": 16.25},
        ],
        "total": 72.40,
        "status": "processing",
        "shipping_address": "8 Via Toledo, Napoli",
        "refunded": False,
    },
}


@dataclass
class Database:
    """One run's private copy of the support desk's data."""

    customers: dict[str, dict[str, Any]] = field(default_factory=dict)
    orders: dict[str, dict[str, Any]] = field(default_factory=dict)
    refunds: list[dict[str, Any]] = field(default_factory=list)
    tickets: list[dict[str, Any]] = field(default_factory=list)


def new_database() -> Database:
    """A fresh copy of the fixed data (writes stay inside one run)."""
    return Database(customers=copy.deepcopy(_CUSTOMERS), orders=copy.deepcopy(_ORDERS))
