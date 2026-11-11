"""Shared pytest configuration.

Tests marked ``@pytest.mark.live`` call real LLM providers and are skipped unless
``AGENTLENS_LIVE_TESTS=1`` is set, so the default run never needs API keys.
"""

import os

import pytest


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.environ.get("AGENTLENS_LIVE_TESTS") == "1":
        return
    skip_live = pytest.mark.skip(reason="live test: set AGENTLENS_LIVE_TESTS=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)
