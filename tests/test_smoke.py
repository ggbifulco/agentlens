"""Smoke tests: the package is importable and exposes a valid version."""

import re

import agentlens


def test_version_is_exposed() -> None:
    assert isinstance(agentlens.__version__, str)


def test_version_is_pep440_like() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:[-.]?\w+)?", agentlens.__version__)
