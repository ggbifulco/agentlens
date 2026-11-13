"""Type aliases shared by the replay modules (re-exported by :mod:`agentlens.replay`)."""

from __future__ import annotations

from typing import Literal

__all__ = ["CassetteMode", "EntryKind"]

CassetteMode = Literal["record", "strict", "lenient", "fallthrough"]
"""How a cassette answers calls; see :mod:`agentlens.replay.cassette`."""

EntryKind = Literal["llm", "tool"]
"""The two kinds of recorded calls. Each kind has its own index sequence."""
