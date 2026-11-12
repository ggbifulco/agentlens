"""Time-ordered identifiers (UUIDv7, RFC 9562).

AgentLens ids sort lexicographically in creation order, which keeps SQLite indexes
compact and makes ``ORDER BY id`` equivalent to ``ORDER BY created``. Python only
ships :func:`uuid.uuid7` from 3.14, so we implement the layout here:

.. code-block:: text

     0                   1                   2                   3
    |  unix_ts_ms (48 bits)                       | ver | counter (12) |
    |var|                 random (62 bits)                            |

The 12-bit ``rand_a`` field is used as a per-process monotonic counter (RFC 9562
section 6.2, method 1), so ids created in the same millisecond still sort in
creation order.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from datetime import UTC, datetime

__all__ = ["id_datetime", "new_id"]

_COUNTER_MAX = 0xFFF  # 12 bits
_clock_ns = time.time_ns  # indirection so tests can freeze the clock
_lock = threading.Lock()
_last_ms = -1
_counter = 0


def _next_timestamp_and_counter() -> tuple[int, int]:
    """Return a ``(unix_ms, counter)`` pair strictly greater than the previous one."""
    global _last_ms, _counter
    now_ms = _clock_ns() // 1_000_000
    with _lock:
        if now_ms > _last_ms:
            _last_ms = now_ms
            # Start from a random value in the lower half so we have headroom to count up.
            _counter = int.from_bytes(os.urandom(2)) & 0x7FF
        else:
            # Same millisecond (or the clock went backwards): keep the old timestamp
            # and bump the counter; on overflow borrow the next millisecond.
            _counter += 1
            if _counter > _COUNTER_MAX:
                _last_ms += 1
                _counter = 0
        return _last_ms, _counter


def new_id() -> str:
    """Return a new UUIDv7 as a canonical 36-character string.

    Ids are unique and monotonically increasing within a process, and roughly
    time-ordered across processes.
    """
    unix_ms, counter = _next_timestamp_and_counter()
    rand_b = int.from_bytes(os.urandom(8)) & ((1 << 62) - 1)
    value = (
        (unix_ms & ((1 << 48) - 1)) << 80
        | 0x7 << 76  # version 7
        | counter << 64
        | 0b10 << 62  # RFC 9562 variant
        | rand_b
    )
    return str(uuid.UUID(int=value))


def id_datetime(id_: str) -> datetime:
    """Return the creation time (UTC, millisecond precision) embedded in a UUIDv7 id.

    Raises:
        ValueError: if ``id_`` is not a valid UUID version 7.
    """
    parsed = uuid.UUID(id_)
    if parsed.version != 7:
        raise ValueError(f"not a UUIDv7: {id_!r}")
    unix_ms = parsed.int >> 80
    return datetime.fromtimestamp(unix_ms / 1000, tz=UTC)
