"""Judge result caches (FR-E4): the same judge model, prompt and output give the same verdict.

A :class:`JudgeCache` maps a key (:func:`judge_cache_key`, a SHA-256 hex digest) to
a small JSON object — the judge's raw reply. Two implementations ship:

* :class:`MemoryJudgeCache` — a dict, for one process (tests, notebooks);
* :class:`DiskJudgeCache` — one JSON file per key under a directory (default
  ``.agentlens/judge_cache/``), so re-running a suite or a CI job with a restored
  cache costs nothing and gives identical scores.

:func:`resolve_cache` turns the ``cache`` parameter of a judge (``None``, ``True``,
``False``, ``"memory"``, ``"off"``, a path or a cache object) into a cache; the
default comes from ``$AGENTLENS_JUDGE_CACHE`` and is the on-disk cache.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Protocol, TypeAlias, runtime_checkable

from pydantic import JsonValue

__all__ = [
    "DEFAULT_JUDGE_CACHE_DIR",
    "JUDGE_CACHE_ENV",
    "CacheSpec",
    "DiskJudgeCache",
    "JudgeCache",
    "MemoryJudgeCache",
    "judge_cache_key",
    "resolve_cache",
    "shared_memory_cache",
]

JUDGE_CACHE_ENV = "AGENTLENS_JUDGE_CACHE"
"""Default judge cache: unset = on-disk at :data:`DEFAULT_JUDGE_CACHE_DIR`, ``off`` = no
cache, ``memory`` = the process-wide :class:`MemoryJudgeCache`, anything else = a directory."""

DEFAULT_JUDGE_CACHE_DIR = Path(".agentlens") / "judge_cache"
"""Where :class:`DiskJudgeCache` writes by default (next to the SQLite store)."""

_OFF = frozenset({"off", "none", "false", "0", "no", "disabled"})
_MEMORY = frozenset({"memory", "mem"})

CacheEntry: TypeAlias = dict[str, JsonValue]


@runtime_checkable
class JudgeCache(Protocol):
    """Where judge replies are kept between calls. Implementations must be thread-safe."""

    def get(self, key: str) -> CacheEntry | None:
        """The entry stored under ``key``, or ``None`` on a miss."""
        ...

    def set(self, key: str, value: CacheEntry) -> None:
        """Store ``value`` (a JSON object) under ``key``, replacing any previous entry."""
        ...


def judge_cache_key(judge_model: str, prompt: str, output: str) -> str:
    """SHA-256 hex digest of (judge model, rendered prompt, output text) (``docs/design.md`` §5).

    The three parts are JSON-encoded as a list before hashing, so no choice of
    strings can make two different triples collide by concatenation.
    """
    payload = json.dumps([judge_model, prompt, output], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class MemoryJudgeCache:
    """An in-process cache (a dict guarded by a lock). Lost when the process exits."""

    def __init__(self) -> None:
        self._data: dict[str, CacheEntry] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> CacheEntry | None:
        """The entry for ``key`` (a copy), or ``None``."""
        with self._lock:
            entry = self._data.get(key)
        return dict(entry) if entry is not None else None

    def set(self, key: str, value: CacheEntry) -> None:
        """Store a copy of ``value`` under ``key``."""
        with self._lock:
            self._data[key] = dict(value)

    def clear(self) -> None:
        """Forget every entry."""
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __repr__(self) -> str:
        return f"MemoryJudgeCache({len(self)} entries)"


class DiskJudgeCache:
    """One JSON file per key: ``<directory>/<key[:2]>/<key>.json``.

    Files are written atomically (temp file + :func:`os.replace`), so concurrent
    runs never read half a file; an unreadable or corrupt file counts as a miss.
    The directory is created on the first write. Plain files make the cache easy
    to inspect, delete selectively, and save/restore between CI jobs.
    """

    def __init__(self, directory: str | os.PathLike[str] = DEFAULT_JUDGE_CACHE_DIR) -> None:
        self.directory = Path(directory)

    def path_for(self, key: str) -> Path:
        """The file that holds ``key``."""
        if not key or not all(c in "0123456789abcdef" for c in key):
            raise ValueError(f"judge cache keys are lowercase hex digests, got {key!r}")
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> CacheEntry | None:
        """The entry for ``key``, or ``None`` if missing or unreadable."""
        try:
            data = json.loads(self.path_for(key).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None

    def set(self, key: str, value: CacheEntry) -> None:
        """Write ``value`` to the key's file atomically."""
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            os.replace(tmp, path)
        except BaseException:
            with contextlib.suppress(OSError):
                os.unlink(tmp)
            raise

    def __repr__(self) -> str:
        return f"DiskJudgeCache({str(self.directory)!r})"


_shared_memory = MemoryJudgeCache()


def shared_memory_cache() -> MemoryJudgeCache:
    """The process-wide :class:`MemoryJudgeCache` used for ``cache="memory"``."""
    return _shared_memory


CacheSpec: TypeAlias = JudgeCache | bool | str | os.PathLike[str] | None
"""What a judge's ``cache`` parameter accepts (see :func:`resolve_cache`)."""


def _from_text(text: str) -> JudgeCache | None:
    word = text.strip().lower()
    if word in _OFF:
        return None
    if word in _MEMORY:
        return _shared_memory
    return DiskJudgeCache(text.strip())


def resolve_cache(cache: CacheSpec = None) -> JudgeCache | None:
    """Turn a judge's ``cache`` parameter into a cache (``None`` = no caching).

    * a :class:`JudgeCache` — used as is;
    * ``None`` or ``True`` — the default from ``$AGENTLENS_JUDGE_CACHE``
      (unset/empty: ``DiskJudgeCache(".agentlens/judge_cache")``);
    * ``False`` — no caching;
    * a string: ``"off"`` (or ``none``/``false``/``0``) — no caching, ``"memory"`` —
      the process-wide :class:`MemoryJudgeCache`, anything else — a directory for
      :class:`DiskJudgeCache`; an :class:`os.PathLike` is always a directory.

    Raises:
        TypeError: for any other value.
    """
    if isinstance(cache, bool) or cache is None:
        if cache is False:
            return None
        env = os.environ.get(JUDGE_CACHE_ENV, "").strip()
        return _from_text(env) if env else DiskJudgeCache()
    if isinstance(cache, str):
        if not cache.strip():
            raise TypeError("cache must not be an empty string")
        return _from_text(cache)
    if isinstance(cache, os.PathLike):
        return DiskJudgeCache(cache)
    if isinstance(cache, JudgeCache):
        return cache
    raise TypeError(
        "cache must be a JudgeCache, True/False/None, 'memory', 'off' or a directory path, "
        f"got {type(cache).__name__}"
    )
