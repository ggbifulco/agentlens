"""Judge caches: keys, in-memory and on-disk storage, and the ``cache`` parameter."""

from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from agentlens.evals.judge import (
    DEFAULT_JUDGE_CACHE_DIR,
    JUDGE_CACHE_ENV,
    DiskJudgeCache,
    JudgeCache,
    MemoryJudgeCache,
    judge_cache_key,
    resolve_cache,
    shared_memory_cache,
)

KEY = judge_cache_key("openai:gpt-4o", "prompt", "output")


def test_key_is_a_sha256_hex_digest_and_deterministic() -> None:
    assert len(KEY) == 64
    assert all(c in "0123456789abcdef" for c in KEY)
    assert judge_cache_key("openai:gpt-4o", "prompt", "output") == KEY


@pytest.mark.parametrize(
    "parts",
    [
        ("openai:gpt-4o-mini", "prompt", "output"),
        ("openai:gpt-4o", "prompt!", "output"),
        ("openai:gpt-4o", "prompt", "output!"),
        ("openai:gpt-4opr", "ompt", "output"),  # no concatenation collisions
    ],
)
def test_key_changes_with_every_part(parts: tuple[str, str, str]) -> None:
    assert judge_cache_key(*parts) != KEY


def test_memory_cache_roundtrip() -> None:
    cache = MemoryJudgeCache()
    assert isinstance(cache, JudgeCache)
    assert cache.get(KEY) is None
    entry = {"reply": '{"score": 4}'}
    cache.set(KEY, entry)
    entry["reply"] = "mutated"
    assert cache.get(KEY) == {"reply": '{"score": 4}'}
    got = cache.get(KEY)
    assert got is not None
    got["reply"] = "mutated too"
    assert cache.get(KEY) == {"reply": '{"score": 4}'}
    assert len(cache) == 1
    assert repr(cache) == "MemoryJudgeCache(1 entries)"
    cache.clear()
    assert len(cache) == 0


def test_disk_cache_roundtrip(tmp_path: Path) -> None:
    cache = DiskJudgeCache(tmp_path / "jc")
    assert isinstance(cache, JudgeCache)
    assert cache.get(KEY) is None
    assert not (tmp_path / "jc").exists()  # nothing created until the first write
    cache.set(KEY, {"reply": "é", "n": 1})
    path = tmp_path / "jc" / KEY[:2] / f"{KEY}.json"
    assert cache.path_for(KEY) == path
    assert path.is_file()
    assert cache.get(KEY) == {"reply": "é", "n": 1}
    # persistent: a new instance on the same directory sees it
    assert DiskJudgeCache(str(tmp_path / "jc")).get(KEY) == {"reply": "é", "n": 1}
    cache.set(KEY, {"reply": "new"})
    assert cache.get(KEY) == {"reply": "new"}
    assert [p.name for p in path.parent.iterdir()] == [path.name]  # no temp files left
    assert repr(cache) == f"DiskJudgeCache({str(tmp_path / 'jc')!r})"


def test_disk_cache_default_directory() -> None:
    assert DiskJudgeCache().directory == DEFAULT_JUDGE_CACHE_DIR == Path(".agentlens/judge_cache")


@pytest.mark.parametrize("content", ["{not json", "[1, 2]"])
def test_disk_cache_corrupt_file_is_a_miss(tmp_path: Path, content: str) -> None:
    cache = DiskJudgeCache(tmp_path)
    path = cache.path_for(KEY)
    path.parent.mkdir(parents=True)
    path.write_text(content, encoding="utf-8")
    assert cache.get(KEY) is None


@pytest.mark.parametrize("key", ["", "../etc/passwd", "ABC", "a/b"])
def test_disk_cache_rejects_non_hex_keys(tmp_path: Path, key: str) -> None:
    with pytest.raises(ValueError, match="hex"):
        DiskJudgeCache(tmp_path).path_for(key)


def test_disk_cache_failed_write_leaves_no_temp_file(tmp_path: Path) -> None:
    cache = DiskJudgeCache(tmp_path)
    with pytest.raises(TypeError):
        cache.set(KEY, {"bad": object()})  # type: ignore[dict-item]
    assert list(cache.path_for(KEY).parent.iterdir()) == []


def test_disk_cache_concurrent_writes(tmp_path: Path) -> None:
    cache = DiskJudgeCache(tmp_path)

    def write(i: int) -> None:
        cache.set(KEY, {"reply": str(i)})

    threads = [threading.Thread(target=write, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    entry = cache.get(KEY)
    assert entry is not None
    assert entry["reply"] in {str(i) for i in range(16)}


# -- resolve_cache -------------------------------------------------------------


def test_resolve_default_is_disk_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(JUDGE_CACHE_ENV, raising=False)
    for value in (None, True):
        cache = resolve_cache(value)
        assert isinstance(cache, DiskJudgeCache)
        assert cache.directory == DEFAULT_JUDGE_CACHE_DIR


@pytest.mark.parametrize("word", ["off", "OFF", "none", "false", "0", "no", "disabled"])
def test_resolve_env_off(monkeypatch: pytest.MonkeyPatch, word: str) -> None:
    monkeypatch.setenv(JUDGE_CACHE_ENV, word)
    assert resolve_cache(None) is None


def test_resolve_env_memory_and_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(JUDGE_CACHE_ENV, "memory")
    assert resolve_cache() is shared_memory_cache()
    monkeypatch.setenv(JUDGE_CACHE_ENV, f" {tmp_path} ")
    cache = resolve_cache(True)
    assert isinstance(cache, DiskJudgeCache)
    assert cache.directory == tmp_path
    monkeypatch.setenv(JUDGE_CACHE_ENV, "  ")
    assert isinstance(resolve_cache(), DiskJudgeCache)


def test_resolve_explicit_values(tmp_path: Path) -> None:
    assert resolve_cache(False) is None
    assert resolve_cache("off") is None
    assert resolve_cache("Memory") is shared_memory_cache()
    by_str = resolve_cache(str(tmp_path))
    assert isinstance(by_str, DiskJudgeCache)
    assert by_str.directory == tmp_path
    by_path = resolve_cache(tmp_path / "off")  # a PathLike is always a directory
    assert isinstance(by_path, DiskJudgeCache)
    assert by_path.directory == tmp_path / "off"
    mine = MemoryJudgeCache()
    assert resolve_cache(mine) is mine


@pytest.mark.parametrize("value", ["", "  ", 3, ["memory"], object()])
def test_resolve_rejects(value: object) -> None:
    with pytest.raises(TypeError):
        resolve_cache(value)  # type: ignore[arg-type]


def test_os_pathlike_subclass(tmp_path: Path) -> None:
    class P(os.PathLike[str]):
        def __fspath__(self) -> str:
            return str(tmp_path)

    cache = resolve_cache(P())
    assert isinstance(cache, DiskJudgeCache)
    assert cache.directory == tmp_path
