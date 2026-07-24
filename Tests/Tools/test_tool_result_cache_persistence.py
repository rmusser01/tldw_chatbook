from __future__ import annotations

import inspect
import json
import os
import pickle
import stat
from pathlib import Path

import pytest

from tldw_chatbook.Tools import tool_executor
from tldw_chatbook.Tools.tool_executor import (
    TOOL_CACHE_MAX_BYTES,
    ToolResultCache,
)


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


async def _finish_load(cache: ToolResultCache) -> None:
    if cache._load_task is not None:
        await cache._load_task


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
async def test_valid_json_cache_roundtrips_with_private_modes(tmp_path):
    cache_path = tmp_path / "tool-cache" / "tool_results.cache"
    first = ToolResultCache(max_size=3, persist_path=cache_path)
    await _finish_load(first)

    await first.set("search", {"query": "needle"}, {"items": [1, 2]})

    envelope = json.loads(cache_path.read_text(encoding="utf-8"))
    assert envelope["version"] == 1
    assert len(envelope["entries"]) == 1
    assert _mode(cache_path.parent) == 0o700
    assert _mode(cache_path) == 0o600

    second = ToolResultCache(max_size=3, persist_path=cache_path)
    assert await second.get("search", {"query": "needle"}) == {"items": [1, 2]}


class _UnsupportedResult:
    pass


@pytest.mark.asyncio
async def test_unsupported_result_remains_memory_only(tmp_path):
    cache_path = tmp_path / "tool-cache" / "tool_results.cache"
    cache = ToolResultCache(max_size=3, persist_path=cache_path)
    await _finish_load(cache)
    result = _UnsupportedResult()

    await cache.set("custom", {}, result)

    assert await cache.get("custom", {}) is result
    envelope = json.loads(cache_path.read_text(encoding="utf-8"))
    assert envelope == {"version": 1, "entries": []}


class _RunOnUnpickle:
    def __init__(self, sentinel: Path) -> None:
        self.sentinel = sentinel

    def __reduce__(self):
        return os.system, (f"touch {self.sentinel}",)


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX mode contract")
async def test_legacy_pickle_is_hardened_left_inert_and_never_deserialized(tmp_path):
    cache_path = tmp_path / "tool_results.cache"
    sentinel = tmp_path / "pickle-executed-SENTINEL"
    legacy = pickle.dumps(_RunOnUnpickle(sentinel))
    cache_path.write_bytes(legacy)
    cache_path.chmod(0o644)

    cache = ToolResultCache(persist_path=cache_path)
    await _finish_load(cache)

    assert not sentinel.exists()
    assert cache.cache == {}
    assert cache_path.read_bytes() == legacy
    assert _mode(cache_path) == 0o600
    source = inspect.getsource(tool_executor)
    assert "import pickle" not in source
    assert "pickle.load" not in source


@pytest.mark.asyncio
async def test_legacy_pickle_is_not_overwritten_by_automatic_persistence(tmp_path):
    cache_path = tmp_path / "tool_results.cache"
    legacy = pickle.dumps({"legacy": ("value", 1)})
    cache_path.write_bytes(legacy)
    cache = ToolResultCache(persist_path=cache_path)
    await _finish_load(cache)

    await cache.set("new", {}, {"ok": True})

    assert await cache.get("new", {}) == {"ok": True}
    assert cache_path.read_bytes() == legacy


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "payload",
    [
        b"{not-json",
        json.dumps({"version": 2, "entries": []}).encode(),
        json.dumps({"version": 1, "entries": "wrong"}).encode(),
        json.dumps(
            {
                "version": 1,
                "entries": [{"key": "not-a-cache-key", "expires_at": 1, "result": {}}],
            }
        ).encode(),
        b"x" * (TOOL_CACHE_MAX_BYTES + 1),
    ],
)
async def test_corrupt_oversized_or_invalid_cache_is_ignored_and_preserved(
    tmp_path,
    payload,
):
    cache_path = tmp_path / "tool_results.cache"
    cache_path.write_bytes(payload)

    cache = ToolResultCache(persist_path=cache_path)
    await _finish_load(cache)

    assert cache.cache == {}
    assert cache_path.read_bytes() == payload


@pytest.mark.asyncio
async def test_nonfinite_and_excessively_deep_results_are_not_persisted(tmp_path):
    cache_path = tmp_path / "tool_results.cache"
    cache = ToolResultCache(max_size=3, persist_path=cache_path)
    await _finish_load(cache)
    deep: object = "leaf"
    for _ in range(30):
        deep = [deep]

    await cache.set("nan", {}, {"value": float("nan")})
    await cache.set("deep", {}, {"value": deep})

    envelope = json.loads(cache_path.read_text(encoding="utf-8"))
    assert envelope == {"version": 1, "entries": []}
    assert await cache.get("nan", {}) is not None
    assert await cache.get("deep", {}) is not None


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "posix", reason="POSIX symlink contract")
async def test_cache_symlink_is_never_followed_and_memory_cache_still_works(tmp_path):
    cache_dir = tmp_path / "tool-cache"
    cache_dir.mkdir()
    outside = tmp_path / "outside-SENTINEL"
    outside.write_text("preserve", encoding="utf-8")
    cache_path = cache_dir / "tool_results.cache"
    cache_path.symlink_to(outside)

    cache = ToolResultCache(persist_path=cache_path)
    await _finish_load(cache)
    await cache.set("search", {}, {"ok": True})

    assert await cache.get("search", {}) == {"ok": True}
    assert outside.read_text(encoding="utf-8") == "preserve"


@pytest.mark.asyncio
async def test_clear_persists_an_empty_versioned_cache(tmp_path):
    cache_path = tmp_path / "tool-cache" / "tool_results.cache"
    cache = ToolResultCache(persist_path=cache_path)
    await _finish_load(cache)
    await cache.set("search", {}, {"ok": True})

    await cache.clear()

    assert cache.cache == {}
    assert json.loads(cache_path.read_text(encoding="utf-8")) == {
        "version": 1,
        "entries": [],
    }
