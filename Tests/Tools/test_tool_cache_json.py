import asyncio
import json

import pytest

from tldw_chatbook.Tools.tool_executor import ToolResultCache


def test_cache_round_trips_through_json(tmp_path):
    persist = tmp_path / "tool_results.cache"

    async def run():
        c1 = ToolResultCache(persist_path=persist)
        await c1.set("mytool", {"a": 1}, {"result": "ok", "n": 3}, ttl=3600)
        await c1._save_to_disk()
        # the on-disk file is valid JSON (not pickle)
        raw = persist.read_text(encoding="utf-8")
        json.loads(raw)  # would raise if pickle
        c2 = ToolResultCache(persist_path=persist)
        await c2._load_from_disk()
        got = await c2.get("mytool", {"a": 1})
        return raw, got

    raw, got = asyncio.run(run())
    assert got == {"result": "ok", "n": 3}
    assert "\x80" not in raw  # not a pickle opcode stream


def test_corrupt_cache_file_degrades_gracefully(tmp_path):
    persist = tmp_path / "tool_results.cache"
    persist.write_text("not valid json {{{", encoding="utf-8")

    async def run():
        c = ToolResultCache(persist_path=persist)
        await c._load_from_disk()  # must not raise
        return await c.get("anything", {})

    assert asyncio.run(run()) is None


def test_get_tool_executor_with_cache_enabled_does_not_importerror(monkeypatch, tmp_path):
    """Regression test: `get_tool_executor()` used to do
    `from ..config import USER_DATA_DIR`, which doesn't exist and raised an
    unguarded ImportError whenever tools.cache_enabled + tools.cache_persist
    resolved true. It must instead resolve the persist directory via
    `get_user_data_dir()`.
    """
    from tldw_chatbook.Tools import tool_executor as te
    import tldw_chatbook.config as cfg

    def fake_get_cli_setting(section, key=None, default=None):
        # get_tool_executor calls get_cli_setting("tools", {}); real
        # get_cli_setting treats a non-string second positional as the
        # default-value slot, so mirror that contract here.
        if section == "tools":
            return {"cache_enabled": True, "cache_persist": True}
        return default

    monkeypatch.setattr(cfg, "get_cli_setting", fake_get_cli_setting)
    monkeypatch.setattr(cfg, "get_user_data_dir", lambda: tmp_path)
    # Force a fresh build regardless of what earlier tests left cached.
    monkeypatch.setattr(te, "_global_executor", None)

    async def run():
        return te.reload_tool_executor()  # must not raise ImportError

    ex = asyncio.run(run())
    try:
        assert ex is not None
        assert ex.cache is not None
        assert ex.cache.persist_path == tmp_path / "tool_cache" / "tool_results.cache"
    finally:
        # Don't leak the monkeypatched executor into later tests.
        monkeypatch.setattr(te, "_global_executor", None)


def test_cache_construct_from_sync_context_does_not_raise(tmp_path):
    """Regression: constructing the cache with a persist_path from a SYNC
    context (no running event loop) must not raise 'no running event loop' --
    the disk load is deferred and started lazily on the first async get/set.
    """
    persist = tmp_path / "tool_results.cache"
    # No running loop here (plain sync call) -- previously raised RuntimeError.
    cache = ToolResultCache(persist_path=persist)
    assert cache._load_task is None  # deferred, not eagerly created

    async def run():
        # First async op triggers the lazy load and round-trips a value.
        await cache.set("t", {"x": 1}, {"ok": True}, ttl=3600)
        return await cache.get("t", {"x": 1})

    assert asyncio.run(run()) == {"ok": True}


def test_cache_lazy_load_reads_prewritten_disk_file(tmp_path):
    """A cache constructed from a sync context still loads an existing on-disk
    file lazily on first async get()."""
    persist = tmp_path / "tool_results.cache"

    async def seed():
        c1 = ToolResultCache(persist_path=persist)
        await c1.set("seed", {"k": 1}, {"v": 42}, ttl=3600)
        await c1._save_to_disk()

    asyncio.run(seed())

    cache = ToolResultCache(persist_path=persist)  # sync construction
    assert cache._load_task is None

    async def run():
        return await cache.get("seed", {"k": 1})

    assert asyncio.run(run()) == {"v": 42}
