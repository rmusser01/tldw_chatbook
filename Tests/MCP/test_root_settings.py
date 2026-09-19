"""Admitted root writes outlive observers and preserve configuration truth."""

import asyncio
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from tldw_chatbook.MCP.local_config_saves import (
    ConfigSaveRequest,
    ConfigSaveResult,
    ConfigSaveUnavailable,
    MCPLocalConfigSaves,
    get_mcp_local_config_saves,
)


def request_for(root, revision=0, config=Path("/profile/config.toml")):
    return ConfigSaveRequest(root, (None, revision), config, Path("/captured"))


@pytest.mark.asyncio
async def test_cancelled_observers_and_drain_leave_fifo_writes_owned():
    owner = MCPLocalConfigSaves()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def first():
        calls.append("first")
        entered.set()
        assert release.wait(5)
        return ConfigSaveResult("saved", "/first", True)

    async def observe(task):
        return await asyncio.shield(task)

    task = owner.submit(request_for("/first"), first)
    assert owner.submit(request_for("/first"), first) is task
    observer = asyncio.create_task(observe(task))
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        second = owner.submit(
            request_for("/second", 1),
            lambda: (
                calls.append("second") or ConfigSaveResult("saved", "/second", True)
            ),
        )
        observer.cancel()
        with pytest.raises(asyncio.CancelledError):
            await observer
        drain = asyncio.create_task(owner.close_and_drain())
        await asyncio.sleep(0)
        assert not drain.done() and calls == ["first"]
        with pytest.raises(ConfigSaveUnavailable):
            owner.submit(request_for("/third", 2), first)
        drain.cancel()
        with pytest.raises(asyncio.CancelledError):
            await drain
        assert not task.done() and not second.done()
    finally:
        release.set()
        await owner.close_and_drain()
    assert calls == ["first", "second"]
    assert owner.state == second.result()
    assert owner.state.result.stored == "/second"


@pytest.mark.asyncio
async def test_cache_warning_survives_noop_and_failed_retry_until_reload():
    owner = MCPLocalConfigSaves()
    request = request_for("/committed")
    await owner.submit(request, lambda: ConfigSaveResult("cache_refresh", "/committed"))
    assert owner.known_root(request.config_path, "/stale") == "/committed"
    await owner.submit(request, lambda: ConfigSaveResult("saved", "/committed"))
    assert owner.state.result.phase == "cache_refresh"
    await owner.submit(request, lambda: ConfigSaveResult("failed"))
    assert owner.known_root(request.config_path, "/stale") == "/committed"
    assert owner.known_root(Path("/other/config.toml"), "/other") == "/other"
    await owner.submit(request, lambda: ConfigSaveResult("saved", "/new", True))
    assert owner.state.result.phase == "saved"
    assert owner.known_root(request.config_path, "/new") == "/new"


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("private detail"), None])
async def test_writer_errors_are_bounded_and_do_not_poison_following_saves(failure):
    owner = MCPLocalConfigSaves()

    def fail():
        if failure:
            raise failure

    failed = await owner.submit(request_for("/first"), fail)
    assert failed.result == ConfigSaveResult("failed")
    saved = await owner.submit(
        request_for("/second", 1), lambda: ConfigSaveResult("saved", "/second", True)
    )
    assert owner.state == saved and saved.result.phase == "saved"


def test_shutdown_fence_prevents_lazy_owner_construction():
    host = SimpleNamespace(_mcp_local_config_saves_closed=True)
    with pytest.raises(ConfigSaveUnavailable):
        get_mcp_local_config_saves(host)
    assert not hasattr(host, "_mcp_local_config_saves")


@pytest.mark.asyncio
async def test_later_config_publication_supersedes_old_cache_warning():
    owner = MCPLocalConfigSaves()
    request = request_for("/old")
    await owner.submit(
        request, lambda: ConfigSaveResult("cache_refresh", "/old", cache_generation=1)
    )
    assert owner.known_root(request.config_path, "/stale", 1) == "/old"
    assert owner.known_root(request.config_path, "/new", 2) == "/new"
    await owner.submit(
        request_for("/new", 1),
        lambda: ConfigSaveResult("saved", "/new", cache_generation=2),
    )
    assert owner.state.result.phase == "saved"
