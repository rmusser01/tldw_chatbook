"""The two MCP controls share FIFO ownership without sharing receipt authority."""

import asyncio
import threading
from pathlib import Path

import pytest

from tldw_chatbook.MCP.local_config_saves import (
    ConfigSaveRequest,
    ConfigSaveResult,
    MCPLocalConfigSaves,
)

PROFILE = Path("/private/config.toml")
BEFORE = (1, 1, 1, 10)
ROOT_FILE = (1, 2, 2, 20)
MASTER_FILE = (1, 3, 3, 30)


def root_request():
    return ConfigSaveRequest("/new-root", (None, 1), PROFILE, Path("/"))


def master_request(enabled=False):
    return ConfigSaveRequest(enabled, None, PROFILE, key="local_tools_enabled")


@pytest.mark.asyncio
@pytest.mark.parametrize("published", [False, True])
async def test_known_sibling_write_preserves_or_repairs_prior_partial_receipt(
    published,
):
    owner = MCPLocalConfigSaves()
    await owner.submit(
        root_request(),
        lambda: ConfigSaveResult(
            "cache_refresh", "/new-root", False, 7, ROOT_FILE, 7, BEFORE
        ),
    )
    await owner.submit(
        master_request(),
        lambda: ConfigSaveResult(
            "saved" if published else "cache_refresh",
            False,
            published,
            8 if published else 7,
            MASTER_FILE,
            7,
            ROOT_FILE,
        ),
    )
    root = owner.state_for("workspace_root")
    assert root.result.phase == ("saved" if published else "cache_refresh")
    assert root.result.file_revision == MASTER_FILE
    assert owner.known_root(PROFILE, "/stale", 7, MASTER_FILE) == (
        "/stale" if published else "/new-root"
    )
    assert owner.state_for("local_tools_enabled").result.stored is False


@pytest.mark.asyncio
async def test_unknown_external_edit_does_not_extend_a_sibling_receipt():
    owner = MCPLocalConfigSaves()
    first = await owner.submit(
        root_request(),
        lambda: ConfigSaveResult(
            "cache_refresh", "/new-root", False, 7, ROOT_FILE, 7, BEFORE
        ),
    )
    await owner.submit(
        master_request(),
        lambda: ConfigSaveResult(
            "cache_refresh", False, False, 7, MASTER_FILE, 7, (9, 9, 9, 9)
        ),
    )
    assert owner.state_for("workspace_root") == first
    assert owner.known_root(PROFILE, "/external", 7, MASTER_FILE) == "/external"


@pytest.mark.asyncio
async def test_partial_master_false_survives_noop_and_failed_retry():
    owner = MCPLocalConfigSaves()
    await owner.submit(
        master_request(),
        lambda: ConfigSaveResult("cache_refresh", False, False, 7, MASTER_FILE),
    )
    await owner.submit(
        master_request(),
        lambda: ConfigSaveResult("saved", False, False, 7, MASTER_FILE),
    )
    assert owner.state_for("local_tools_enabled").result.phase == "cache_refresh"
    await owner.submit(master_request(), lambda: ConfigSaveResult("failed"))
    assert (
        owner.known_value("local_tools_enabled", PROFILE, True, 7, MASTER_FILE) is False
    )
    assert (
        owner.known_value("local_tools_enabled", PROFILE, True, 8, MASTER_FILE) is True
    )


@pytest.mark.asyncio
async def test_mixed_writes_share_order_and_do_not_replace_newer_pending_sibling():
    owner = MCPLocalConfigSaves()
    entered, release = threading.Event(), threading.Event()
    calls = []

    def root_write():
        calls.append("root")
        entered.set()
        assert release.wait(5)
        return ConfigSaveResult(
            "cache_refresh", "/new-root", False, 7, ROOT_FILE, 7, BEFORE
        )

    root = owner.submit(root_request(), root_write)
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        master = owner.submit(
            master_request(),
            lambda: (
                calls.append("master")
                or ConfigSaveResult(
                    "cache_refresh", False, False, 7, MASTER_FILE, 7, ROOT_FILE
                )
            ),
        )
        newer_root = owner.submit(
            ConfigSaveRequest("/later", (None, 2), PROFILE, Path("/")),
            lambda: calls.append("later root") or ConfigSaveResult("failed"),
        )
        assert owner.state_for("workspace_root").request.value == "/later"
        assert owner.state_for("workspace_root").result is None
        assert calls == ["root"]
    finally:
        release.set()
        await owner.close_and_drain()
    assert calls == ["root", "master", "later root"]
    assert owner.state_for("workspace_root") == newer_root.result()
    assert owner.state_for("workspace_root").result.phase == "failed"
    assert owner.known_root(PROFILE, "/stale", 7, MASTER_FILE) == "/new-root"
    assert root.done() and master.done()
