"""Run starts with nothing to reuse take no catalog fingerprint, on stock sources (TASK-33620.15.38).

Measured live on origin/dev e8eeae7db9 (Anthropic haiku via the review proxy,
160x45, a fresh profile, a timing-only hook with zero-wrap markers): every
Console send called ``run_catalog_fingerprint`` and then composed from the
stores; none reused or kept a catalog. A live send freezes its MCP definition
ceiling at composition (``mcp_definition_capture == "composition"``), and
such a run never reuses a kept catalog, so its fingerprint decided nothing.

These run the stock route a live send takes (installed stores, the default
manifest provider, the shared preparation) with nothing replaced on it; the
fingerprint is counted on the service instance, which no stock check reads.
Both are red on origin/dev, which takes a fingerprint before every
composition. ``Tests/UI/test_console_mcp_catalog_anchor.py`` pins how the
next run start proves or refuses a composition kept without one.
"""

from __future__ import annotations

import asyncio
import sys
import threading
from contextlib import contextmanager

import pytest

from Tests.Chat import test_console_async_mcp_snapshot as snapshot_controls
from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
from tldw_chatbook.Chat import console_run_start_tools as tools
from tldw_chatbook.MCP import console_tool_preparation
from tldw_chatbook.MCP.local_control_service import _default_manifest_provider

snapshot_case = snapshot_controls.snapshot_case
mcp_sources = snapshot_controls.mcp_sources
local_root = snapshot_controls.local_root
catalog_store = snapshot_controls.catalog_store


@pytest.fixture
def stock_case(snapshot_case, monkeypatch):
    """The snapshot case on the default manifest provider; nothing kept."""
    snapshot_case.local.manifest_provider = _default_manifest_provider
    # Store files written moments ago by the fixture count as settled.
    monkeypatch.setattr(tools, "SETTLE_NS", 0)
    tools.forget()
    yield snapshot_case
    tools.forget()


class Fingerprints:
    """Count run starts' catalog fingerprints on the service instance."""

    def __init__(self, service, monkeypatch) -> None:
        self.calls = 0
        real = service.run_catalog_fingerprint

        async def counted(*args, **kwargs):
            self.calls += 1
            return await real(*args, **kwargs)

        monkeypatch.setattr(service, "run_catalog_fingerprint", counted)


@contextmanager
def compositions():
    """Count the shared preparation's runs (a cold composition) on every thread."""
    code = console_tool_preparation.prepare_console_tools.__code__
    seen = []

    def observe(frame, event, _argument):
        if event == "call" and frame.f_code is code:
            seen.append(True)

    previous_main, previous_threads = sys.getprofile(), threading.getprofile()
    threading.setprofile_all_threads(observe)
    try:
        yield seen
    finally:
        threading.setprofile_all_threads(previous_threads)
        sys.setprofile(previous_main)


def _names(provider) -> list[str]:
    return [entry.name for entry in provider.list_catalog()]


@pytest.mark.asyncio
async def test_a_run_that_freezes_its_ceiling_at_composition_takes_no_fingerprint(
    stock_case, monkeypatch
):
    """A live send's run never reuses, so it takes no fingerprint."""
    case = stock_case
    fingerprints = Fingerprints(case.service, monkeypatch)
    loop = asyncio.get_running_loop()
    for _ in range(2):
        provider = MCPToolProvider(
            service=case.service,
            main_loop=loop,
            _freeze_definition_maximum_on_compose=True,
        )
        with compositions() as composed:
            assert await tools.compose_run_mcp_provider(case.service, provider) is provider
        assert composed, "a ceiling run did not compose through the shared preparation"
        assert _names(provider), "the stock sources offer tools"
        assert provider._maximum_tool_ids is not None, "the ceiling was not frozen"
    assert fingerprints.calls == 0, "a run that never reuses took a catalog fingerprint"


@pytest.mark.asyncio
async def test_a_first_stock_run_start_takes_no_fingerprint_and_the_next_reuses(
    stock_case, monkeypatch
):
    """The second run start's fingerprint proves the first composition."""
    case = stock_case
    fingerprints = Fingerprints(case.service, monkeypatch)
    loop = asyncio.get_running_loop()
    first = MCPToolProvider(service=case.service, main_loop=loop)
    with compositions() as composed:
        assert await tools.compose_run_mcp_provider(case.service, first) is first
    assert composed and _names(first)
    assert fingerprints.calls == 0, "a first run start took a catalog fingerprint"
    second = MCPToolProvider(service=case.service, main_loop=loop)
    with compositions() as composed:
        assert await tools.compose_run_mcp_provider(case.service, second) is second
    assert not composed, "the proven composition was not reused"
    assert fingerprints.calls == 1
    assert _names(second) == _names(first)
