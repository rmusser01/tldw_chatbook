"""The run-start catalog reuse key covers every input a composition reads.

TASK-33620.15.1. A Console run start reuses the previous run's composed MCP
catalog while nothing it was composed from has changed. The mounted tests in
``Tests/UI/test_console_mcp_catalog_reuse.py`` pin the store, connection,
profile and maxima inputs on the real service. These pin the three inputs
that harness never varies -- the built-in manifest's identity, plugin server
ownership and the built-in raw-name exclusions -- over a service double, so
dropping any one of them from the reuse check turns its test red.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.mcp_tool_provider import MCPToolProvider
from tldw_chatbook.Chat import console_run_start_tools as tools


class _Service:
    """The reads a run-start composition makes, counted; fixed store files."""

    def __init__(self, tmp_path) -> None:
        permissions = tmp_path / "mcp_permissions.json"
        local_store = tmp_path / "local_mcp_store.json"
        permissions.write_text("{}")
        local_store.write_text("{}")
        self.permission_store = SimpleNamespace(path=permissions)
        self.local_service = SimpleNamespace(
            store=SimpleNamespace(path=local_store),
            connection_states=lambda profiles=(): tuple(False for _ in profiles),
            get_inventory=lambda: None,
        )
        self.manifest = ("source", "a" * 64)
        self.catalog_reads = 0

    async def run_catalog_fingerprint(self, profiles=()) -> tuple:
        return (
            ("file", "p" * 64, tools._stamp(self.permission_store.path)),
            (
                ("file", "l" * 64, tools._stamp(self.local_service.store.path)),
                self.manifest,
                self.local_service.connection_states(profiles),
            ),
        )

    def get_kill_switch(self) -> bool:
        return False

    async def local_external_catalog(self) -> list:
        self.catalog_reads += 1
        return []

    def effective_tool_states(self, hub_tools, **_kwargs) -> dict:
        return {}


@pytest.fixture
def service(tmp_path, monkeypatch):
    """A fresh double; every store file reads as settled; nothing kept."""
    monkeypatch.setattr(tools, "time", SimpleNamespace(time_ns=lambda: 2**62))
    tools.forget()
    yield _Service(tmp_path)
    tools.forget()


async def _composed_reads(service: _Service, **provider_kwargs) -> int:
    """Run one run start; return how many times it read the server catalog."""
    before = service.catalog_reads
    provider = MCPToolProvider(
        service=service, main_loop=asyncio.get_running_loop(), **provider_kwargs
    )
    assert await tools.compose_run_mcp_provider(service, provider) is provider
    return service.catalog_reads - before


@pytest.mark.asyncio
async def test_a_changed_builtin_manifest_recomposes_the_catalog(service):
    assert await _composed_reads(service) == 1
    assert await _composed_reads(service) == 0, "an unchanged catalog is reused"
    service.manifest = ("source", "b" * 64)
    assert await _composed_reads(service) == 1, "reused across a manifest change"
    assert await _composed_reads(service) == 0


@pytest.mark.asyncio
async def test_changed_plugin_ownership_recomposes_the_catalog(service):
    assert await _composed_reads(service) == 1
    assert await _composed_reads(service) == 0, "an unchanged catalog is reused"
    owned = {"owned_profile_ids": frozenset({"plugin-server"})}
    assert await _composed_reads(service, **owned) == 1, "reused across ownership"
    assert await _composed_reads(service, **owned) == 0


@pytest.mark.asyncio
async def test_changed_builtin_exclusions_recompose_the_catalog(service):
    assert await _composed_reads(service) == 1
    assert await _composed_reads(service) == 0, "an unchanged catalog is reused"
    excluded = {"builtin_raw_name_exclusions": ("library_search",)}
    assert await _composed_reads(service, **excluded) == 1, "reused across exclusions"
    assert await _composed_reads(service, **excluded) == 0
