"""The run-start catalog reuse key covers every input a composition reads.

TASK-33620.15.1. A Console run start reuses the previous run's composed MCP
catalog while nothing it was composed from has changed. The mounted tests in
``Tests/UI/test_console_mcp_catalog_reuse.py`` pin the store, connection,
profile and maxima inputs on the real service. These pin the three inputs
that harness never varies -- the built-in manifest's identity, plugin server
ownership and the built-in raw-name exclusions -- over a service double, so
dropping any one of them from the reuse check turns its test red. The
pre-merge review added three more: a runtime governance deny, a provider
other than the plain ``MCPToolProvider``, and a key input that changes while
the composition awaits its reads.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from Tests.MCP.test_catalog_fingerprint import GOVERNANCE_CHECKS, RefusingOne
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
def settled(monkeypatch):
    """Every store file reads as settled; nothing kept before or after."""
    monkeypatch.setattr(tools, "time", SimpleNamespace(time_ns=lambda: 2**62))
    tools.forget()
    yield
    tools.forget()


@pytest.fixture
def service(tmp_path, settled):
    """A fresh double."""
    return _Service(tmp_path)


class _Governed(_Service):
    """The double over the real local service's fingerprint and governance."""

    def __init__(self, tmp_path, policy) -> None:
        from tldw_chatbook.MCP.local_control_service import LocalMCPControlService
        from tldw_chatbook.MCP.local_store import LocalMCPStore

        super().__init__(tmp_path)
        self.local_service = LocalMCPControlService(
            store=LocalMCPStore(tmp_path / "governed_mcp_store.json"),
            manifest_provider=lambda: {"tools": []},
            policy_enforcer=policy,
        )

    async def run_catalog_fingerprint(self, profiles=()) -> tuple:
        permission, _local = await super().run_catalog_fingerprint(profiles)
        return permission, self.local_service.catalog_fingerprint(profiles)


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


@pytest.mark.asyncio
@pytest.mark.parametrize("refused", GOVERNANCE_CHECKS)
async def test_a_runtime_governance_deny_recomposes_the_catalog(
    tmp_path, settled, refused
):
    """A deny comes from the runtime policy, not the hashed stores."""
    policy = RefusingOne()
    service = _Governed(tmp_path, policy)
    assert await _composed_reads(service) == 1
    assert await _composed_reads(service) == 0, "an unchanged catalog is reused"
    policy.refused = refused
    assert await _composed_reads(service) == 1, "reused after a runtime deny"


class _KeywordComposition(MCPToolProvider):
    """A subclass composing on its own terms (it takes the run's switch)."""

    async def compose_catalog(self, **kwargs) -> None:
        await super().compose_catalog(**kwargs)


class _OwnComposition(MCPToolProvider):
    """A subclass whose ``compose_catalog`` reads its own switch, as plugins do."""

    async def compose_catalog(self) -> None:
        await super().compose_catalog()


@pytest.mark.asyncio
@pytest.mark.parametrize("subclass", [_KeywordComposition, _OwnComposition])
async def test_only_a_plain_mcp_provider_reuses_a_catalog(service, subclass):
    """A subclass's key does not cover what it composes from: it always composes."""
    loop = asyncio.get_running_loop()
    for _ in range(2):
        before = service.catalog_reads
        provider = subclass(service=service, main_loop=loop)
        local = tools.LocalKillSwitchRead()
        assert (
            await tools.compose_run_mcp_provider(service, provider, local) is provider
        )
        assert service.catalog_reads - before == 1, "a subclass's catalog was reused"
        assert local.read and not local.value
    assert await _composed_reads(service) == 1, "a plain run took a subclass's catalog"
    assert await _composed_reads(service) == 0


@pytest.mark.asyncio
async def test_a_profile_switch_during_composition_is_never_reused(service):
    """The kept key is the one the composition read after its await."""
    profile = {"id": "default"}
    real = service.local_external_catalog

    async def switching() -> list:
        profile["id"] = "named"  # The workspace profile changes meanwhile.
        return await real()

    def active() -> str:
        return profile["id"]

    service.local_external_catalog = switching
    assert await _composed_reads(service, profile_id_provider=active) == 1
    service.local_external_catalog = real
    profile["id"] = "default"
    assert await _composed_reads(service, profile_id_provider=active) == 1, (
        "a catalog composed for the named profile was reused for the default"
    )
    assert await _composed_reads(service, profile_id_provider=active) == 0
