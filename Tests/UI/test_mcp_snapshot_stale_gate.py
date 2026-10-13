"""The MCP workbench reuses catalog snapshots when the fingerprint is unchanged."""

from types import SimpleNamespace

import pytest

from tldw_chatbook.UI.MCP_Modules.mcp_workbench import MCPWorkbench

pytestmark = pytest.mark.bootstrap_profile


class _Service:
    def __init__(self, fingerprint=None, error: bool = False) -> None:
        self._fingerprint = fingerprint if fingerprint is not None else ("fp", 1)
        self._error = error
        self.fingerprint_calls = 0

    async def run_catalog_fingerprint(self, profiles: tuple = ()) -> tuple:
        self.fingerprint_calls += 1
        if self._error:
            raise PermissionError("admission refused")
        return self._fingerprint


class _GateWorkbench(MCPWorkbench):
    """Bare shell: app_instance is a read-only property on the real widget."""

    app_instance = None

    def __init__(self) -> None:  # skip the heavy real widget init
        pass


def _workbench(service, *, source: str = "local"):
    workbench = _GateWorkbench()
    workbench._source = source
    workbench.app_instance = SimpleNamespace()
    collections = []
    marker = object()

    async def collect():
        collections.append(marker)
        return [f"snapshot-{len(collections)}"]

    workbench._collect_snapshots = collect
    return workbench, collections


@pytest.mark.asyncio
async def test_unchanged_fingerprint_reuses_cached_snapshots():
    service = _Service()
    workbench, collections = _workbench(service)

    first = await workbench._collect_snapshots_stale_gated(service)
    second = await workbench._collect_snapshots_stale_gated(service)
    assert len(collections) == 1, "unchanged fingerprint must not re-collect"
    assert second is first


@pytest.mark.asyncio
async def test_fingerprint_change_forces_a_full_walk():
    service = _Service(fingerprint=("fp", 1))
    workbench, collections = _workbench(service)

    await workbench._collect_snapshots_stale_gated(service)
    service._fingerprint = ("fp", 2)
    await workbench._collect_snapshots_stale_gated(service)
    assert len(collections) == 2


@pytest.mark.asyncio
async def test_force_and_server_source_always_collect():
    service = _Service()
    workbench, collections = _workbench(service)
    await workbench._collect_snapshots_stale_gated(service)

    await workbench._collect_snapshots_stale_gated(service, force=True)
    assert len(collections) == 2, "manual refresh bypasses the cache"

    workbench._source = "server"
    await workbench._collect_snapshots_stale_gated(service)
    assert len(collections) == 3, "server mode always collects"


@pytest.mark.asyncio
async def test_admission_refusal_falls_back_to_a_full_walk():
    service = _Service(error=True)
    workbench, collections = _workbench(service)

    snapshots = await workbench._collect_snapshots_stale_gated(service)
    assert len(collections) == 1 and snapshots == ["snapshot-1"]
    assert service.fingerprint_calls == 1
    assert not hasattr(workbench.app_instance, "_mcp_workbench_snapshot_cache")
