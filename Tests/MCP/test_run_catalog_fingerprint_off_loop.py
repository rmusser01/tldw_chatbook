"""A run's catalog fingerprint reads its stock sources off the UI loop (TASK-33620.15.38).

Measured live on origin/dev 29592ecc9b (Anthropic haiku via the review proxy,
160x45, fresh profiles; task notes): each Console run start checks its kept
MCP catalog with ``run_catalog_fingerprint``. Its two store reads are
storage-admission scopes whose binding checks resolve each store's
recovery-activation generation (``recovery_activation.selected_path``, 22
calls per send) with stat and read syscalls, and the built-in manifest's
source is read and hashed. All of it ran on the UI loop: 11-15 ms of CPU per
send at load1 ~8, and at load1 ~120 a median 81 ms of wall (20 ms of CPU) on
first sends, because the loop waited to be scheduled again after each syscall.

Stock, installed sources are now read on one owned worker, the reads
unchanged; the governance checks (before and after the hop), the receiver
re-check and the connection states stay on the loop. These pin each moved
read, the result, and the refusals, on real installed stores (the
``mcp_sources`` fixture binds them, so the recovery-activation check runs).
They observe actual calls with a profile hook and replace no callable on the
stock route: the route is chosen by the readers' identities.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from Tests.MCP.test_external_catalog_worker_ownership import (
    _LoopClient,
    _LoopGovernance,
    catalog_store as catalog_store,  # noqa: PLC0414 - fixture
    local_root as local_root,  # noqa: PLC0414 - fixture
    mcp_sources as mcp_sources,  # noqa: PLC0414 - fixture
)
from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired
from tldw_chatbook.MCP import recovery_activation
from tldw_chatbook.MCP.local_control_service import (
    LocalMCPControlService,
    MCPGovernanceDenied,
)
from tldw_chatbook.MCP.local_store import LocalMCPStore
from tldw_chatbook.MCP.permission_store import MCPPermissionStore
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)

#: Every read one fingerprint makes, by the label ``_Reads`` gives it.
FINGERPRINT_READS = frozenset(
    {
        "permission_store",
        "local_store",
        "manifest_source",
        "recovery_activation:mcp_permissions.json",
        "recovery_activation:local_mcp_store.json",
        "recovery_readable:mcp_permissions.json",
        "recovery_readable:local_mcp_store.json",
    }
)
GOVERNANCE = ["mcp.external_profiles.list.local", "mcp.inventory.list.local"]


def _service(store, permissions, sessions=None):
    """A real service over installed stores, with the stock manifest source.

    Governance and the MCP client assert they are used on the loop's thread.
    """
    governance = _LoopGovernance()
    client = _LoopClient(sessions or {})
    local = LocalMCPControlService(
        store=store, client=client, policy_enforcer=governance
    )
    service = UnifiedMCPControlPlaneService(
        target_store=None,
        context_store=None,
        local_service=local,
        server_service=None,
    )
    service._permission_store = permissions
    return service, local, governance, client


@pytest.fixture
def permissions(mcp_sources):
    store = mcp_sources[3]
    store.set_kill_switch(True)  # Persist a real policy file to read.
    store.set_kill_switch(False)
    return store


def _under_fingerprint(frame) -> bool:
    frame = frame.f_back
    while frame is not None:
        if "fingerprint" in frame.f_code.co_name:
            return True
        frame = frame.f_back
    return False


class _Reads:
    """Record ``(label, thread)`` per fingerprint read, on every thread.

    ``hold``: park the local store's read while it runs off the caller's
    thread (a read on the caller's thread is recorded, never held, so a
    baseline that reads on the loop fails its assertion instead of hanging).
    """

    def __init__(self, *, hold=False, on_local_read=None):
        self.loop_thread = threading.current_thread()
        self.seen: list[tuple[str, threading.Thread]] = []
        self.stores = {
            MCPPermissionStore.catalog_fingerprint.__wrapped__.__code__: "permission_store",
            LocalMCPStore.catalog_fingerprint.__wrapped__.__code__: "local_store",
        }
        self.check = recovery_activation.selected_path.__code__
        self.readable = recovery_activation.readable.__code__
        self.read_bytes = Path.read_bytes.__code__
        self.hold = hold
        self.on_local_read = on_local_read
        self.entered = threading.Event()
        self.release = threading.Event()

    def observe(self, frame, event, _argument):
        if event != "call":
            return
        code = frame.f_code
        label = self.stores.get(code)
        if label is None:
            if code is self.check:
                label = "recovery_activation:" + frame.f_locals["canonical"].name
            elif code is self.readable:
                store = frame.f_locals["store"]
                path = getattr(store, "_recovery_original_path", store.path)
                label = "recovery_readable:" + Path(path).name
            elif code is self.read_bytes and frame.f_locals["self"].name == "server.py":
                label = "manifest_source"
            else:
                return
            if not _under_fingerprint(frame):
                return
        thread = threading.current_thread()
        self.seen.append((label, thread))
        if label != "local_store":
            return
        if self.on_local_read is not None:
            self.on_local_read()
        self.entered.set()
        if self.hold and thread is not self.loop_thread:
            if not self.release.wait(8):
                raise AssertionError("the fingerprint worker was never released")

    def labels(self) -> set[str]:
        return {label for label, _thread in self.seen}

    def on_the_loop(self) -> list[str]:
        return sorted(
            {label for label, thread in self.seen if thread is self.loop_thread}
        )

    @contextmanager
    def installed(self):
        previous_main, previous_threads = sys.getprofile(), threading.getprofile()
        threading.setprofile_all_threads(self.observe)
        try:
            yield self
        finally:
            self.release.set()
            threading.setprofile_all_threads(previous_threads)
            sys.setprofile(previous_main)


@pytest.mark.asyncio
async def test_the_check_reads_both_stores_and_the_manifest_off_the_loop(
    catalog_store, permissions
):
    """Red on origin/dev: every read below ran on the UI loop's thread."""
    service, _local, governance, client = _service(catalog_store, permissions)
    with _Reads().installed() as reads:
        await service.run_catalog_fingerprint((("one", False),))
    assert FINGERPRINT_READS <= reads.labels(), sorted(reads.labels())
    assert not reads.on_the_loop(), f"read on the UI loop: {reads.on_the_loop()}"
    # What stays on the loop: the governance checks (before the hop and again
    # after it) and the connection states.
    assert governance.calls == GOVERNANCE * 2
    assert client.observations == 1


@pytest.mark.asyncio
async def test_the_off_loop_check_returns_what_the_loop_route_reads(
    catalog_store, permissions
):
    """Guard (green on origin/dev too): the same fingerprint, read for read."""
    sessions = {"one": SimpleNamespace(_closed=False)}
    service, local, _governance, _client = _service(
        catalog_store, permissions, sessions
    )
    profiles = (("one", False), ("two", False))
    moved = await service.run_catalog_fingerprint(profiles)
    loop = (permissions.catalog_fingerprint(), local.catalog_fingerprint(profiles))
    assert moved == loop
    assert moved[0][0] == "file" and moved[1][0][0] == "file"
    assert moved[1][1][0] == "source" and moved[1][2] == (True, False)
    catalog_store.save_discovery_snapshot("one", {"tools": [{"name": "second"}]})
    changed = await service.run_catalog_fingerprint(profiles)
    assert changed[1][0][:2] != moved[1][0][:2]
    assert changed == (
        permissions.catalog_fingerprint(),
        local.catalog_fingerprint(profiles),
    )


@pytest.mark.asyncio
async def test_a_store_replaced_while_the_check_reads_fails_the_check(
    catalog_store, permissions
):
    """Red on origin/dev: the check described a store that was no longer selected.

    Nothing on the loop could replace a source in the middle of a loop-side
    read; while a worker reads, the loop runs, so the check confirms its
    receivers after the hop and refuses as a changed store selection does.
    """
    service, local, _governance, _client = _service(catalog_store, permissions)
    stand_in = object()

    def replace():
        local.store = stand_in

    try:
        with _Reads(on_local_read=replace).installed() as reads:
            with pytest.raises(RecoveryRequired, match="mcp_source_selection_changed"):
                await service.run_catalog_fingerprint(())
    finally:
        local.store = catalog_store
    assert "local_store" in reads.labels()
    await service.run_catalog_fingerprint(())  # Restored: the check passes.


@pytest.mark.asyncio
async def test_a_store_replaced_after_the_worker_read_fails_the_check(
    catalog_store, permissions
):
    """Red on origin/dev: nothing re-checks the selected stores once they are read.

    The replacement lands on the loop after the worker's last read and before
    the check resumes there, the gap only the loop-side re-check covers. On
    origin/dev there is no off-loop read to finish, so it never lands and the
    check describes stores it did not confirm.
    """
    from tldw_chatbook.MCP import console_snapshot

    service, local, _governance, _client = _service(catalog_store, permissions)
    loop = asyncio.get_running_loop()
    loop_thread = threading.current_thread()
    captured = getattr(console_snapshot, "_CapturedFingerprint", None)
    code = None if captured is None else captured.read_fingerprint_sources.__code__
    replaced = []

    def replace():
        replaced.append(local.store)
        local.store = object()

    def observe(frame, event, _argument):
        if (
            event == "return"
            and frame.f_code is code
            and threading.current_thread() is not loop_thread
        ):
            loop.call_soon_threadsafe(replace)

    previous_main, previous_threads = sys.getprofile(), threading.getprofile()
    threading.setprofile_all_threads(observe)
    try:
        with pytest.raises(RecoveryRequired, match="mcp_source_selection_changed"):
            await service.run_catalog_fingerprint(())
    finally:
        threading.setprofile_all_threads(previous_threads)
        sys.setprofile(previous_main)
        local.store = catalog_store
    assert replaced == [catalog_store]


@pytest.mark.asyncio
async def test_a_governance_refusal_fails_the_check_before_the_local_store_read(
    catalog_store, permissions
):
    """Guard (green on origin/dev too): a refused listing never reads the local store."""
    service, _local, governance, _client = _service(catalog_store, permissions)
    governance.denied = True
    with _Reads().installed() as reads:
        with pytest.raises(MCPGovernanceDenied):
            await service.run_catalog_fingerprint(())
    assert "local_store" not in reads.labels()
    assert "manifest_source" not in reads.labels()


@pytest.mark.asyncio
async def test_a_governance_refusal_during_the_worker_read_fails_the_check(
    catalog_store, permissions
):
    """Red on origin/dev: the read ran on the loop, so there was no hop to refuse after.

    The reuse decision runs in the loop turn after the check returns. On the
    loop route nothing could change governance between its check and that
    decision; while a worker reads, the loop runs, so the check asks again.
    """
    service, _local, governance, client = _service(catalog_store, permissions)
    with _Reads(hold=True).installed() as reads:
        task = asyncio.create_task(service.run_catalog_fingerprint((("one", False),)))
        deadline = time.monotonic() + 4
        while not reads.entered.is_set() and not task.done():
            assert time.monotonic() < deadline, "the local store was never read"
            await asyncio.sleep(0.01)
        assert not reads.on_the_loop(), f"read on the UI loop: {reads.on_the_loop()}"
        governance.denied = True
        reads.release.set()
        with pytest.raises(MCPGovernanceDenied, match="governance changed"):
            await task
    assert governance.calls == GOVERNANCE + GOVERNANCE[:1]
    assert client.observations == 0


@pytest.mark.asyncio
async def test_an_uninstalled_store_keeps_the_loop_route(tmp_path):
    """Guard (green on origin/dev too): only installed stock stores leave the loop.

    A store with no installed binding has no source identity to re-check
    after a hop, so its read stays where it was, on the caller's loop.
    """
    store = LocalMCPStore(tmp_path / "local_mcp_store.json")
    permissions = MCPPermissionStore(tmp_path / "mcp_permissions.json")
    service, _local, governance, _client = _service(store, permissions)
    with _Reads().installed() as reads:
        result = await service.run_catalog_fingerprint(())
    assert {"permission_store", "local_store"} <= reads.labels()
    assert {"permission_store", "local_store"} <= set(reads.on_the_loop())
    assert governance.calls == GOVERNANCE
    assert result[1][1][0] == "source"


@pytest.mark.asyncio
async def test_a_custom_manifest_provider_keeps_the_loop_route(
    catalog_store, permissions
):
    """Guard (green on origin/dev too): a manifest callback is called on the loop."""
    service, local, _governance, _client = _service(catalog_store, permissions)
    called = []

    def manifest():
        called.append(threading.current_thread())
        return {"tools": []}

    local.manifest_provider = manifest
    with _Reads().installed() as reads:
        result = await service.run_catalog_fingerprint(())
    assert called == [threading.current_thread()]
    assert {"permission_store", "local_store"} <= set(reads.on_the_loop())
    assert result[1][1] == ("manifest", {"tools": []})


@pytest.mark.asyncio
async def test_custom_fingerprint_readers_keep_the_loop_route(
    catalog_store, permissions
):
    """Guard (green on origin/dev too): a replaced reader is called where it was."""
    service, local, _governance, _client = _service(catalog_store, permissions)
    called = []
    real = local.catalog_fingerprint

    def custom(profiles=()):
        called.append(threading.current_thread())
        return real(profiles)

    local.catalog_fingerprint = custom
    with _Reads().installed() as reads:
        await service.run_catalog_fingerprint(())
    assert called == [threading.current_thread()]
    assert reads.on_the_loop() == sorted(FINGERPRINT_READS)


@pytest.mark.asyncio
async def test_a_cancelled_check_retires_its_worker_read_first(
    catalog_store, permissions
):
    """Red on origin/dev: no worker read to hold (the read ran on the loop).

    The read is owned: a cancelled run start keeps the service's producer
    fence until the worker's read has finished, so maintenance cannot
    proceed under a read still in flight.
    """
    service, _local, _governance, _client = _service(catalog_store, permissions)
    with _Reads(hold=True).installed() as reads:
        task = asyncio.create_task(service.run_catalog_fingerprint(()))
        deadline = time.monotonic() + 4
        while not reads.entered.is_set() and not task.done():
            assert time.monotonic() < deadline, "the local store was never read"
            await asyncio.sleep(0.01)
        assert not reads.on_the_loop(), f"read on the UI loop: {reads.on_the_loop()}"
        service._maintenance_close_admission()
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done(), "cancellation left the worker's read behind"
        assert not await service._maintenance_drain(time.monotonic() + 0.03)
        reads.release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
    assert await service._maintenance_drain(time.monotonic() + 1)
