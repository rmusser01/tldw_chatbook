"""Real store reads retain native custody; live catalog projection stays on its loop."""

import asyncio
import inspect
import sys
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from Tests.Backup_Recovery import test_mcp_source_lifetimes as source_fixtures
from Tests.Backup_Recovery import test_participant_lifetimes as participant_fixtures
from tldw_chatbook.Backup_Recovery import bootstrap, raw_participants, storage_admission
from tldw_chatbook.MCP.local_control_service import (
    LocalMCPControlService,
    MCPGovernanceDenied,
)
from tldw_chatbook.MCP.local_store import (
    LocalExternalMCPProfile,
    LocalMCPStore,
    LocalMCPStoreLoadError,
)
from tldw_chatbook.MCP.unified_control_plane_service import (
    UnifiedMCPControlPlaneService,
)


mcp_sources = source_fixtures.mcp_sources
local_root = participant_fixtures.local_root


class _ReadProbe:
    """Observe actual original code without replacing any source-checked method."""

    def __init__(self, store, *, hold=False, fail_after_release=False):
        self.store = store
        self.loop_thread = threading.current_thread()
        self.code = inspect.unwrap(LocalMCPStore._read_payload).__code__
        self.read_threads = []
        self.leases = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.hold = hold
        self.fail_after_release = fail_after_release

    def observe(self, frame, event, arg):
        if event != "call" or frame.f_code is not self.code:
            return
        if frame.f_locals.get("self") is not self.store:
            return
        thread = threading.current_thread()
        self.read_threads.append(thread)
        with storage_admission._changed:
            for state in tuple(raw_participants._states.values()):
                if state.source is self.store:
                    self.leases.extend(state.leases)
        self.entered.set()
        # A baseline caller-loop read must fail the assertion without deadlocking
        # the loop that is supposed to release this worker-only barrier.
        if self.hold and thread is not self.loop_thread:
            if not self.release.wait(8):
                raise AssertionError("native catalog worker was never released")
            if self.fail_after_release:
                raise OSError("observed catalog worker read failure")

    @contextmanager
    def installed(self):
        previous_main = sys.getprofile()
        previous_threads = threading.getprofile()
        threading.setprofile_all_threads(self.observe)
        try:
            yield self
        finally:
            self.release.set()
            threading.setprofile_all_threads(previous_threads)
            sys.setprofile(previous_main)


class _LoopGovernance:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.thread = threading.current_thread()
        self.calls = []
        self.denied = False

    def require_allowed(self, **kwargs):
        assert threading.current_thread() is self.thread
        assert asyncio.get_running_loop() is self.loop
        self.calls.append(kwargs["action_id"])
        if self.denied:
            raise MCPGovernanceDenied("catalog governance changed")


class _LoopClient:
    def __init__(self, sessions):
        self.loop = asyncio.get_running_loop()
        self.thread = threading.current_thread()
        self._sessions = sessions
        self.observations = 0

    @property
    def sessions(self):
        assert threading.current_thread() is self.thread
        assert asyncio.get_running_loop() is self.loop
        self.observations += 1
        return self._sessions


class _LoopPluginConnections:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.thread = threading.current_thread()
        self.calls = []

    def is_connected(self, profile_id):
        assert threading.current_thread() is self.thread
        assert asyncio.get_running_loop() is self.loop
        self.calls.append(profile_id)
        return True


@pytest.fixture
def catalog_store(mcp_sources):
    store = mcp_sources[0]
    store.save_profile(LocalExternalMCPProfile(profile_id="one", command="echo"))
    store.save_discovery_snapshot("one", {"tools": [{"name": "first"}]})
    store.save_profile_runtime_state("one", {"last_action": "refresh", "ok": True})
    return store


def _service(store, *, local_type=LocalMCPControlService):
    governance = _LoopGovernance()
    client = _LoopClient({"one": SimpleNamespace(_closed=False)})
    local = local_type(
        store=store,
        client=client,
        policy_enforcer=governance,
        manifest_provider=lambda: {},
    )
    service = UnifiedMCPControlPlaneService(
        target_store=None,
        context_store=None,
        local_service=local,
        server_service=None,
    )
    return service, local, governance, client


async def _worker_entered(probe, task):
    deadline = time.monotonic() + 4
    while not probe.entered.is_set() and not task.done():
        assert time.monotonic() < deadline, "catalog did not reach its real store read"
        await asyncio.sleep(0.01)
    assert probe.entered.is_set()
    assert (
        probe.read_threads[0] is not threading.current_thread()
    ), "blocking actual local-store read still executes on the caller loop"
    assert not task.done(), "native worker barrier did not retain the accepted call"


async def _settle(task, probe):
    probe.release.set()
    try:
        await task
    except (asyncio.CancelledError, Exception):
        pass


@pytest.mark.asyncio
async def test_standard_catalog_reads_one_fresh_bundle_off_loop(catalog_store):
    service, local, governance, client = _service(catalog_store)
    plugin = _LoopPluginConnections()
    local.connection_ownership = plugin
    catalog_store.save_profile(
        LocalExternalMCPProfile(profile_id="closed", command="echo")
    )
    client._sessions["closed"] = SimpleNamespace(_closed=True)
    catalog_store.save_profile(
        LocalExternalMCPProfile(
            profile_id="plugin",
            command="echo",
            plugin_owner={
                "installation_id": "installed",
                "revision_digest": "revision",
                "component_id": "component",
                "definition_digest": "definition",
                "environment": {},
                "data_root": None,
                "session_isolation": "separate",
                "literal_headers": {},
            },
        )
    )
    probe = _ReadProbe(catalog_store)
    with probe.installed():
        records = await service.local_external_catalog()
    assert len(probe.read_threads) == 1, "catalog repeats the full real store read"
    assert all(
        thread is not threading.current_thread() for thread in probe.read_threads
    )
    assert probe.leases and all(
        lease not in storage_admission._live_leases for lease in probe.leases
    )
    by_id = {record["profile_id"]: record for record in records}
    assert by_id["one"]["discovery_snapshot"] == {"tools": [{"name": "first"}]}
    assert by_id["one"]["runtime_state"] == {"last_action": "refresh", "ok": True}
    assert by_id["one"]["is_connected"] is True
    assert by_id["closed"]["discovery_snapshot"] is None
    assert by_id["closed"]["runtime_state"] is None
    assert by_id["closed"]["is_connected"] is False
    assert by_id["plugin"]["is_connected"] is True
    assert plugin.calls == ["plugin"] and client.observations == 1
    assert governance.calls == ["mcp.external_profiles.list.local"] * 2
    # No observation is retained for a later catalog request.
    catalog_store.save_profile_runtime_state("one", {"ok": False})
    with _ReadProbe(catalog_store).installed() as second:
        changed = await service.local_external_catalog()
    assert len(second.read_threads) == 1
    assert changed[0]["runtime_state"] == {"ok": False}


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel", [False, True])
async def test_accepted_catalog_retains_producer_and_native_custody(
    catalog_store, cancel
):
    service, _, _, client = _service(catalog_store)
    probe = _ReadProbe(catalog_store, hold=True)
    task = None
    pause = None
    with probe.installed():
        task = asyncio.create_task(service.local_external_catalog())
        try:
            await _worker_entered(probe, task)
            assert probe.leases and all(
                lease in storage_admission._live_leases and lease._key is not None
                for lease in probe.leases
            )
            # The real loop remains schedulable while its native worker is held.
            heartbeat = []
            asyncio.get_running_loop().call_soon(heartbeat.append, True)
            await asyncio.sleep(0)
            assert heartbeat == [True] and client.observations == 0
            service._maintenance_close_admission()
            pause = storage_admission._begin_local_pause()
            if cancel:
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
            assert not task.done()
            assert not await service._maintenance_drain(time.monotonic() + 0.03)
            assert not pause.drain(time.monotonic() + 0.03)
            with pytest.raises(
                bootstrap.RecoveryRequired, match="runtime_producer_paused"
            ):
                await service.local_external_catalog()
            probe.release.set()
            if cancel:
                with pytest.raises(asyncio.CancelledError):
                    await task
                assert client.observations == 0
            else:
                assert (await task)[0]["profile_id"] == "one"
            assert await service._maintenance_drain(time.monotonic() + 0.5)
            assert pause.drain(time.monotonic() + 0.5)
            assert all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            await _settle(task, probe)
            if pause is not None:
                pause.resume()
            if service._producer_lifetime.closed:
                service._maintenance_resume()


@pytest.mark.asyncio
@pytest.mark.parametrize("replace", ["service", "store"])
async def test_changed_catalog_owner_cannot_publish(catalog_store, tmp_path, replace):
    service, local, _, client = _service(catalog_store)
    replacement_store = LocalMCPStore(tmp_path / "replacement.json")
    probe = _ReadProbe(catalog_store, hold=True)
    with probe.installed():
        task = asyncio.create_task(service.local_external_catalog())
        try:
            await _worker_entered(probe, task)
            if replace == "service":
                service.local_service = LocalMCPControlService(store=replacement_store)
            else:
                local.store = replacement_store
            probe.release.set()
            with pytest.raises(
                bootstrap.RecoveryRequired, match="mcp_source_selection_changed"
            ):
                await task
            assert client.observations == 0
            assert probe.leases and all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
        finally:
            await _settle(task, probe)


@pytest.mark.asyncio
async def test_catalog_rechecks_loop_governance_after_worker(catalog_store):
    service, _, governance, client = _service(catalog_store)
    probe = _ReadProbe(catalog_store, hold=True)
    with probe.installed():
        task = asyncio.create_task(service.local_external_catalog())
        try:
            await _worker_entered(probe, task)
            governance.denied = True
            probe.release.set()
            with pytest.raises(MCPGovernanceDenied, match="governance changed"):
                await task
            assert client.observations == 0
        finally:
            await _settle(task, probe)


@pytest.mark.asyncio
async def test_cancelled_catalog_drains_worker_error_without_replacing_cancellation(
    catalog_store,
):
    service, _, _, client = _service(catalog_store)
    probe = _ReadProbe(catalog_store, hold=True, fail_after_release=True)
    with probe.installed():
        task = asyncio.create_task(service.local_external_catalog())
        try:
            await _worker_entered(probe, task)
            task.cancel()
            await asyncio.sleep(0)
            assert not task.done()
            probe.release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not service._producer_lifetime.calls
            assert probe.leases and all(
                lease not in storage_admission._live_leases for lease in probe.leases
            )
            assert client.observations == 0
        finally:
            await _settle(task, probe)


@pytest.mark.asyncio
async def test_store_failure_propagates_and_retires_native_catalog_work(catalog_store):
    service, _, _, client = _service(catalog_store)
    catalog_store.path.write_text("{broken", encoding="utf-8")
    with _ReadProbe(catalog_store).installed() as probe:
        with pytest.raises(LocalMCPStoreLoadError):
            await service.local_external_catalog()
    assert len(probe.read_threads) == 1
    assert probe.leases and all(
        lease not in storage_admission._live_leases for lease in probe.leases
    )
    assert not service._producer_lifetime.calls and client.observations == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["custom", "service_subclass", "store_subclass"])
async def test_custom_catalog_routes_keep_caller_loop_contract(
    catalog_store, tmp_path, route
):
    if route == "custom":

        class Custom:
            store = None

            def get_external_servers(self):
                assert asyncio.get_running_loop() is loop
                return [{"profile_id": "custom", "is_connected": True}]

        loop = asyncio.get_running_loop()
        local = Custom()
        service = UnifiedMCPControlPlaneService(
            target_store=None,
            context_store=None,
            local_service=local,
            server_service=None,
        )
        assert await service.local_external_catalog() == [
            {"profile_id": "custom", "is_connected": True, "runtime_state": None}
        ]
        return
    if route == "store_subclass":

        class CustomStore(LocalMCPStore):
            pass

        store = CustomStore(tmp_path / "custom-store.json")
        store.save_profile(LocalExternalMCPProfile(profile_id="one", command="echo"))
        service, _, _, _ = _service(store)
    else:

        class CustomLocal(LocalMCPControlService):
            pass

        store = catalog_store
        service, _, _, _ = _service(store, local_type=CustomLocal)
    with _ReadProbe(store).installed() as probe:
        records = await service.local_external_catalog()
    assert records[0]["profile_id"] == "one"
    assert probe.read_threads == [threading.current_thread()] * 2
