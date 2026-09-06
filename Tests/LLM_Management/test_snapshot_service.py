"""Service/store boundary and exact-generation lifecycle barriers."""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Event_Handlers.LLM_Management_Events.server_lifecycle import (
    ServerLaunchClaim,
)
from tldw_chatbook.LLM_Management.snapshot_admission import prepare_launch
from tldw_chatbook.LLM_Management.snapshot_models import (
    ReadinessObservation,
    SlotObservation,
    SlotReceipt,
    SnapshotError,
)
from tldw_chatbook.LLM_Management.snapshot_settings import SnapshotPreferences
from tldw_chatbook.LLM_Management.snapshot_store import SnapshotStore


class Client:
    def __init__(self, descriptor, store):
        self.descriptor = descriptor
        self.store = store
        self.dispatched = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.calls = []
        self.error = None
        self.after_ack = None
        self.closed = 0
        self.build = "427291b5b34c"

    async def readiness(self):
        return ReadinessObservation(
            slots=(
                SlotObservation(
                    slot_id=0,
                    busy=False,
                    tokens=None,
                    context_size=4096,
                    observed_at=1.0,
                ),
            ),
            build_info=self.build,
            model_path=str(self.descriptor._model_paths[0]),
            runtime_values=(),
        )

    async def save(self, slot_id, filename):
        self.calls.append(("save", slot_id, filename))
        self.dispatched.set()
        await self.release.wait()
        if self.error:
            raise self.error
        path = self.store.prepare_launch_directory(self.descriptor.launch_id) / filename
        path.write_bytes(b"saved cache")
        if self.after_ack:
            self.after_ack()
        return SlotReceipt(slot_id=slot_id, filename=filename, tokens=7, bytes=11)

    async def restore(self, slot_id, filename):
        self.calls.append(("restore", slot_id, filename))
        self.dispatched.set()
        await self.release.wait()
        if self.error:
            raise self.error
        return SlotReceipt(slot_id=slot_id, filename=filename, tokens=7, bytes=11)

    async def aclose(self):
        self.closed += 1


@pytest.fixture
def harness(tmp_path, monkeypatch):
    from tldw_chatbook.LLM_Management import snapshot_service as module

    runtime, model = tmp_path / "llama-server", tmp_path / "model.gguf"
    runtime.write_bytes(b"runtime")
    model.write_bytes(b"model")
    store = SnapshotStore(tmp_path / "snapshots")
    state = SimpleNamespace(current=None, clients={}, store=store)

    def descriptor(launch_id, *, parallel=1):
        claim = ServerLaunchClaim("llamacpp")
        state.current = claim
        return prepare_launch(
            (
                str(runtime),
                "--model",
                str(model),
                "--host",
                "127.0.0.1",
                "--port",
                "8080",
                "--ctx-size",
                "4096",
                "--parallel",
                str(parallel),
                "--flash-attn",
                "off",
                "--fit",
                "off",
                "--device",
                "none",
                "--n-gpu-layers",
                "0",
                "--no-mmproj",
            ),
            {},
            claim,
            launch_id,
        )

    def client_factory(value):
        client = Client(value, store)
        state.clients[value.launch_id] = client
        return client

    monkeypatch.setattr(module, "SnapshotClient", client_factory)
    monkeypatch.setattr(
        module,
        "load_snapshot_preferences",
        lambda: SnapshotPreferences(enabled=True, keep_count=1),
    )
    state.service = module.LlamaCppSnapshotService(
        store, lambda claim: state.current is claim and not claim.cancel_event.is_set()
    )
    state.descriptor = descriptor
    state.first = descriptor("launch-a")
    state.service.attach(state.first)
    return state


async def settled(service):
    async with asyncio.timeout(5):
        while (
            service.view().operation_id is not None
            and service.view().status != "outcome_unknown"
        ):
            await asyncio.sleep(0.001)


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["save", "restore"])
@pytest.mark.parametrize("failure", [ValueError, OSError])
async def test_preference_admission_failure_is_safe_and_reserves_nothing(
    harness, monkeypatch, action, failure
):
    from tldw_chatbook.LLM_Management import snapshot_service as module

    h = harness
    await h.service.refresh()
    before = tuple(h.store.working.rglob("*"))

    def invalid():
        raise failure("private-preference-canary")

    monkeypatch.setattr(module, "load_snapshot_preferences", invalid)
    with pytest.raises(SnapshotError, match="preferences_unavailable") as raised:
        if action == "save":
            h.service.start_save(0)
        else:
            h.service.start_restore("saved-id", 0)
    assert not raised.value.submission_possible
    assert raised.value.__cause__ is None
    assert h.service.view().operation_id is None
    assert not h.clients["launch-a"].calls
    assert tuple(h.store.working.rglob("*")) == before
    await h.service.shutdown()


@pytest.mark.asyncio
async def test_confirmed_stop_reconciliation_failure_still_closes_and_settles(
    harness, monkeypatch
):
    h = harness
    await h.service.refresh()

    def fail_reconcile(_terminated):
        raise RuntimeError("private-reconcile-canary")

    monkeypatch.setattr(h.store, "reconcile", fail_reconcile)
    await h.service.server_stopped(h.first.claim, confirmed=True)
    assert h.clients["launch-a"].closed == 1
    assert h.service.view().status == "stopped"
    assert h.service.view().operation_id is None
    assert h.service.view().message == "cleanup_failed"
    await h.service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("entry", ["initialize", "refresh"])
async def test_entry_reconciliation_cleans_only_settled_work_offthread(
    harness, monkeypatch, entry
):
    from Tests.LLM_Management.snapshot_fixtures import commit_test_snapshot
    from tldw_chatbook.LLM_Management import snapshot_store as store_module
    from tldw_chatbook.LLM_Management.snapshot_service import LlamaCppSnapshotService

    h = harness
    work = {}
    for state in ("terminal", "reserved", "acknowledged", "unknown"):
        item = h.store.reserve_save("residue-" + state, 0)
        item.path.write_bytes(state.encode())
        if state != "reserved":
            h.store.set_operation_state(item, state)
        work[state] = item
    record = commit_test_snapshot(h.store, payload=b"deleted snapshot", slot_id=0)
    binary = h.store.catalog / record.filename
    unlink = store_module._unlink_verified

    def fail_unlink(path, identity):
        if path == binary:
            raise OSError("private-unlink-canary")
        return unlink(path, identity)

    with monkeypatch.context() as fault:
        fault.setattr(store_module, "_unlink_verified", fail_unlink)
        assert h.store.delete(record.snapshot_id)
    tombstone = h.store.catalog / f"{record.snapshot_id}.deleting"
    assert tombstone.exists() and binary.exists()
    assert not (h.store.catalog / f"{record.snapshot_id}.json").exists()
    calls = []
    reconcile = SnapshotStore.reconcile

    def observed(store, terminated):
        calls.append((threading.get_ident(), terminated))
        return reconcile(store, terminated)

    monkeypatch.setattr(SnapshotStore, "reconcile", observed)
    owner = (
        h.service
        if entry == "refresh"
        else LlamaCppSnapshotService(None, lambda _claim: False)
    )
    try:
        if entry == "initialize":
            await owner.initialize(lambda: h.store.root)
        else:
            await owner.refresh()
        assert calls and all(
            thread != threading.get_ident() and not terminated
            for thread, terminated in calls
        )
        assert not work["terminal"].path.exists()
        for state in ("reserved", "acknowledged", "unknown"):
            assert work[state].path.read_bytes() == state.encode()
            assert (
                work[state].path.parent / f"{work[state].operation_id}.json"
            ).exists()
        assert not binary.exists() and not tombstone.exists()
        assert owner.view().catalog.residual_bytes > 0
        before = len(calls)
        await owner.browse_catalog()
        assert len(calls) == before
    finally:
        await owner.shutdown()
        if owner is not h.service:
            await h.service.shutdown()


@pytest.mark.asyncio
async def test_entry_cleanup_warning_does_not_disable_ready_management(
    harness, monkeypatch
):
    h = harness
    monkeypatch.setattr(h.store, "reconcile", lambda _terminated: ("cleanup_failed",))
    await h.service.refresh()
    assert h.service.view().status == "idle"
    assert h.service.view().disabled_reason is None
    assert h.service.view().message == "cleanup_incomplete"
    monkeypatch.setattr(h.store, "reconcile", lambda _terminated: ())
    await h.service.refresh()
    assert h.service.view().message is None
    await h.service.shutdown()


@pytest.mark.asyncio
async def test_entry_reconciliation_waits_for_acknowledged_publication_lock(
    harness, monkeypatch
):
    from tldw_chatbook.LLM_Management import snapshot_store as module

    h = harness
    await h.service.refresh()
    publishing, reconcile_entered, release = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )
    flush = module._flush_binary
    reconcile = h.store.reconcile

    def held_flush(stream):
        publishing.set()
        assert release.wait(30)
        flush(stream)

    def entered_reconcile(terminated):
        assert not terminated
        reconcile_entered.set()
        return reconcile(terminated)

    monkeypatch.setattr(module, "_flush_binary", held_flush)
    monkeypatch.setattr(h.store, "reconcile", entered_reconcile)
    refresh = None
    try:
        h.service.start_save(0)
        assert await asyncio.to_thread(publishing.wait, 5)
        refresh = asyncio.create_task(h.service.refresh())
        assert await asyncio.to_thread(reconcile_entered.wait, 5)
        refresh.cancel()
        await asyncio.sleep(0)
        assert not refresh.done()
    finally:
        release.set()
        if refresh:
            await refresh
        await settled(h.service)
        await h.service.shutdown()
    assert len(h.store.list_records().records) == 1
    assert not any(h.store.working.rglob("*.bin"))


@pytest.mark.asyncio
async def test_catalog_compatibility_projection_tracks_ready_and_invalidated_launch(
    harness,
):
    from Tests.LLM_Management.snapshot_fixtures import commit_test_snapshot
    from tldw_chatbook.LLM_Management.snapshot_service import LlamaCppSnapshotService

    h = harness
    try:
        await h.service.refresh()
        h.service.start_save(0)
        await settled(h.service)
        matching = h.store.list_records().records[0]
        different = commit_test_snapshot(h.store, payload=b"other model", slot_id=1)
        await h.service.browse_catalog()
        assert dict(h.service.view().snapshot_compatibility) == {
            matching.snapshot_id: "matching",
            different.snapshot_id: "different",
        }
        assert h.service.view().storage_location == str(h.store.root)
        assert str(h.store.root) not in repr(h.service.view())
        await h.service.browse_catalog(limit=1)
        assert h.service.view().snapshot_compatibility == (
            (different.snapshot_id, "different"),
        )
        h.current = None
        assert h.service.view().snapshot_compatibility == (
            (different.snapshot_id, "unknown"),
        )
        detached = LlamaCppSnapshotService(h.store, lambda claim: False)
        try:
            await detached.browse_catalog()
            assert set(dict(detached.view().snapshot_compatibility).values()) == {
                "unknown"
            }
        finally:
            await detached.shutdown()
    finally:
        await h.service.shutdown()


@pytest.mark.asyncio
async def test_disabled_preference_preserves_attached_capability_and_live_retention(
    harness, monkeypatch
):
    from tldw_chatbook.LLM_Management import snapshot_service as module

    h = harness
    try:
        await h.service.refresh()
        h.service.start_save(0)
        await settled(h.service)
        monkeypatch.setattr(
            module,
            "load_snapshot_preferences",
            lambda: SnapshotPreferences(enabled=False, keep_count=2),
        )
        for _ in range(2):
            h.service.start_save(0)
            await settled(h.service)
        records = h.store.list_records().records
        assert len(records) == 2
        h.service.start_restore(records[0].snapshot_id, 0)
        await settled(h.service)
        assert [call[0] for call in h.clients["launch-a"].calls] == [
            "save",
            "save",
            "save",
            "restore",
        ]
        assert h.store.list_records().records == records
    finally:
        await h.service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("destination_context", [6, 7, 8])
async def test_restore_capacity_admission_precedes_unknown_and_post(
    harness, monkeypatch, destination_context
):
    h = harness
    h.service.attach(h.descriptor("launch-b", parallel=2))
    client = h.clients["launch-b"]
    original_readiness = client.readiness

    async def readiness():
        observation = await original_readiness()
        return ReadinessObservation(
            build_info=observation.build_info,
            model_path=observation.model_path,
            runtime_values=observation.runtime_values,
            slots=(
                *observation.slots,
                SlotObservation(
                    slot_id=1,
                    busy=False,
                    tokens=None,
                    context_size=destination_context,
                    observed_at=1.0,
                ),
            ),
        )

    monkeypatch.setattr(client, "readiness", readiness)
    try:
        await h.service.refresh()
        h.service.start_save(0)
        await settled(h.service)
        record = h.store.list_records().records[0]
        assert record.tokens == 7
        transitions = []
        original_state = h.store.set_operation_state

        def set_state(working, state):
            transitions.append(state)
            return original_state(working, state)

        monkeypatch.setattr(h.store, "set_operation_state", set_state)
        h.service.start_restore(record.snapshot_id, 1)
        await settled(h.service)
        restores = [call for call in client.calls if call[0] == "restore"]
        if destination_context < 7:
            assert restores == []
            assert "unknown" not in transitions
            assert h.service.view().message == "destination_context_too_small"
        else:
            assert len(restores) == 1
            assert restores[0][1] == 1
            assert "unknown" in transitions
        assert h.store.list_records().records == (record,)
    finally:
        await h.service.shutdown()


@pytest.mark.asyncio
async def test_duplicate_save_is_rejected_before_return_and_navigation_cannot_cancel(
    harness,
):
    h = harness
    await h.service.refresh()
    client = h.clients["launch-a"]
    client.release.clear()
    paints = []
    unsubscribe = h.service.subscribe(lambda: paints.append(h.service.view().status))
    h.service.start_save(0)
    with pytest.raises(SnapshotError):
        h.service.start_save(0)
    await asyncio.wait_for(client.dispatched.wait(), 5)
    unsubscribe()  # Navigation releases its subscription, not the app's operation.
    paint_count = len(paints)
    client.release.set()
    await settled(h.service)
    assert len(client.calls) == 1
    assert len(h.store.list_records().records) == 1
    assert len(paints) == paint_count
    await h.service.shutdown()


@pytest.mark.asyncio
async def test_invalidated_acknowledgement_with_keep_one_preserves_old_snapshot(
    harness,
):
    h = harness
    await h.service.refresh()
    h.service.start_save(0)
    await settled(h.service)
    old = h.store.list_records().records[0]
    h.clients["launch-a"].after_ack = lambda: h.first.files[1].path.write_bytes(
        b"changed model"
    )
    h.service.start_save(0)
    await settled(h.service)
    assert h.store.list_records().records == (old,)
    assert (h.store.catalog / old.filename).read_bytes() == b"saved cache"
    await h.service.shutdown()


@pytest.mark.asyncio
async def test_unknown_outcome_blocks_mutation_but_allows_delete_until_confirmed_stop(
    harness,
):
    h = harness
    await h.service.refresh()
    h.service.start_save(0)
    await settled(h.service)
    old = h.store.list_records().records[0]
    client = h.clients["launch-a"]
    client.error = SnapshotError("outcome_unknown", submission_possible=True)
    h.service.start_save(0)
    await settled(h.service)
    with pytest.raises(SnapshotError):
        h.service.start_restore(old.snapshot_id, 0)
    await h.service.delete_snapshot(old.snapshot_id)
    assert h.store.list_records().records == ()
    assert len(client.calls) == 2
    await h.service.server_stopped(h.first.claim, confirmed=False)
    assert any(h.store.working.rglob("*.bin"))
    h.current = None
    await h.service.server_stopped(h.first.claim, confirmed=True)
    assert not any(h.store.working.rglob("*.bin"))
    await h.service.shutdown()


@pytest.mark.asyncio
async def test_old_response_and_cleanup_do_not_paint_or_reap_replacement(harness):
    h = harness
    await h.service.refresh()
    old = h.clients["launch-a"]
    old.release.clear()
    h.service.start_save(0)
    await asyncio.wait_for(old.dispatched.wait(), 5)
    replacement = h.descriptor("launch-b")
    h.service.attach(replacement)
    await h.service.refresh()
    new = h.clients["launch-b"]
    new.release.clear()
    h.service.start_save(0)
    await asyncio.wait_for(new.dispatched.wait(), 5)
    old.release.set()
    await h.service.server_stopped(h.first.claim, confirmed=True)
    assert h.service.view().launch_id == "launch-b"
    assert h.service.view().status == "awaiting_ack"
    assert any((h.store.working / "launch-b").glob("*.bin"))
    new.release.set()
    await settled(h.service)
    assert len(h.store.list_records().records) == 1
    assert len(old.calls) == len(new.calls) == 1
    await h.service.shutdown()


@pytest.mark.asyncio
async def test_stop_during_restore_staging_waits_for_local_handles_before_cleanup(
    harness, monkeypatch
):
    h = harness
    await h.service.refresh()
    h.service.start_save(0)
    await settled(h.service)
    record = h.store.list_records().records[0]
    entered, release = threading.Event(), threading.Event()
    stage = h.store.stage_restore

    def delayed(*args):
        working = stage(*args)
        entered.set()
        assert release.wait(5)
        return working

    monkeypatch.setattr(h.store, "stage_restore", delayed)
    h.service.start_restore(record.snapshot_id, 0)
    assert await asyncio.to_thread(entered.wait, 5)
    h.first.claim.cancel_event.set()
    stopping = asyncio.create_task(
        h.service.server_stopped(h.first.claim, confirmed=True)
    )
    await asyncio.sleep(0)
    assert not stopping.done()
    release.set()
    await asyncio.wait_for(stopping, 5)
    assert [call[0] for call in h.clients["launch-a"].calls] == ["save"]
    assert h.store.list_records().records == (record,)
    assert not any(h.store.working.rglob("*.bin"))
    await h.service.shutdown()


@pytest.mark.asyncio
async def test_shutdown_cancels_network_without_deleting_unknown_and_is_idempotent(
    harness,
):
    h = harness
    await h.service.refresh()
    client = h.clients["launch-a"]
    client.release.clear()
    h.service.start_save(0)
    await asyncio.wait_for(client.dispatched.wait(), 5)
    await asyncio.wait_for(
        asyncio.gather(h.service.shutdown(), h.service.shutdown()), 5
    )
    assert len(client.calls) == 1
    assert client.closed == 1
    assert any(h.store.working.rglob("*.bin"))
    with pytest.raises(SnapshotError):
        h.service.start_save(0)


@pytest.mark.asyncio
async def test_later_observation_change_invalidates_instead_of_relabelling(harness):
    h = harness
    await h.service.refresh()
    assert h.service.view().disabled_reason is None
    h.clients["launch-a"].build = "different-build"
    await h.service.refresh()
    assert h.service.view().disabled_reason is not None
    with pytest.raises(SnapshotError):
        h.service.start_save(0)
    assert h.clients["launch-a"].calls == []
    await h.service.shutdown()


@pytest.mark.asyncio
async def test_failed_unknown_persistence_prevents_post(harness, monkeypatch):
    h = harness
    await h.service.refresh()
    original = h.store.set_operation_state

    def fail_unknown(working, state):
        if state == "unknown":
            raise SnapshotError("storage_failed", submission_possible=False)
        return original(working, state)

    monkeypatch.setattr(h.store, "set_operation_state", fail_unknown)
    h.service.start_save(0)
    await settled(h.service)
    assert h.clients["launch-a"].calls == []
    assert not any(h.store.working.rglob("*.bin"))
    await h.service.shutdown()


@pytest.mark.asyncio
async def test_invalidation_during_publication_worker_preserves_keep_one(
    harness, monkeypatch
):
    from tldw_chatbook.LLM_Management import snapshot_store

    h = harness
    await h.service.refresh()
    h.service.start_save(0)
    await settled(h.service)
    old = h.store.list_records().records[0]
    entered, release = threading.Event(), threading.Event()
    flush = snapshot_store._flush_binary

    def paused(stream):
        flush(stream)
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(snapshot_store, "_flush_binary", paused)
    h.service.start_save(0)
    assert await asyncio.to_thread(entered.wait, 5)
    h.first.claim.cancel_event.set()
    release.set()
    await settled(h.service)
    assert h.store.list_records().records == (old,)
    assert (h.store.catalog / old.filename).read_bytes() == b"saved cache"
    await h.service.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("action", ["save", "restore"])
@pytest.mark.parametrize("observation", ["pending_success", "failure", "invalidated"])
async def test_background_readiness_during_staging_only_blocks_on_lost_evidence(
    harness, monkeypatch, action, observation
):
    """A pending reentry probe must not revoke the operation's fresh evidence."""
    h = harness
    await h.service.refresh()
    h.service.start_save(0)
    await settled(h.service)
    saved = h.store.list_records().records[0]
    client = h.clients["launch-a"]
    client.calls.clear()
    client.dispatched.clear()
    staged, release_stage = threading.Event(), threading.Event()
    probing, release_probe = asyncio.Event(), asyncio.Event()
    method = "reserve_save" if action == "save" else "stage_restore"
    original_stage = getattr(h.store, method)
    original_readiness = client.readiness

    def stage(*args):
        working = original_stage(*args)
        staged.set()
        assert release_stage.wait(5)
        return working

    async def readiness():
        probing.set()
        await release_probe.wait()
        if observation == "failure":
            raise SnapshotError("connection_failed", submission_possible=False)
        return await original_readiness()

    monkeypatch.setattr(h.store, method, stage)
    refresh = None
    try:
        if action == "save":
            h.service.start_save(0)
        else:
            h.service.start_restore(saved.snapshot_id, 0)
        assert await asyncio.to_thread(staged.wait, 5)
        monkeypatch.setattr(client, "readiness", readiness)
        refresh = asyncio.create_task(h.service.refresh())
        async with asyncio.timeout(5):
            await probing.wait()
        if observation == "failure":
            release_probe.set()
            await refresh
        elif observation == "invalidated":
            h.first.claim.cancel_event.set()
        release_stage.set()
        if observation == "pending_success":
            async with asyncio.timeout(5):
                while (
                    not client.dispatched.is_set() and h.service.view().message is None
                ):
                    await asyncio.sleep(0.001)
        else:
            await settled(h.service)
        assert client.dispatched.is_set() is (observation == "pending_success")
        if action == "restore" or observation != "pending_success":
            assert (h.store.catalog / saved.filename).read_bytes() == b"saved cache"
    finally:
        release_stage.set()
        release_probe.set()
        if refresh is not None:
            await refresh
        await settled(h.service)
        await h.service.shutdown()


@pytest.mark.asyncio
async def test_failed_action_refresh_prevents_post_even_after_prior_ready(
    harness, monkeypatch
):
    h = harness
    await h.service.refresh()

    async def unavailable():
        raise SnapshotError("connection_failed", submission_possible=False)

    monkeypatch.setattr(h.clients["launch-a"], "readiness", unavailable)
    h.service.start_save(0)
    await settled(h.service)
    assert h.clients["launch-a"].calls == []
    assert h.store.list_records().records == ()
    await h.service.shutdown()


@pytest.mark.asyncio
async def test_mismatched_receipt_never_persists_acknowledgement(harness, monkeypatch):
    h = harness
    await h.service.refresh()

    async def mismatched(slot_id, filename):
        return SlotReceipt(slot_id=slot_id + 1, filename=filename, tokens=7, bytes=11)

    monkeypatch.setattr(h.clients["launch-a"], "save", mismatched)
    h.service.start_save(0)
    await settled(h.service)
    assert h.service.view().status == "outcome_unknown"
    manifests = list((h.store.working / "launch-a").glob("*.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["state"] == "unknown"
    await h.service.shutdown()


@pytest.mark.asyncio
async def test_initialize_is_offthread_once_and_shutdown_settles_cancelled_awaiter(
    tmp_path,
):
    from tldw_chatbook.LLM_Management.snapshot_service import LlamaCppSnapshotService

    service = LlamaCppSnapshotService(None, lambda claim: False)
    entered, release = threading.Event(), threading.Event()
    threads = []

    def root():
        threads.append(threading.get_ident())
        entered.set()
        assert release.wait(5)
        return tmp_path / "snapshots"

    initializing = asyncio.create_task(service.initialize(root))
    assert await asyncio.to_thread(entered.wait, 5)
    assert service.view().status == "preparing"
    initializing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await initializing
    shutdown = asyncio.create_task(service.shutdown())
    await asyncio.sleep(0)
    assert not shutdown.done()
    release.set()
    await asyncio.wait_for(shutdown, 5)
    assert threads != [threading.get_ident()]
    assert len(threads) == 1
    assert service.store is not None


@pytest.mark.asyncio
async def test_snapshot_composition_is_lazy_and_shutdown_owner_is_called(monkeypatch):
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.LLM_Management import snapshot_service

    constructions = []

    class Owner:
        def __init__(self, store, is_current):
            constructions.append(store)
            self.closed = 0
            self.initialized = 0

        async def initialize(self, root_factory):
            self.initialized += 1

        async def shutdown(self):
            self.closed += 1

    monkeypatch.setattr(snapshot_service, "LlamaCppSnapshotService", Owner)
    app = SimpleNamespace(is_running=True)
    TldwCli._wire_llamacpp_snapshot_service(app)
    assert constructions == []
    owner = TldwCli.llamacpp_snapshot_service.fget(app)
    assert TldwCli.llamacpp_snapshot_service.fget(app) is owner
    await app._llamacpp_snapshot_setup_task
    assert constructions == [None]
    assert owner.initialized == 1

    _stub_other_app_lifecycles(app)
    await TldwCli._shutdown_app_owned_lifecycles(app)
    assert owner.closed == 1


def _stub_other_app_lifecycles(app):

    class Noop:
        async def shutdown(self):
            pass

    async def no_op():
        pass

    app.audio_cpp_model_install_owner = Noop()
    app._meeting_session_owner = None
    app._shutdown_console_image_edits = no_op
    app._shutdown_console_runtime = no_op
    app._shutdown_file_notes_session_owner = no_op
    app._shutdown_collections_capture_runtime = no_op
    app._shutdown_notes_sync_runtime = no_op
    app._shutdown_actor_pack_import = no_op
    app._shutdown_actor_pack_export = no_op
    app._shutdown_raw_cli_runtime = no_op
    app._shutdown_terminal_session_manager = no_op
    app._shutdown_console_settings_durability = no_op
    app._shutdown_persona_buddy = no_op


@pytest.mark.asyncio
async def test_unused_snapshot_owner_is_not_created_during_shutdown(monkeypatch):
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.LLM_Management import snapshot_service

    class App(SimpleNamespace):
        llamacpp_snapshot_service = TldwCli.llamacpp_snapshot_service

    app = App(is_running=True)
    TldwCli._wire_llamacpp_snapshot_service(app)
    monkeypatch.setattr(
        snapshot_service,
        "LlamaCppSnapshotService",
        lambda *args: pytest.fail("unused snapshot owner was created at shutdown"),
    )
    _stub_other_app_lifecycles(app)
    await TldwCli._shutdown_app_owned_lifecycles(app)
    assert app._llamacpp_snapshot_service is None
    assert app._llamacpp_snapshot_setup_task is None


@pytest.mark.parametrize("is_running", [False, True])
def test_snapshot_getter_without_running_loop_and_public_injection(is_running):
    from tldw_chatbook.app import TldwCli
    from tldw_chatbook.LLM_Management.snapshot_service import LlamaCppSnapshotService

    app = SimpleNamespace(is_running=is_running)
    TldwCli._wire_llamacpp_snapshot_service(app)
    owner = TldwCli.llamacpp_snapshot_service.fget(app)
    assert isinstance(owner, LlamaCppSnapshotService)
    assert owner.store is None
    assert app._llamacpp_snapshot_setup_task is None
    injected = LlamaCppSnapshotService(None, lambda claim: False)
    TldwCli.llamacpp_snapshot_service.fset(app, injected)
    assert TldwCli.llamacpp_snapshot_service.fget(app) is injected


@pytest.mark.asyncio
async def test_snapshot_pre_run_explicit_root_survives_first_running_access(tmp_path):
    from tldw_chatbook.app import TldwCli

    app = SimpleNamespace(is_running=False)
    TldwCli._wire_llamacpp_snapshot_service(app)
    owner = TldwCli.llamacpp_snapshot_service.fget(app)
    assert app._llamacpp_snapshot_setup_task is None
    root = tmp_path / "explicit-profile"
    await owner.initialize(lambda: root)
    app.is_running = True
    assert TldwCli.llamacpp_snapshot_service.fget(app) is owner
    await app._llamacpp_snapshot_setup_task
    assert owner.store.root == root
    await owner.shutdown()


@pytest.mark.asyncio
async def test_app_shutdown_settles_lazy_snapshot_initialization(tmp_path, monkeypatch):
    from tldw_chatbook import app as app_module

    entered, release = threading.Event(), threading.Event()

    def root():
        entered.set()
        assert release.wait(5)
        return tmp_path

    monkeypatch.setattr(app_module, "get_user_data_dir", root)
    app = SimpleNamespace(is_running=True)
    app_module.TldwCli._wire_llamacpp_snapshot_service(app)
    owner = app_module.TldwCli.llamacpp_snapshot_service.fget(app)
    _stub_other_app_lifecycles(app)
    try:
        assert await asyncio.to_thread(entered.wait, 5)
        shutdown = asyncio.create_task(
            app_module.TldwCli._shutdown_app_owned_lifecycles(app)
        )
        await asyncio.sleep(0)
        assert not shutdown.done()
    finally:
        release.set()
    await asyncio.wait_for(shutdown, 5)
    assert app._llamacpp_snapshot_setup_task.done()
    assert owner.store.root == tmp_path / "llamacpp_snapshots"


@pytest.mark.asyncio
async def test_explicit_stop_does_not_hold_lifecycle_lock_while_catalog_worker_settles(
    harness, monkeypatch
):
    from tldw_chatbook.Event_Handlers.LLM_Management_Events import server_lifecycle
    from tldw_chatbook.LLM_Management import snapshot_store

    h = harness
    await h.service.refresh()
    h.service.start_save(0)
    await settled(h.service)
    old = h.store.list_records().records[0]

    class Process:
        running = True

        def poll(self):
            return None if self.running else 0

        def terminate(self):
            self.running = False

        def wait(self, timeout):
            return 0

    app = SimpleNamespace(
        _llm_server_lifecycle_lock=threading.RLock(),
        _llm_server_launch_claims={"llamacpp": h.first.claim},
        llamacpp_server_process=Process(),
        llamacpp_snapshot_service=h.service,
        screen_stack=[],
        notify=lambda *a, **k: None,
    )
    h.first.claim._snapshot_context = server_lifecycle.SnapshotLaunchContext(
        h.first, h.store.prepare_launch_directory("launch-a")
    )
    h.service._is_current = lambda claim: server_lifecycle.snapshot_claim_is_live(
        app, claim
    )
    entered, release = threading.Event(), threading.Event()
    flush = snapshot_store._flush_binary

    def paused(stream):
        flush(stream)
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(snapshot_store, "_flush_binary", paused)
    h.service.start_save(0)
    assert await asyncio.to_thread(entered.wait, 5)
    assert await asyncio.wait_for(
        server_lifecycle.stop_server_process(app, "llamacpp", "llama.cpp"), 1
    )
    assert h.first.claim.cancel_event.is_set()
    release.set()
    await h.service.server_stopped(h.first.claim, confirmed=True)
    assert h.store.list_records().records == (old,)
    await h.service.shutdown()


@pytest.mark.asyncio
async def test_catalog_paging_is_local_and_newest_request_wins(harness, monkeypatch):
    h = harness
    await h.service.refresh()
    h.service.start_save(0)
    await settled(h.service)
    record = h.store.list_records().records[0]
    # Actual pagination uses real committed records; do not fabricate DTOs.
    from Tests.LLM_Management.snapshot_fixtures import commit_test_snapshot

    second = commit_test_snapshot(h.store, payload=b"another cache", slot_id=1)
    client = h.clients["launch-a"]

    async def no_http():
        pytest.fail("Catalog paging must not call readiness")

    monkeypatch.setattr(client, "readiness", no_http)
    await h.service.browse_catalog(offset=1, limit=1)
    assert h.service.view().catalog.records == (record,)
    await h.service.browse_catalog(offset=0, limit=1)
    assert h.service.view().catalog.records == (second,)
    assert h.service.view().catalog.next_offset == 1
    entered, release = threading.Event(), threading.Event()
    original = h.store.list_records

    def paused(offset=0, limit=50):
        page = original(offset, limit)
        if offset == 1:
            entered.set()
            assert release.wait(5)
        return page

    monkeypatch.setattr(h.store, "list_records", paused)
    old_browse = asyncio.create_task(h.service.browse_catalog(offset=1, limit=1))
    assert await asyncio.to_thread(entered.wait, 5)
    await h.service.browse_catalog(offset=0, limit=1)
    release.set()
    await old_browse
    assert h.service.view().catalog.records == (second,)
    assert len(client.calls) == 1
    await h.service.shutdown()


@pytest.mark.asyncio
async def test_stale_readiness_cannot_paint_replacement_or_admit_dead_child(
    harness, monkeypatch
):
    h = harness
    entered, release = asyncio.Event(), asyncio.Event()
    old_client = h.clients["launch-a"]
    original = old_client.readiness

    async def delayed():
        entered.set()
        await release.wait()
        return await original()

    monkeypatch.setattr(old_client, "readiness", delayed)
    old_refresh = asyncio.create_task(h.service.refresh())
    await asyncio.wait_for(entered.wait(), 5)
    replacement = h.descriptor("launch-b")
    h.service.attach(replacement)
    await h.service.refresh()
    paints = []
    unsubscribe = h.service.subscribe(lambda: paints.append(h.service.view()))
    release.set()
    await old_refresh
    assert paints == []
    assert h.service.view().launch_id == "launch-b"
    h.current = None
    await h.service.refresh()
    with pytest.raises(SnapshotError):
        h.service.start_save(0)
    assert old_client.calls == h.clients["launch-b"].calls == []
    unsubscribe()
    await h.service.shutdown()
