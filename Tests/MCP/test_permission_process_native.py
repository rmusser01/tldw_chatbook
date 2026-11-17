"""Real stock permission IPC, using original private-profile bootstrap selection."""

import asyncio
import gc
import json
import threading
import time
import weakref

import pytest


@pytest.fixture(autouse=True)
def process_audit(tmp_path, monkeypatch):
    from tldw_chatbook.MCP import permission_process as process

    # These controls explicitly create the owner. Production forwarding has its
    # own routed tests; fixture seeding here must retain the stock local path.
    monkeypatch.setattr(process, "forward", lambda *args, **kwargs: (False, None))

    created = []
    original = process.PermissionProcess.__init__

    def initialize(broker, source):
        original(broker, source)
        created.append(broker)

    monkeypatch.setattr(process.PermissionProcess, "__init__", initialize)
    yield created
    records = []
    for broker in created:
        assert broker.retired, "test left an owned native process"
        records.append(
            {
                "pid": None if broker.process is None else broker.process.pid,
                "retired": broker.retired,
                "unresolved": broker.unresolved is not None,
                "receipts": broker.receipts,
                "origins": broker.origins,
            }
        )
    (tmp_path / "permission-process-audit.json").write_text(
        json.dumps(records, indent=2)
    )
    # Test isolation only: retain on-disk evidence and never claim uncertainty cleared.
    process._brokers.difference_update(created)


@pytest.fixture
def source(tmp_path, monkeypatch):
    from Tests.Backup_Recovery.config_test_support import install_config_source
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore

    data = tmp_path / "data"
    data.mkdir(mode=0o700)
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f'[paths]\ndata_dir = "{data.as_posix()}"\n', encoding="utf-8"
    )
    monkeypatch.setenv("TLDW_CONFIG_PATH", str(config_path))
    config = install_config_source(monkeypatch)
    return MCPPermissionStore(config.get_user_data_dir() / "mcp_permissions.json")


def test_real_warm_read_write_caller_stamp_and_graceful_retirement(source):
    from tldw_chatbook.MCP.permission_process import PermissionProcess

    broker = PermissionProcess(source)
    try:
        assert broker.call("get_kill_switch") is False
        token = source._permission_publication.token
        assert token is not None
        assert broker.call("get_kill_switch") is False
        assert source._permission_publication.token is token
        payload = broker.call("load", publish=False)
        assert source._permission_publication.token is token
        assert "updated_at" not in payload
        broker.call("set_kill_switch", value=True)
        assert source._permission_publication.token is None
        assert not source._permission_publication.pending
        assert broker.call("get_kill_switch") is True
        payload = broker.call("load", publish=False)
        payload.pop("updated_at")
        broker.call("save", payload=payload)
        assert isinstance(payload["updated_at"], str)
        assert json.loads(source.path.read_text()) == payload
        assert all(receipt["retired"] for receipt in broker.receipts)
        assert all(receipt["startup"] >= 1 for receipt in broker.receipts)
        assert broker.origins["tldw_chatbook.MCP.permission_store"][0].endswith(
            "permission_store.py"
        )
    finally:
        if broker.unresolved is None:
            broker.close()
        else:
            broker.contain_unresolved()
    assert broker.retired and broker.process.poll() == 0


def test_captured_first_load_preserves_existing_parent_publication(source):
    from tldw_chatbook.MCP.permission_process import PermissionProcess

    source.load()
    token = source._permission_publication.token
    broker = PermissionProcess(source)
    try:
        assert broker.call("load", publish=False)["kill_switch"] is False
        assert source._permission_publication.token is token
    finally:
        broker.contain_unresolved() if broker.unresolved else broker.close()


def test_changed_noop_observation_revokes_hook_before_child_continuation(
    source, monkeypatch
):
    from tldw_chatbook.Agents.hooks_v2.mcp_executor import MCPHookPermission
    from tldw_chatbook.MCP import permission_process as process

    source.set_kill_switch(False)
    source.load()
    hook = object.__new__(MCPHookPermission)
    hook.store, hook.publication = source, source._permission_publication
    hook.token, hook.session_grant = hook.publication.token, None
    hook.check_memory()
    payload = (
        json.loads(source.path.read_text()) if source.path.exists() else source.load()
    )
    payload["kill_switch"] = True
    source.path.write_text(json.dumps(payload), encoding="utf-8")
    broker = process.PermissionProcess(source)
    observed = []
    read, write = process.read_message, process.write_message

    def reading(stream):
        message = read(stream)
        if message.get("type") == "Observe":
            observed.append(message["ordinal"])
        return message

    def writing(stream, message):
        if observed and message.get("type") == "Acknowledge":
            assert hook.publication.token is None
            with pytest.raises(PermissionError):
                hook.check_memory()
        return write(stream, message)

    monkeypatch.setattr(process, "read_message", reading)
    monkeypatch.setattr(process, "write_message", writing)
    try:
        broker.call("set_kill_switch", value=True)
        assert observed and hook.publication.token is not None
        with pytest.raises(PermissionError):
            hook.check_memory()
    finally:
        broker.contain_unresolved() if broker.unresolved else broker.close()


@pytest.mark.parametrize("substitution", ["source", "config", "config_value"])
def test_actual_parent_substitution_at_protected_open_refuses_no_replay(
    source, monkeypatch, substitution
):
    from tldw_chatbook.MCP import permission_process as process
    from tldw_chatbook.Backup_Recovery import mcp_source_participants as sources

    broker = process.PermissionProcess(source)
    read = process.read_message
    replaced = []

    def reading(stream):
        message = read(stream)
        if (
            message.get("type") == "Check"
            and message["data"]["site"] == "protected_open"
            and not replaced
        ):
            replaced.append(True)
            if substitution == "source":
                monkeypatch.setattr(source, "load", lambda: {})
            elif substitution == "config":
                config = sources._BINDINGS[source].config
                monkeypatch.setattr(
                    config,
                    "_CONFIG_CACHE_SOURCE",
                    config._CONFIG_CACHE_SOURCE.with_name("changed.toml"),
                )
            else:
                config = sources._BINDINGS[source].config
                monkeypatch.setitem(
                    config._CONFIG_CACHE["paths"], "data_dir", "changed"
                )
        return message

    monkeypatch.setattr(process, "read_message", reading)
    try:
        with pytest.raises(process.ProtocolError, match="source_changed"):
            broker.call("get_kill_switch")
        assert (
            replaced and broker.unresolved and source._permission_publication.uncertain
        )
        with pytest.raises(process.ProtocolError):
            broker.call("get_kill_switch")
        with pytest.raises(process.ProtocolError, match="unresolved"):
            broker.close()
    finally:
        broker.contain_unresolved() if broker.unresolved else broker.close()
    assert broker.retired


@pytest.mark.parametrize("fault", ["command", "shutdown"])
def test_failed_native_ownership_withholds_stamp_and_drained(
    source, monkeypatch, fault
):
    from tldw_chatbook.MCP import permission_process as process

    popen = process.subprocess.Popen

    def launch(arguments, **kwargs):
        arguments = list(arguments)
        arguments[5] = arguments[5].replace(
            "runpy.run_module('tldw_chatbook.MCP.permission_worker',run_name='__main__')",
            f"__import__('Tests.MCP.permission_worker_faults',fromlist=['run']).run({fault!r})",
        )
        return popen(arguments, **kwargs)

    monkeypatch.setattr(process.subprocess, "Popen", launch)
    broker = process.PermissionProcess(source)
    payload = source.load()
    payload["kill_switch"] = True
    try:
        if fault == "command":
            with pytest.raises(process.ProtocolError):
                broker.call("save", payload=payload)
            assert "updated_at" not in payload
            assert json.loads(source.path.read_text())["kill_switch"] is True
            assert source._permission_publication.uncertain
            assert broker.unresolved is not None
            with pytest.raises(process.ProtocolError):
                broker.call("get_kill_switch")
        else:
            assert broker.call("get_kill_switch") is False
        with pytest.raises(process.ProtocolError):
            broker.close()
        fault_path = source.path.parents[2] / "permission-fault.json"
        assert fault_path.exists()
        witness = json.loads(fault_path.read_text())
        assert witness["pid"] == broker.process.pid
        if fault == "command":
            assert witness["disk_effect"] and witness["deferred_stamps"] == 1
        else:
            assert witness["native_close_attempted"]
    finally:
        broker.contain_unresolved() if broker.unresolved else broker.close()


@pytest.mark.asyncio
async def test_repeated_cancellation_waits_for_real_selected_operation(
    source, monkeypatch
):
    from tldw_chatbook.MCP import permission_process as process
    from tldw_chatbook.MCP.console_snapshot import _owned_worker

    broker = process.PermissionProcess(source)
    entered, release = threading.Event(), threading.Event()
    read = process.read_message

    def reading(stream):
        message = read(stream)
        if (
            message.get("type") == "Check"
            and message["data"]["site"] == "protected_open"
            and not entered.is_set()
        ):
            entered.set()
            assert release.wait(10)
        return message

    monkeypatch.setattr(process, "read_message", reading)
    task = asyncio.create_task(_owned_worker(lambda: broker.call("get_kill_switch")))
    try:
        for _ in range(1000):
            if entered.is_set():
                break
            await asyncio.sleep(0.01)
        assert entered.is_set()
        for _ in range(3):
            task.cancel()
            await asyncio.sleep(0.01)
            assert not task.done() and broker.lifetime.calls
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not broker.lifetime.calls and broker.unresolved is None
        assert broker.receipts[-1]["retired"]
    finally:
        release.set()
        if not task.done():
            await asyncio.gather(task, return_exceptions=True)
        broker.contain_unresolved() if broker.unresolved else broker.close()


@pytest.mark.asyncio
async def test_direct_broker_census_drains_startup_and_resumes_lazily(source):
    from tldw_chatbook.MCP import permission_process as process
    from tldw_chatbook.Backup_Recovery import runtime_maintenance as maintenance
    from tldw_chatbook.Backup_Recovery import storage_admission as storage
    from tldw_chatbook.Backup_Recovery.bootstrap import RecoveryRequired

    broker = process.PermissionProcess(source)
    try:
        broker.call("get_kill_switch")
        first_pid = broker.process.pid
        assert hasattr(
            process, "participant"
        ), "direct broker missing from maintenance census"
        participant = process.participant
        assert maintenance._permission_process_owner() is participant
        runtime = object.__new__(maintenance.RuntimeMaintenance)
        runtime.task = asyncio.current_task()
        runtime._settled, runtime.pause = True, None
        with pytest.raises(RecoveryRequired, match="runtime_producers_not_settled"):
            runtime.retire_local_caches()
        assert storage._pause is None
        hook = maintenance._bind(
            participant, "MCP.permission_process", "PermissionProcessLifetime"
        )
        closed = []
        await maintenance._settle_stage([hook], closed, time.monotonic() + 10)
        assert broker.retired and broker.process.poll() == 0
        with pytest.raises((process.ProtocolError, RecoveryRequired)):
            broker.call("get_kill_switch")
        await maintenance._resume_hooks(closed)
        assert broker.process is None
        assert broker.call("get_kill_switch") is False
        assert broker.process.pid != first_pid
    finally:
        broker.contain_unresolved() if broker.unresolved else broker.close()


def test_repair_invalidates_before_backup_and_captured_load_does_not_publish(
    source, monkeypatch
):
    from tldw_chatbook.MCP import permission_process as process

    source.load()
    source.path.write_text("{bad json", encoding="utf-8")
    broker = process.PermissionProcess(source)
    events = []
    write = process.write_message

    def writing(stream, message):
        if message.get("type") == "Acknowledge":
            events.append(source._permission_publication.pending)
            assert source._permission_publication.token is None
            assert source.path.read_text() == "{bad json"
        return write(stream, message)

    monkeypatch.setattr(process, "write_message", writing)
    try:
        payload, receipt = broker.captured_load(lambda: None)
        # Idle brokers retain weak receiver metadata; the receipt is reconstructed
        # with the exact original parent objects, rather than retaining the source.
        assert receipt == broker.capture.receipt
        assert payload["kill_switch"] is False and events == [True]
        assert source.path.with_suffix(".json.bak").read_text() == "{bad json"
        assert not source._permission_publication.pending
        assert source._permission_publication.token is None
    finally:
        broker.contain_unresolved() if broker.unresolved else broker.close()


def test_lost_terminal_reply_after_save_never_stamps_or_replays(
    source, monkeypatch, process_audit
):
    from tldw_chatbook.MCP import permission_process as process
    from tldw_chatbook.MCP.permission_store import MCPPermissionStore

    sibling = MCPPermissionStore(source.path)
    owned_source = MCPPermissionStore(source.path)
    payload = source.load()
    payload["kill_switch"] = True
    broker = process.PermissionProcess(owned_source)
    read = process.read_message
    terminals = []

    def reading(stream):
        message = read(stream)
        if message.get("type") == "Settled":
            terminals.append(message)
            raise process.ProtocolError("injected_lost_response")
        return message

    monkeypatch.setattr(process, "read_message", reading)
    try:
        with pytest.raises(process.ProtocolError, match="lost_response"):
            broker.call("save", payload=payload)
        assert len(terminals) == 1 and "updated_at" not in payload
        disk = source.path.read_bytes()
        assert json.loads(disk)["kill_switch"] is True
        assert sibling._mcp_persistence_error == "mcp_persistence_incomplete"
        with pytest.raises(process.ProtocolError):
            broker.call("save", payload=payload)
        with pytest.raises(process.ProtocolError):
            process.PermissionProcess(source)
        assert source.path.read_bytes() == disk and len(terminals) == 1
        broker_ref, source_ref = weakref.ref(broker), weakref.ref(owned_source)
        process_audit.remove(broker)
        del broker, owned_source
        gc.collect()
        assert broker_ref() in process._brokers and source_ref() is not None
        with pytest.raises(process.ProtocolError):
            process.PermissionProcess(sibling)
        broker = broker_ref()
        process_audit.append(broker)
    finally:
        broker.contain_unresolved() if broker.unresolved else broker.close()


def test_original_generation_cas_refuses_without_poison_or_caller_stamp(source):
    from tldw_chatbook.MCP import permission_process as process
    from tldw_chatbook.MCP.permission_store import ProfileMutationError

    payload = source.load()
    broker = process.PermissionProcess(source)
    try:
        with pytest.raises(ProfileMutationError, match="stale_store"):
            broker.call("save", payload=payload, expected_generation="wrong")
        assert broker.unresolved is None and "updated_at" not in payload
        assert not source.path.exists()
        assert broker.call("get_kill_switch") is False
        snapshot = source.read_snapshot_strict()
        broker.call("save", payload=payload, expected_generation=snapshot.generation)
        assert "updated_at" in payload
    finally:
        broker.contain_unresolved() if broker.unresolved else broker.close()


def test_external_mutation_fence_reserves_broker_until_composite_exits(source):
    from tldw_chatbook.MCP import permission_process as process

    broker = process.PermissionProcess(source)
    started, done = threading.Event(), threading.Event()
    outcomes = []

    def call():
        started.set()
        try:
            outcomes.append(broker.call("get_kill_switch"))
        finally:
            done.set()

    thread = threading.Thread(target=call)
    try:
        with source.mutation_fence():
            thread.start()
            assert started.wait(2)
            assert not done.wait(0.05) and broker.process is None
        thread.join(10)
        assert not thread.is_alive() and outcomes == [False]
    finally:
        thread.join(10)
        broker.contain_unresolved() if broker.unresolved else broker.close()


def test_save_preserves_strict_snapshot_error_after_positive_retirement(source):
    from tldw_chatbook.MCP import permission_process as process
    from tldw_chatbook.MCP.permission_store import PermissionStoreSnapshotError

    payload = source.load()
    source.path.write_text("{bad json", encoding="utf-8")
    broker = process.PermissionProcess(source)
    try:
        with pytest.raises(PermissionStoreSnapshotError, match="invalid_json"):
            broker.call("save", payload=payload, expected_generation="old")
        assert broker.unresolved is None and "updated_at" not in payload
        assert source.path.read_text() == "{bad json"
    finally:
        broker.contain_unresolved() if broker.unresolved else broker.close()
