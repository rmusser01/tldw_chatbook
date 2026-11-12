"""Finite original callback capture controls for the connected native relocation."""

from __future__ import annotations

import asyncio
import sys
import threading
from types import MethodType

import pytest

from Tests.Chat.test_console_first_send_atomicity import _controller, _until
from Tests.Chat.test_console_native_commit_integration import (
    _hold_original_commit,
    _saved_state,
)
from tldw_chatbook.Chat import console_native_commit as native
from tldw_chatbook.Chat.console_chat_store import ConsoleDurableAcceptanceFingerprint

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.bootstrap_profile,
    pytest.mark.requires_cleanup,
]


@pytest.mark.parametrize("custom", [False, True])
async def test_native_settlement_keeps_actual_captured_helper_or_custom_callback(
    tmp_path, monkeypatch, owned_console_databases, custom
):
    """A later module slot cannot replace a held stock native callback body."""
    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    original_helper = native._store_settle_accepted_durable_turn
    original_runner = native._run_owned_native_call
    entered, release = asyncio.Event(), asyncio.Event()
    captured = []
    bindings = []
    custom_calls = []
    custom_error = RuntimeError("original custom terminal callback refused")
    if custom:

        def custom_settlement(*args, **kwargs):
            custom_calls.append((args, kwargs))
            raise custom_error

        monkeypatch.setattr(store, "settle_accepted_durable_turn", custom_settlement)

    async def hold_original_runner(binding, callback):
        if isinstance(binding.turn_owner, ConsoleDurableAcceptanceFingerprint):
            # The actual original settle closure captures its original callback.
            actual = dict(zip(callback.__code__.co_freevars, callback.__closure__))[
                "callback"
            ].cell_contents
            captured.append(actual)
            bindings.append(binding)
            entered.set()
            await release.wait()
        return await original_runner(binding, callback)

    monkeypatch.setattr(native, "_run_owned_native_call", hold_original_runner)
    commit_entered, commit_release, commit_exited, commits = _hold_original_commit(
        monkeypatch, store, after_commit=True
    )
    baseline_connections = db.registered_connection_count()
    submit = asyncio.create_task(
        controller.submit_draft("captured once", session_id="session-1")
    )
    successor_calls = []

    def successor_helper(*args, **kwargs):
        successor_calls.append((args, kwargs))
        raise AssertionError("successor helper replaced a captured native body")

    try:
        assert await _until(commit_entered.is_set, timeout=5)
        assert _saved_state(tmp_path / "controller.sqlite")[0] == [("accepted",)]
        submit.cancel()
        await asyncio.sleep(0.05)
        commit_release.set()
        await asyncio.wait_for(entered.wait(), 5)
        assert len(captured) == 1
        if custom:
            assert captured[0] is custom_settlement
        else:
            assert isinstance(captured[0], MethodType)
            assert captured[0].__self__ is store
            assert captured[0].__func__ is type(store).settle_accepted_durable_turn
        assert bindings[0].settlement_helper is original_helper
        monkeypatch.setattr(
            native, "_store_settle_accepted_durable_turn", successor_helper
        )
        release.set()
        result = await asyncio.wait_for(asyncio.shield(submit), 5)
        assert (
            result.accepted
            and not result.provider_started
            and not result.should_clear_draft
        )
        assert len(commits) == 1 and commit_exited.is_set()
        assert gateway.calls == 0 and successor_calls == []
        assert db.registered_connection_count() == baseline_connections
        checkpoints, rows = _saved_state(tmp_path / "controller.sqlite")
        assert len(rows) == 2
        if custom:
            assert len(custom_calls) == 1
            assert checkpoints == [("accepted",)]
            recovery = store.dispatch_recovery_for_session("session-1")
            assert recovery is not None and recovery.recovery_needed
        else:
            assert checkpoints == []
            assert next(row[2] for row in rows if row[0] == "assistant") == "failed"
    finally:
        commit_release.set()
        release.set()
        await asyncio.gather(submit, return_exceptions=True)


# Reuse the existing actual Workspace reader lifetime fixture, including its
# original module/registry/private-source dependencies and physical retirement.
from Tests.Chat import (
    test_console_configuration_worker_lifetime as configuration_controls,
)

catalog_store = configuration_controls.catalog_store
local_root = configuration_controls.local_root
mcp_sources = configuration_controls.mcp_sources
snapshot_case = configuration_controls.snapshot_case
runtime_case = configuration_controls.runtime_case
configuration_workspace = configuration_controls.configuration_workspace


@pytest.mark.parametrize("route", ["runtime", "received"])
async def test_started_configuration_keeps_original_helper_during_native_read(
    configuration_workspace, monkeypatch, route
):
    from Tests.Chat.test_console_configuration_preparation import _selection
    from tldw_chatbook.Chat import console_configuration_preparation as preparation
    from tldw_chatbook.Chat.console_preparation_reads import preparation_reads_for

    case = configuration_workspace
    original = preparation._controller_capture_turn_configuration_snapshot
    code = original.__code__
    defaults = original.__defaults__
    keyword_defaults = tuple((original.__kwdefaults__ or {}).items())
    probe = configuration_controls._OriginalConfigurationWorkspaceRead(case.registry)
    task = None
    successor_calls = []

    async def successor(*args, **kwargs):
        successor_calls.append((args, kwargs))
        raise AssertionError(
            "successor helper replaced an executing configuration body"
        )

    kwargs = {} if route == "runtime" else {"selection": _selection(case)}
    with probe.installed():
        try:
            task = asyncio.create_task(
                case.controller.capture_turn_configuration_snapshot(
                    case.session.id, **kwargs
                )
            )
            await configuration_controls._wait_for_original_reader(probe, task)
            (read,) = preparation_reads_for(
                case.controller._preparation_reads, case.session.id
            )
            assert read.task is task and read in case.runtime._preparation_reads
            assert not read.retired.done() and not probe.retired()
            monkeypatch.setattr(
                preparation,
                "_controller_capture_turn_configuration_snapshot",
                successor,
            )
            probe.release.set()
            captured = await asyncio.wait_for(asyncio.shield(task), 5)
            assert captured.session_id == case.session.id
            assert captured.provider_selection.provider == "deepseek"
            assert captured.session_settings == case.store.effective_session_settings(
                case.session.id
            )
            assert successor_calls == []
            assert original.__code__ is code and original.__defaults__ is defaults
            assert tuple((original.__kwdefaults__ or {}).items()) == keyword_defaults
            assert read.retired.done() and probe.retired()
            assert (
                preparation_reads_for(case.runtime._preparation_reads, case.session.id)
                == ()
            )
        finally:
            probe.release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            if probe.entered.is_set():
                assert await configuration_controls._until(probe.retired, 5)


@pytest.mark.parametrize("when", ["preinstalled", "held"])
@pytest.mark.parametrize("expected", ["{}", "[]"])
async def test_original_omitted_metadata_uses_live_captured_method_default(
    tmp_path, monkeypatch, owned_console_databases, when, expected
):
    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    method = type(store).settle_accepted_durable_turn
    code = native._store_settle_accepted_durable_turn.__code__
    recorded = []
    errors = []

    def starting(actual_code, offset):
        try:
            frame = sys._getframe()
            while frame is not None and frame.f_code is not actual_code:
                frame = frame.f_back
            assert frame is not None
            if frame.f_locals.get("self") is store:
                recorded.append(
                    {
                        "metadata_json": frame.f_locals["metadata_json"],
                        "thread": threading.get_ident(),
                    }
                )
        except BaseException as error:  # noqa: BLE001 -- passive observation records errors without interrupting native work.
            errors.append(type(error).__name__)

    monitor = sys.monitoring
    slot = next(
        i
        for i in range(5, 0, -1)
        if i != monitor.DEBUGGER_ID and monitor.get_tool(i) is None
    )
    monitor.use_tool_id(slot, "finite-original-metadata-default-abi")
    monitor.register_callback(slot, monitor.events.PY_START, starting)
    monitor.set_local_events(slot, code, monitor.events.PY_START)
    original = native._run_owned_native_call
    entered, release = asyncio.Event(), asyncio.Event()

    async def hold(binding, callback):
        if isinstance(binding.turn_owner, ConsoleDurableAcceptanceFingerprint):
            entered.set()
            await release.wait()
        return await original(binding, callback)

    monkeypatch.setattr(native, "_run_owned_native_call", hold)

    def change_default():
        monkeypatch.setattr(
            method,
            "__kwdefaults__",
            {**method.__kwdefaults__, "metadata_json": expected},
        )

    if when == "preinstalled":
        change_default()
    commit_entered, commit_release, commit_exited, commits = _hold_original_commit(
        monkeypatch, store, after_commit=True
    )
    baseline = db.registered_connection_count()
    task = asyncio.create_task(
        controller.submit_draft("actual default", session_id="session-1")
    )
    try:
        assert await _until(commit_entered.is_set, timeout=5)
        assert _saved_state(tmp_path / "controller.sqlite")[0] == [("accepted",)]
        task.cancel()
        await asyncio.sleep(0.05)
        commit_release.set()
        await asyncio.wait_for(entered.wait(), 5)
        if when == "held":
            change_default()
        assert method.__kwdefaults__["metadata_json"] == expected
        release.set()
        result = await asyncio.wait_for(asyncio.shield(task), 5)
        assert result.accepted and not result.provider_started
        assert len(commits) == 1 and commit_exited.is_set() and gateway.calls == 0
        assert db.registered_connection_count() == baseline
        assert errors == [] and len(recorded) == 1
        assert recorded[0]["thread"] != threading.get_ident()
        checkpoints, rows = _saved_state(tmp_path / "controller.sqlite")
        if expected == "[]":
            assert checkpoints == [("accepted",)], (
                "Original nonobject-JSON default must retain accepted recovery instead of writing terminal state"
            )
            assert next(row[2] for row in rows if row[0] == "assistant") == "accepted"
        else:
            assert checkpoints == []
            assert next(row[2] for row in rows if row[0] == "assistant") == "failed"
        assert recorded[0]["metadata_json"] == expected, recorded
    finally:
        commit_release.set()
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        monitor.set_local_events(slot, code, 0)
        monitor.register_callback(slot, monitor.events.PY_START, None)
        monitor.free_tool_id(slot)


async def test_explicit_native_metadata_wins_over_original_nonobject_default(
    tmp_path, monkeypatch, owned_console_databases
):
    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    method = type(store).settle_accepted_durable_turn
    monkeypatch.setattr(
        method, "__kwdefaults__", {**method.__kwdefaults__, "metadata_json": "[]"}
    )
    original = native._run_owned_native_call
    entered, release = asyncio.Event(), asyncio.Event()
    captured = []

    async def hold_once(binding, callback):
        if (
            isinstance(binding.turn_owner, ConsoleDurableAcceptanceFingerprint)
            and not entered.is_set()
        ):
            captured.append(binding)
            entered.set()
            await release.wait()
        return await original(binding, callback)

    monkeypatch.setattr(native, "_run_owned_native_call", hold_once)
    commit_entered, commit_release, commit_exited, commits = _hold_original_commit(
        monkeypatch, store, after_commit=True
    )
    task = asyncio.create_task(
        controller.submit_draft("explicit metadata", session_id="session-1")
    )
    try:
        assert await _until(commit_entered.is_set, timeout=5)
        task.cancel()
        await asyncio.sleep(0.05)
        commit_release.set()
        await asyncio.wait_for(entered.wait(), 5)
        assert len(captured) == 1 and commit_exited.is_set()
        owner = controller._ordinary_native_commit_owner(
            next(iter(controller._ordinary_native_commits))
        )
        preparation_id = owner.acceptance.preparation_id
        fingerprint = store.durable_acceptance_fingerprint_for(preparation_id)
        assert fingerprint is captured[0].turn_owner
        assert controller._preparation_by_id(preparation_id).state.value == "accepted"
        baseline = db.registered_connection_count()
        completed = await native.settle_accepted_durable_turn_owned(
            store,
            preparation_id,
            fingerprint=fingerprint,
            terminal_state="failed",
            content="Explicit original metadata",
            persistence=store.persistence,
            database=store.persistence.db,
            metadata_json="{}",
        )
        assert (
            completed.settled
            and completed.error is None
            and not completed.caller_cancelled
        )
        assert db.registered_connection_count() == baseline
        checkpoints, rows = _saved_state(tmp_path / "controller.sqlite")
        assert checkpoints == []
        assistant = next(row for row in rows if row[0] == "assistant")
        assert assistant[1] == "Explicit original metadata" and assistant[2] == "failed"
        assert gateway.calls == 0
    finally:
        commit_release.set()
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    assert commit_exited.is_set() and len(commits) == 1 and gateway.calls == 0


async def test_native_settlement_keeps_captured_body_after_helper_slot_is_deleted(
    tmp_path, monkeypatch, owned_console_databases
):
    """Deleting a late module slot cannot discard an already captured body."""
    db, store, controller, gateway = _controller(tmp_path)
    owned_console_databases(db, controller)
    original_helper = native._store_settle_accepted_durable_turn
    original_runner = native._run_owned_native_call
    entered, release = asyncio.Event(), asyncio.Event()
    captured, outcomes = [], []

    async def hold(binding, callback):
        terminal = isinstance(binding.turn_owner, ConsoleDurableAcceptanceFingerprint)
        if terminal:
            captured.append(binding)
            entered.set()
            await release.wait()
        outcome = await original_runner(binding, callback)
        if terminal:
            outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(native, "_run_owned_native_call", hold)
    commit_entered, commit_release, commit_exited, commits = _hold_original_commit(
        monkeypatch, store, after_commit=True
    )
    baseline = db.registered_connection_count()
    task = asyncio.create_task(controller.submit_draft("held deleted slot", session_id="session-1"))
    try:
        assert await _until(commit_entered.is_set, timeout=5)
        task.cancel()
        await asyncio.sleep(0.05)
        commit_release.set()
        await asyncio.wait_for(entered.wait(), 5)
        assert len(captured) == 1
        assert captured[0].settlement_helper is original_helper
        monkeypatch.delattr(native, "_store_settle_accepted_durable_turn")
        release.set()
        result = await asyncio.wait_for(asyncio.shield(task), 5)
        assert result.accepted and not result.provider_started and not result.should_clear_draft
        assert len(commits) == 1 and commit_exited.is_set() and gateway.calls == 0
        assert db.registered_connection_count() == baseline
        assert len(outcomes) == 1
        assert outcomes[0][0] is True and outcomes[0][1] is None, outcomes
        checkpoints, rows = _saved_state(tmp_path / "controller.sqlite")
        assert checkpoints == [] and len(rows) == 2
        assert next(row[2] for row in rows if row[0] == "assistant") == "failed"
    finally:
        # Restore the actual original slot after physical callback retirement.
        commit_release.set()
        release.set()
        await asyncio.gather(task, return_exceptions=True)
        monkeypatch.setattr(native, "_store_settle_accepted_durable_turn", original_helper, raising=False)
