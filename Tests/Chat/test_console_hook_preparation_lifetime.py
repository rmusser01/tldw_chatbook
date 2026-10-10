"""Finite original hook reads retain received Send custody until native return."""

from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import sys
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest

from Tests.Agents.test_hook_permissions import hook_file as _hook_file
from Tests.Chat.test_console_hook_preparation_demand import _save_hooks
from Tests.Chat.test_console_prompt_queue_coordinator import SequencedGateway
from Tests.UI.test_console_hook_refresh_lifetime import _OriginalVisit, _until
from tldw_chatbook.Chat.attachment_core import PendingAttachment
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController
from tldw_chatbook.Chat.console_chat_models import ConsoleProviderSelection
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Chat.console_turn_context import (
    ConsoleTurnConfigurationSnapshot,
    ConsoleTurnCustodyRequest,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]
hook_file = _hook_file


@pytest.fixture
async def preparation_runtime(hook_file):
    _save_hooks(hook_file, {})
    store = ConsoleChatStore()
    session = store.create_session(session_id="hook-preparation", ephemeral=True)
    store.set_session_draft(session.id, "retain this received draft")
    runtime = ConsoleRuntime(app=None)
    owner = runtime.ensure_hook_permissions()
    assert owner.snapshot().ready
    gateway = SequencedGateway()
    controller = ConsoleChatController(
        store=store,
        provider_gateway=gateway,
        provider="llama_cpp",
        model="test-model",
        hook_permissions_accessor=runtime.ensure_hook_permissions,
    )
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    attachment = PendingAttachment(
        file_path="/received-image.png",
        display_name="received-image.png",
        file_type="image",
        insert_mode="attachment",
        attachment_id="received-image",
        data=b"received image",
        mime_type="image/png",
    )
    assert store.add_pending_attachment(session.id, attachment)
    request = ConsoleTurnCustodyRequest(
        turn_id="received-hook-turn",
        session_id=session.id,
        draft=session.draft,
        attachment_ids=(attachment.attachment_id,),
        configuration=ConsoleTurnConfigurationSnapshot.capture(
            session_id=session.id,
            provider_selection=ConsoleProviderSelection(
                provider="llama_cpp", explicit_model="test-model"
            ),
        ),
    )
    state = SimpleNamespace(
        runtime=runtime,
        owner=owner,
        controller=controller,
        store=store,
        session=session,
        gateway=gateway,
        request=request,
        attachment=attachment,
        tasks=[],
    )
    try:
        yield state
    finally:
        for task in state.tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*state.tasks, return_exceptions=True)
        await runtime.dispose()
        owner.close()


async def test_original_admission_snapshot_keeps_received_custody_through_stop_and_cancel(
    preparation_runtime,
):
    state = preparation_runtime
    probe = _OriginalVisit(state.owner)
    task = None
    with probe.installed():
        try:
            turn_id = state.runtime.accept_turn(state.request)
            record = state.runtime._turn_custody[turn_id]
            task = record.task
            state.tasks.append(task)
            claim = state.store.received_turn_for_session(state.session.id)
            assert claim is not None
            assert await _until(
                probe.entered.is_set, 5
            ), "original snapshot not reached"
            probe.assert_live()
            assert record.store is state.store and record.received_claim is claim
            assert record.inputs.attachments == (state.attachment,)
            assert state.controller.stop_active_run()
            assert claim.sealed
            for _ in range(2):
                task.cancel()
                await asyncio.sleep(0.05)
                probe.assert_live()
                assert (
                    not task.done()
                ), "cancellation abandoned original native snapshot"
                assert state.store.received_turn_for_session(state.session.id) is claim
                assert state.runtime._turn_custody[turn_id] is record
                assert record.request is state.request
                assert record.store is state.store and record.received_claim is claim
                assert record.inputs.attachments[0] is state.attachment
                assert any(row is state.session for row in state.store.sessions())
                assert state.gateway.user_turns == []
        finally:
            probe.release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            assert await _until(
                probe.retired, 5
            ), "original native snapshot did not retire"
    probe.assert_retired()
    assert state.store.received_turn_for_session(state.session.id) is None
    assert not state.runtime.has_custodied_turns(state.session.id)
    assert state.gateway.user_turns == []
    assert state.session.draft == state.request.draft
    (recovery,) = state.runtime.recoveries_for_session(state.session.id)
    assert recovery.draft == state.request.draft
    assert recovery.attachments == (state.attachment,)
    assert recovery.attachments[0] is state.attachment


async def _stock_request(state):
    """Use the supported full capture before arming the native producer probe."""
    configuration = await state.controller.capture_turn_configuration_snapshot(
        state.session.id
    )
    return replace(
        state.request,
        configuration=replace(
            configuration,
            capabilities={
                **configuration.capabilities,
                "vision": True,
                "max_history_images": 1,
            },
        ),
    )


def _start_received(state, request):
    turn_id = state.runtime.accept_turn(request)
    record = state.runtime._turn_custody[turn_id]
    state.tasks.append(record.task)
    claim = state.store.received_turn_for_session(state.session.id)
    assert claim is not None
    return record, claim


def _assert_received_live(state, record, claim):
    assert not record.task.done()
    assert state.runtime._turn_custody[record.turn_id] is record
    assert record.store is state.store and record.received_claim is claim
    assert state.store.received_turn_for_session(state.session.id) is claim
    assert record.request is not None
    assert record.inputs.attachments[0] is state.attachment
    assert any(row is state.session for row in state.store.sessions())
    assert state.gateway.user_turns == []


async def test_original_v2_configuration_outlives_close_and_repeated_cancellation(
    preparation_runtime, monkeypatch
):
    from tldw_chatbook.Agents.hook_permissions import HookPermissions
    from tldw_chatbook.Chat.console_runtime import CONSOLE_SESSION_CLOSE_GRACE_SECONDS

    state = preparation_runtime
    request = await _stock_request(state)
    probe = _OriginalVisit(state.owner)
    probe.snapshot_code = HookPermissions.v2_configuration.__code__
    original_reason = state.controller.hook_admission_reason
    custom_calls = []

    async def custom_reason():
        # The supported injected callback must keep its zero-argument shape.
        custom_calls.append(True)
        return await original_reason()

    monkeypatch.setattr(state.controller, "hook_admission_reason", custom_reason)
    record = task = closing = None
    with probe.installed():
        try:
            record, claim = _start_received(state, request)
            task = record.task
            assert await _until(
                probe.entered.is_set, 5
            ), "original v2 capture not reached"
            probe.assert_live()
            assert custom_calls
            _assert_received_live(state, record, claim)
            closing = asyncio.create_task(
                state.runtime.close_session(
                    state.session.id,
                    expected_revision=state.controller.lifecycle_impact(
                        session_id=state.session.id
                    ).revision,
                )
            )
            state.tasks.append(closing)
            assert CONSOLE_SESSION_CLOSE_GRACE_SECONDS == 2.0
            await asyncio.sleep(CONSOLE_SESSION_CLOSE_GRACE_SECONDS + 0.05)
            for _ in range(2):
                closing.cancel()
                await asyncio.sleep(0.05)
                assert not closing.done(), "Close retired a live native v2 capture"
                probe.assert_live()
                _assert_received_live(state, record, claim)
                assert claim.sealed
        finally:
            probe.release.set()
            if record is not None:
                await asyncio.gather(task, return_exceptions=True)
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)
            assert await _until(probe.retired, 5)
    probe.assert_retired()
    assert state.gateway.user_turns == []
    assert not state.runtime.has_custodied_turns(state.session.id)
    assert state.store.received_turn_for_session(state.session.id) is None
    assert not any(row is state.session for row in state.store.sessions())


async def test_replaced_controller_cannot_detach_original_native_read_from_dispose(
    preparation_runtime, monkeypatch
):
    from tldw_chatbook.Chat.console_hook_preparation import hook_preparation_reads_for
    from tldw_chatbook.Chat.console_runtime import (
        CONSOLE_RUNTIME_SHUTDOWN_GRACE_SECONDS,
    )

    state = preparation_runtime
    probe = _OriginalVisit(state.owner)
    record = task = closing = None
    ended = []
    original_end = state.store.end_app_runtime

    def end_after_native_return():
        assert probe.retired(), "store ended before original native read returned"
        ended.append(True)
        return original_end()

    monkeypatch.setattr(state.store, "end_app_runtime", end_after_native_return)
    replacement = ConsoleChatController(
        store=state.store,
        provider_gateway=state.gateway,
        hook_permissions_accessor=state.runtime.ensure_hook_permissions,
    )
    with probe.installed():
        try:
            record, claim = _start_received(state, state.request)
            task = record.task
            assert await _until(probe.entered.is_set, 5)
            probe.assert_live()
            (read,) = hook_preparation_reads_for(
                state.runtime._preparation_reads, state.session.id
            )
            assert read in state.controller._preparation_reads
            assert read.task is record.task and not read.retired.done()
            state.runtime.set_chat_controller(replacement)
            assert read in state.runtime._preparation_reads
            closing = asyncio.create_task(state.runtime.dispose())
            state.tasks.append(closing)
            assert CONSOLE_RUNTIME_SHUTDOWN_GRACE_SECONDS == 3.0
            await asyncio.sleep(CONSOLE_RUNTIME_SHUTDOWN_GRACE_SECONDS + 0.05)
            for _ in range(2):
                closing.cancel()
                await asyncio.sleep(0.05)
                assert not closing.done(), "replacement detached the original creator"
                assert ended == []
                assert not read.retired.done()
                probe.assert_live()
                _assert_received_live(state, record, claim)
        finally:
            probe.release.set()
            if record is not None:
                await asyncio.gather(task, return_exceptions=True)
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)
            assert await _until(probe.retired, 5)
            await state.controller.shutdown()
    probe.assert_retired()
    assert read.retired.done() and not read.retired.cancelled()
    assert hook_preparation_reads_for(state.runtime._preparation_reads) == ()
    assert state.gateway.user_turns == []
    assert not state.runtime.has_custodied_turns(state.session.id)


@pytest.mark.parametrize("replacement", ["permission", "store", "session"])
async def test_original_snapshot_refuses_changed_source_without_successor_effects(
    preparation_runtime, replacement
):
    from tldw_chatbook.Agents.hook_permissions import HookPermissions

    state = preparation_runtime
    probe = _OriginalVisit(state.owner)
    record = task = other_owner = successor_claim = None
    successor_store = ConsoleChatStore()
    successor_session = successor_store.create_session(
        session_id=state.session.id, ephemeral=True
    )
    with probe.installed():
        try:
            record, claim = _start_received(state, state.request)
            task = record.task
            assert await _until(probe.entered.is_set, 5)
            probe.assert_live()
            if replacement == "permission":
                other_owner = HookPermissions()
                state.runtime._hook_permissions = other_owner
            elif replacement == "store":
                successor_claim = successor_store.claim_received_turn(
                    state.session.id, "successor-hook-turn"
                )
                assert successor_claim is not None
                state.runtime.set_chat_store(successor_store)
            else:
                # A same-ID replacement must not inherit the original session witness.
                state.store._sessions[state.session.id] = successor_session
            probe.release.set()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
            assert await _until(probe.retired, 5)
            assert state.gateway.user_turns == []
            assert not state.runtime.has_custodied_turns(state.session.id)
            assert not record.inputs.durable_accepted
            assert not state.store.messages_for_session(state.session.id)
            if successor_claim is not None:
                assert (
                    successor_store.received_turn_for_session(state.session.id)
                    is successor_claim
                )
                assert not successor_store.messages_for_session(state.session.id)
            if replacement in {"store", "session"}:
                assert state.runtime.recoveries_for_session(state.session.id) == ()
        finally:
            probe.release.set()
            if record is not None:
                await asyncio.gather(task, return_exceptions=True)
            assert await _until(probe.retired, 5)
            state.runtime.set_chat_store(state.store)
            state.store._sessions[state.session.id] = state.session
            state.runtime._hook_permissions = state.owner
            if successor_claim is not None:
                successor_store.release_received_turn(successor_claim)
            if other_owner is not None:
                other_owner.close()
    probe.assert_retired()


class _OriginalWorkspaceRead:
    """Pause inside the original SQLite reader with its real handle admitted."""

    def __init__(self, runtime, registry):
        self.runtime = runtime
        self.registry = registry
        self.context_code = ConsoleRuntime._hooks_v2_context_key.__code__
        self.reader_code = type(registry).read_change_review_consent.__code__
        self.entered = threading.Event()
        self.release = threading.Event()
        self.returned = threading.Event()
        self.connection = self.participant = self.lease = self.thread = None

    def _line(self, code, _line):
        from tldw_chatbook.Backup_Recovery import storage_admission as storage

        frame = sys._getframe(1)
        if (
            self.entered.is_set()
            or code is not self.reader_code
            or frame.f_locals.get("self") is not self.registry
        ):
            return
        connection = frame.f_locals.get("conn")
        if connection is None:
            return
        parent = frame.f_back
        while parent is not None:
            if (
                parent.f_code is self.context_code
                and parent.f_locals.get("self") is self.runtime
            ):
                break
            parent = parent.f_back
        if parent is None:
            return
        self.connection = connection
        self.thread = threading.current_thread()
        self.participant = self.registry.db._maintenance_participant
        with storage._lock:
            self.lease = self.participant.connections.get(connection)
            assert self.lease in storage._live_leases
            assert self.lease.resource_thread is self.thread
        self.entered.set()
        assert self.release.wait(20), "original workspace reader was not released"

    def _returned(self, code, _offset, _result):
        if code is not self.context_code:
            return
        frame = sys._getframe(1)
        if frame.f_locals.get("self") is self.runtime and self.entered.is_set():
            self.returned.set()

    @contextlib.contextmanager
    def installed(self):
        monitoring = sys.monitoring
        tool = next(value for value in range(6) if monitoring.get_tool(value) is None)
        monitoring.use_tool_id(tool, "hook-preparation-original-workspace")
        try:
            monitoring.register_callback(tool, monitoring.events.LINE, self._line)
            monitoring.register_callback(
                tool, monitoring.events.PY_RETURN, self._returned
            )
            monitoring.register_callback(
                tool, monitoring.events.PY_UNWIND, self._returned
            )
            monitoring.set_local_events(tool, self.reader_code, monitoring.events.LINE)
            monitoring.set_local_events(
                tool, self.context_code, monitoring.events.PY_RETURN
            )
            monitoring.set_events(tool, monitoring.events.PY_UNWIND)
            yield
        finally:
            self.release.set()
            monitoring.set_events(tool, 0)
            monitoring.set_local_events(tool, self.reader_code, 0)
            monitoring.set_local_events(tool, self.context_code, 0)
            for event in (
                monitoring.events.LINE,
                monitoring.events.PY_RETURN,
                monitoring.events.PY_UNWIND,
            ):
                monitoring.register_callback(tool, event, None)
            monitoring.free_tool_id(tool)

    def _closed(self):
        if self.connection is None:
            return True
        try:
            sqlite3.Connection.in_transaction.__get__(self.connection)
        except sqlite3.ProgrammingError:
            return True
        return False

    def assert_live(self):
        from tldw_chatbook.Backup_Recovery import storage_admission as storage

        assert self.connection is not None and not self._closed()
        assert self.thread is not threading.current_thread()
        with storage._lock:
            assert self.lease in storage._live_leases
            assert self.participant.connections.get(self.connection) is self.lease
        assert not self.returned.is_set()

    def retired(self):
        from tldw_chatbook.Backup_Recovery import storage_admission as storage

        if self.connection is None:
            return True
        with storage._lock:
            return (
                self.returned.is_set()
                and self._closed()
                and self.lease not in storage._live_leases
                and self.connection not in self.participant.connections
            )


async def test_original_workspace_context_outlives_dispose_cancelled_during_grace(
    preparation_runtime, monkeypatch
):
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_hook_preparation import hook_preparation_reads_for
    from tldw_chatbook.Chat.console_runtime import (
        CONSOLE_RUNTIME_SHUTDOWN_GRACE_SECONDS,
    )
    from tldw_chatbook.DB.Workspace_DB import WorkspaceDB
    from tldw_chatbook.Workspaces.change_review_consent import (
        ChangeReviewConsentService,
    )
    from tldw_chatbook.Workspaces.registry_service import LocalWorkspaceRegistryService

    state = preparation_runtime
    database = WorkspaceDB(
        config.get_user_data_dir() / "hook-preparation.sqlite",
        client_id="hook-lifetime",
    )
    registry = LocalWorkspaceRegistryService(database)
    registry.create_workspace(
        workspace_id="hook-owned", name="Hook owned", assistant_defaults=None
    )
    consent = ChangeReviewConsentService(registry)
    app = SimpleNamespace(
        app_config=config.load_settings(),
        workspace_registry_service=registry,
        change_review_consent_service=consent,
    )
    state.runtime._app = app
    state.runtime.set_chat_controller(state.controller)
    state.session.workspace_id = "hook-owned"
    record = task = closing = None
    probe = _OriginalWorkspaceRead(state.runtime, registry)
    grace_entered = asyncio.Event()
    original_wait = state.runtime._bounded_wait
    ended = []
    original_end = state.store.end_app_runtime

    def end_after_native_return():
        assert probe.retired(), "store ended before original workspace read returned"
        ended.append(True)
        return original_end()

    monkeypatch.setattr(state.store, "end_app_runtime", end_after_native_return)

    async def observe_original_grace(*args, **kwargs):
        grace_entered.set()
        return await original_wait(*args, **kwargs)

    monkeypatch.setattr(state.runtime, "_bounded_wait", observe_original_grace)
    try:
        request = await _stock_request(state)
        database.close()
        # A retained real owner requires the .12 context comparison even with no hooks.
        state.runtime.ensure_hooks_v2(state.session.id, (), lambda *_: True)
        with probe.installed():
            try:
                record, claim = _start_received(state, request)
                task = record.task
                assert await _until(
                    probe.entered.is_set, 5
                ), "original workspace reader not reached"
                probe.assert_live()
                (read,) = hook_preparation_reads_for(
                    state.runtime._preparation_reads, state.session.id
                )
                assert read.task is task and not read.retired.done()
                closing = asyncio.create_task(state.runtime.dispose())
                state.tasks.append(closing)
                await asyncio.wait_for(grace_entered.wait(), 5)
                closing.cancel()
                await asyncio.sleep(0.05)
                assert not closing.done()
                probe.assert_live()
                _assert_received_live(state, record, claim)
                assert CONSOLE_RUNTIME_SHUTDOWN_GRACE_SECONDS == 3.0
                await asyncio.sleep(CONSOLE_RUNTIME_SHUTDOWN_GRACE_SECONDS + 0.05)
                closing.cancel()
                await asyncio.sleep(0.05)
                assert not closing.done(), "dispose abandoned a live WorkspaceDB read"
                assert ended == []
                assert not read.retired.done()
                probe.assert_live()
                _assert_received_live(state, record, claim)
            finally:
                probe.release.set()
                if task is not None:
                    await asyncio.gather(task, return_exceptions=True)
                if closing is not None:
                    await asyncio.gather(closing, return_exceptions=True)
                assert await _until(probe.retired, 5)
        assert read.retired.done() and not read.retired.cancelled()
        assert hook_preparation_reads_for(state.runtime._preparation_reads) == ()
        assert state.gateway.user_turns == []
        assert not state.runtime.has_custodied_turns(state.session.id)
    finally:
        probe.release.set()
        await state.runtime.dispose()
        consent.shutdown()
        database.close()


async def test_completed_native_read_does_not_extend_unrelated_custom_submit_tail(
    preparation_runtime, monkeypatch
):
    from tldw_chatbook.Chat.console_chat_controller import ConsoleSubmitResult
    from tldw_chatbook.Chat.console_hook_preparation import (
        drain_hook_preparation_reads,
        hook_preparation_reads_for,
    )

    state = preparation_runtime
    tail_entered, tail_release = asyncio.Event(), asyncio.Event()
    probe = _OriginalVisit(state.owner)
    task = closing = None

    async def custom_submit(*_args, **_kwargs):
        assert await state.controller.hook_admission_reason() is None
        tail_entered.set()
        while not tail_release.is_set():
            try:
                await tail_release.wait()
            except asyncio.CancelledError:
                pass  # Deliberately unrelated, cancellation-resistant adapter work.
        return ConsoleSubmitResult(False, False, "custom work finished")

    monkeypatch.setattr(state.controller, "submit_draft", custom_submit)
    with probe.installed():
        try:
            record, _claim = _start_received(state, state.request)
            task = record.task
            assert await _until(probe.entered.is_set, 5)
            probe.assert_live()
            (read,) = hook_preparation_reads_for(state.runtime._preparation_reads)
            probe.release.set()
            await asyncio.wait_for(tail_entered.wait(), 5)
            assert await _until(probe.retired, 5)
            assert read.retired.done() and not read.retired.cancelled()
            assert not task.done()
            assert (
                await asyncio.wait_for(
                    drain_hook_preparation_reads(state.runtime._preparation_reads),
                    0.5,
                )
                is False
            )
            closing = asyncio.create_task(state.runtime.dispose())
            state.tasks.append(closing)
            await asyncio.wait_for(asyncio.shield(closing), 5)
            assert not task.done(), "the custom tail must still be independent work"
            assert not state.runtime.has_custodied_turns(state.session.id)
            assert state.store.received_turn_for_session(state.session.id) is None
            assert state.gateway.user_turns == []
        finally:
            probe.release.set()
            tail_release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)
            assert await _until(probe.retired, 5)
    probe.assert_retired()


async def test_attaching_then_replacing_standalone_creator_preserves_native_retirement(
    preparation_runtime,
):
    from tldw_chatbook.Chat.console_hook_preparation import hook_preparation_reads_for
    from tldw_chatbook.Chat.console_runtime import (
        CONSOLE_RUNTIME_SHUTDOWN_GRACE_SECONDS,
    )

    state = preparation_runtime
    standalone = ConsoleChatController(
        store=state.store,
        provider_gateway=state.gateway,
        hook_permissions_accessor=lambda: state.owner,
    )
    replacement = ConsoleChatController(
        store=state.store,
        provider_gateway=state.gateway,
        hook_permissions_accessor=state.runtime.ensure_hook_permissions,
    )
    probe = _OriginalVisit(state.owner)
    task = closing = None
    with probe.installed():
        try:
            task = asyncio.create_task(standalone.hook_admission_reason())
            state.tasks.append(task)
            assert await _until(probe.entered.is_set, 5)
            probe.assert_live()
            (read,) = hook_preparation_reads_for(standalone._preparation_reads)
            assert read.task is task and read.session_id is None
            assert not state.runtime.has_custodied_turns()
            assert hook_preparation_reads_for(state.runtime._preparation_reads) == ()
            state.runtime.set_chat_controller(standalone)
            assert hook_preparation_reads_for(state.runtime._preparation_reads) == (
                read,
            )
            state.runtime.set_chat_controller(replacement)
            assert hook_preparation_reads_for(state.runtime._preparation_reads) == (
                read,
            )
            closing = asyncio.create_task(state.runtime.dispose())
            state.tasks.append(closing)
            await asyncio.sleep(CONSOLE_RUNTIME_SHUTDOWN_GRACE_SECONDS + 0.05)
            for _ in range(2):
                task.cancel()
                closing.cancel()
                await asyncio.sleep(0.05)
                probe.assert_live()
                assert not task.done() and not closing.done()
                assert not read.retired.done()
                assert read in standalone._preparation_reads
                assert read in state.runtime._preparation_reads
                assert not state.runtime.has_custodied_turns()
        finally:
            probe.release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            if closing is not None:
                await asyncio.gather(closing, return_exceptions=True)
            assert await _until(probe.retired, 5)
            await standalone.shutdown()
            await replacement.shutdown()
            await state.controller.shutdown()
    probe.assert_retired()
    assert read.retired.done() and not read.retired.cancelled()
    assert hook_preparation_reads_for(standalone._preparation_reads) == ()
    assert hook_preparation_reads_for(state.runtime._preparation_reads) == ()
    assert state.gateway.user_turns == []


async def test_cold_runtime_hook_preparation_needs_no_controller_or_store(hook_file):
    from tldw_chatbook.Chat.console_hook_preparation import hook_preparation_reads_for

    _save_hooks(hook_file, {})
    runtime = ConsoleRuntime(app=None)
    try:
        assert runtime._chat_controller is None and runtime._chat_store is None
        assert await runtime.prepare_hooks_v2("standalone-session") is None
        assert runtime._chat_controller is None and runtime._chat_store is None
        assert hook_preparation_reads_for(runtime._preparation_reads) == ()
        assert runtime.ensure_hook_permissions().snapshot().ready
        assert runtime.get_hooks_v2("standalone-session") is None
    finally:
        await runtime.dispose()
    assert runtime._disposed
    assert hook_preparation_reads_for(runtime._preparation_reads) == ()


@pytest.mark.parametrize("replacement", ["app", "context_provider"])
async def test_original_v2_capture_refuses_changed_controller_input_before_context(
    preparation_runtime, replacement, monkeypatch
):
    from tldw_chatbook.Agents.hook_permissions import HookPermissions

    state = preparation_runtime
    request = await _stock_request(state)
    state.runtime.ensure_hooks_v2(state.session.id, (), lambda *_: True)
    # The stock route now shares its admission read with v2 preparation
    # (ADR-225 decision 3; Tests/Chat/test_console_send_hook_read_sharing.py).
    # A supported custom admission callback keeps the fresh v2 capture whose
    # native read this control holds open.
    original_reason = state.controller.hook_admission_reason

    async def custom_reason():
        return await original_reason()

    monkeypatch.setattr(state.controller, "hook_admission_reason", custom_reason)
    probe = _OriginalVisit(state.owner)
    probe.snapshot_code = HookPermissions.v2_configuration.__code__
    original_observe = probe.observe
    context_code = ConsoleRuntime._hooks_v2_context_key.__code__
    context_entries, successor_calls = [], []
    original_app = state.controller.app
    original_provider = state.controller._turn_context_provider
    task = None

    def observe(frame, event, arg):
        if (
            event == "call"
            and frame.f_code is context_code
            and frame.f_locals.get("self") is state.runtime
        ):
            context_entries.append(True)
        return original_observe(frame, event, arg)

    def successor_provider(*_args, **_kwargs):
        successor_calls.append(True)
        return request.configuration

    probe.observe = observe
    with probe.installed():
        try:
            record, _claim = _start_received(state, request)
            task = record.task
            assert await _until(
                probe.entered.is_set, 5
            ), "original v2 capture not reached"
            probe.assert_live()
            if replacement == "app":
                state.controller.app = SimpleNamespace(app_config={})
            else:
                state.controller._turn_context_provider = successor_provider
            probe.release.set()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 5)
            assert await _until(probe.retired, 5)
            assert (
                context_entries == []
            ), "source drift reached the next original context reader"
            assert successor_calls == []
            assert state.gateway.user_turns == []
            assert not record.inputs.durable_accepted
            assert not state.runtime.has_custodied_turns(state.session.id)
        finally:
            probe.release.set()
            if task is not None:
                await asyncio.gather(task, return_exceptions=True)
            assert await _until(probe.retired, 5)
            state.controller.app = original_app
            state.controller._turn_context_provider = original_provider
    probe.assert_retired()


async def test_cold_canonical_permission_owner_is_constructed_on_native_worker(
    hook_file,
):
    from tldw_chatbook.Agents.hook_permissions import HookPermissions

    _save_hooks(hook_file, {})
    runtime = ConsoleRuntime(app=None)
    store = ConsoleChatStore()
    store.create_session(ephemeral=True)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=None,
        hook_permissions_accessor=runtime.ensure_hook_permissions,
    )
    runtime.set_chat_store(store)
    runtime.set_chat_controller(controller)
    caller_thread = threading.current_thread()
    constructors, snapshots = [], []
    constructor_code = HookPermissions.__init__.__code__
    snapshot_code = HookPermissions.snapshot.__code__
    accessor_code = ConsoleRuntime.ensure_hook_permissions.__code__

    def observe(frame, event, _arg):
        if event != "call":
            return
        if frame.f_code is constructor_code:
            parent = frame.f_back
            while parent is not None:
                if (
                    parent.f_code is accessor_code
                    and parent.f_locals.get("self") is runtime
                ):
                    constructors.append(threading.current_thread())
                    break
                parent = parent.f_back
        elif frame.f_code is snapshot_code:
            snapshots.append(threading.current_thread())

    previous, previous_threads = sys.getprofile(), threading.getprofile()
    try:
        assert runtime._hook_permissions is None
        threading.setprofile_all_threads(observe)
        assert await controller.hook_admission_reason() is None
        assert len(constructors) == 1 and constructors[0] is not caller_thread
        assert snapshots == constructors
    finally:
        threading.setprofile_all_threads(previous_threads)
        sys.setprofile(previous)
        await runtime.dispose()


async def test_completed_hook_lifecycle_remains_current_after_controller_replacement(
    preparation_runtime,
):
    state = preparation_runtime
    engine = state.runtime.ensure_hooks_v2(state.session.id, (), lambda *_: True)
    first = await state.runtime.prepare_hooks_v2(state.session.id)
    assert first is not None and first.engine is engine and first.live
    assert first.current()
    replacement = ConsoleChatController(
        store=state.store,
        provider_gateway=state.gateway,
        hook_permissions_accessor=state.runtime.ensure_hook_permissions,
    )
    try:
        state.runtime.set_chat_controller(replacement)
        second = await state.runtime.prepare_hooks_v2(state.session.id)
        assert second is first
        assert second.current(), "a completed finite read pinned the old controller"
        scope = second.open_scope()
        try:
            await asyncio.wait_for(second.wait(scope), 5)
        finally:
            second.close_scope(scope)
        assert state.runtime.get_hooks_v2(state.session.id) is engine
        assert state.gateway.user_turns == []
    finally:
        await replacement.shutdown()
        state.runtime.set_chat_controller(state.controller)
