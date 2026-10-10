"""Created-start entry reaches original MCP reads without blocking its owner loop.

Install under Tests/Chat only after the acceptance/plan amendment. The captured
coordinator boundary intentionally prevents provider/App/ledger admission. The
original start method, stock MCP guards and supported async snapshot stay live.
Full durable admission/cleanup remains covered by test_console_chat_start.
"""

import asyncio
import copy
import threading
from types import CodeType, SimpleNamespace

import pytest

from Tests.Chat import test_console_async_mcp_snapshot as snapshot_controls
from tldw_chatbook.Backup_Recovery import bootstrap, storage_admission
from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

local_root = snapshot_controls.local_root
mcp_sources = snapshot_controls.mcp_sources
catalog_store = snapshot_controls.catalog_store
snapshot_case = snapshot_controls.snapshot_case


class _CapturedStart:
    def __init__(self):
        self.loop = asyncio.get_running_loop()
        self.thread = threading.current_thread()
        self.requests = []

    async def start(self, request):
        assert asyncio.get_running_loop() is self.loop
        assert threading.current_thread() is self.thread
        self.requests.append(request)
        return SimpleNamespace(launch_status="not_started", reason="observed_boundary")


class _StartProbe(snapshot_controls._MaximumProbe):
    def __init__(self, case, **kwargs):
        super().__init__(case.source, case.permissions, **kwargs)
        self.controller = case.controller
        codes = [
            value
            for value in ConsoleChatController._start_created_chat.__code__.co_consts
            if type(value) is CodeType and value.co_name == "start"
        ]
        assert len(codes) == 1
        self.start_code = codes[0]
        self.start_tasks = []

    def observe(self, frame, event, arg):
        if (
            event == "call"
            and frame.f_code is self.start_code
            and frame.f_locals.get("self") is self.controller
        ):
            task = asyncio.current_task()
            if task not in self.start_tasks:
                self.start_tasks.append(task)
        super().observe(frame, event, arg)


def _route(case):
    snapshot_controls._loop_projection(case)
    controller = case.controller
    controller._owner_loop = asyncio.get_running_loop()
    original = controller._chat_start
    captured = _CapturedStart()
    controller._chat_start = captured
    target = case.session
    target.persisted_conversation_id = "saved-created-chat"
    target.agent_handoff_revision = 1
    target.agent_handoff_state = "pending"
    target.draft = "approved opening"
    approved = {
        "source_run_id": "approved-source-run",
        "session_id": "approved-source-session",
        "source_incarnation": "approved-source-incarnation",
        "opening_prompt": target.draft,
        "workspace_id": None,
    }
    return original, captured, target, approved


def _assert_retired(probe):
    assert probe.leases
    assert all(lease not in storage_admission._live_leases for lease in probe.leases)


@pytest.mark.asyncio
async def test_created_start_reads_original_native_mcp_sources_off_owner_loop(
    snapshot_case,
):
    case = snapshot_case
    _, captured, target, approved = _route(case)
    probe = _StartProbe(case)
    with probe.installed():
        outcome = await asyncio.to_thread(
            case.controller._start_created_chat, approved, target
        )
    assert outcome == {"launch_status": "not_started", "reason": "observed_boundary"}
    assert len(captured.requests) == 1
    assert probe.read_threads and probe.permission_threads
    assert all(
        thread is not threading.current_thread()
        for thread in probe.read_threads + probe.permission_threads
    ), "actual created-start route still reads native MCP sources on its owner loop"
    request = captured.requests[0]
    assert request.configuration.session_id == target.id
    assert set(request.configuration.mcp_definition_maximum) == {
        "local:one::first",
        "builtin:tldw_chatbook::sample_builtin",
    }
    assert not case.controller._fleet_wake._automatic_primary_claims
    _assert_retired(probe)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["target", "store", "coordinator", "owner_loop", "workspace", "service"]
)
async def test_created_start_refuses_changed_owner_during_actual_native_read(
    snapshot_case, change
):
    case = snapshot_case
    _, captured, target, approved = _route(case)
    original_store, original_service = (
        case.controller.store,
        case.app.unified_mcp_service,
    )
    original_loop = case.controller._owner_loop
    probe = _StartProbe(case, hold=True)
    task = None
    with probe.installed():
        task = asyncio.create_task(
            asyncio.to_thread(case.controller._start_created_chat, approved, target)
        )
        try:
            await snapshot_controls.catalog_controls._worker_entered(probe, task)
            if change == "target":
                case.store._sessions[target.id] = copy.copy(target)
            elif change == "store":
                case.controller.store = copy.copy(case.store)
            elif change == "coordinator":
                case.controller._chat_start = _CapturedStart()
            elif change == "owner_loop":
                case.controller._owner_loop = None
            elif change == "workspace":
                target.workspace_id = "changed-workspace"
            else:
                case.app.unified_mcp_service = SimpleNamespace()
            probe.release.set()
            with pytest.raises(
                bootstrap.RecoveryRequired, match="console_snapshot_owner_changed"
            ):
                await task
            assert not captured.requests
            assert not case.controller._fleet_wake._automatic_primary_claims
            _assert_retired(probe)
        finally:
            await snapshot_controls.catalog_controls._settle(task, probe)
            case.controller.store = original_store
            case.store._sessions[target.id] = target
            case.app.unified_mcp_service = original_service
            case.controller._chat_start = captured
            case.controller._owner_loop = original_loop


@pytest.mark.asyncio
async def test_created_start_refuses_replaced_equal_target_before_native_capture(
    snapshot_case,
):
    case = snapshot_case
    _, captured, target, approved = _route(case)
    replacement = copy.copy(target)
    assert replacement is not target
    assert replacement.incarnation_id == target.incarnation_id
    case.store._sessions[target.id] = replacement
    probe = _StartProbe(case)
    try:
        with probe.installed():
            with pytest.raises(
                bootstrap.RecoveryRequired, match="console_snapshot_owner_changed"
            ):
                await asyncio.to_thread(
                    case.controller._start_created_chat, approved, target
                )
        assert not captured.requests
        assert not probe.read_threads and not probe.permission_threads
        assert not case.controller._fleet_wake._automatic_primary_claims
    finally:
        case.store._sessions[target.id] = target


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["handoff_aba", "context_epoch", "approved_fields"])
async def test_created_start_keeps_pre_await_request_facts_during_native_capture(
    snapshot_case, change
):
    case = snapshot_case
    original, captured, target, approved = _route(case)
    before = dict(approved)
    revision = target.agent_handoff_revision
    epoch = case.store.conversation_context_epoch(target.id)
    probe = _StartProbe(case, hold=True)
    task = None
    with probe.installed():
        task = asyncio.create_task(
            asyncio.to_thread(case.controller._start_created_chat, approved, target)
        )
        try:
            await snapshot_controls.catalog_controls._worker_entered(probe, task)
            if change == "handoff_aba":
                # Explicit owner-state interleaving; actual edit persistence and
                # writer cleanup stay in the original full native-start tests.
                target.draft = "intervening edit"
                target.agent_handoff_revision += 1
                target.draft = before["opening_prompt"]
                target.agent_handoff_revision += 1
            elif change == "context_epoch":
                case.store._bump_conversation_context_epoch(target.id)
            else:
                approved.update(
                    source_run_id="replacement-run",
                    session_id="replacement-source",
                    source_incarnation="replacement-incarnation",
                    opening_prompt="replacement approval text",
                    workspace_id="replacement-destination",
                )
            probe.release.set()
            await task
            assert len(captured.requests) == 1
            request = captured.requests[0]
            assert request.draft_revision == revision
            assert request.context_epoch == epoch
            assert request.source_run_id == before["source_run_id"]
            assert request.source_session_id == before["session_id"]
            assert request.source_session_incarnation == before["source_incarnation"]
            assert request.opening_prompt == before["opening_prompt"]
            assert request.workspace_id == "global"
            if change != "approved_fields":
                # Exercise the installed coordinator's original target guard;
                # no admission/claim/attempt is requested in this leaf fixture.
                assert original._target_unchanged(request) is False
            assert not case.controller._fleet_wake._automatic_primary_claims
            _assert_retired(probe)
        finally:
            await snapshot_controls.catalog_controls._settle(task, probe)


@pytest.mark.asyncio
async def test_created_start_cancel_retains_actual_native_read_until_retirement(
    snapshot_case,
):
    case = snapshot_case
    _, captured, target, approved = _route(case)
    probe = _StartProbe(case, hold=True)
    task = None
    with probe.installed():
        task = asyncio.create_task(
            asyncio.to_thread(case.controller._start_created_chat, approved, target)
        )
        try:
            await snapshot_controls.catalog_controls._worker_entered(probe, task)
            assert len(probe.start_tasks) == 1
            inner = probe.start_tasks[0]
            inner.cancel()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            assert not inner.done() and not task.done()
            assert probe.leases and all(
                lease in storage_admission._live_leases for lease in probe.leases
            )
            probe.release.set()
            # The thread-safe Future first produces concurrent cancellation;
            # to_thread's asyncio Future translates it back for its awaiter.
            with pytest.raises(asyncio.CancelledError):
                await task
            assert inner.cancelled()
            assert not captured.requests
            assert not case.controller._fleet_wake._automatic_primary_claims
            _assert_retired(probe)
        finally:
            await snapshot_controls.catalog_controls._settle(task, probe)
