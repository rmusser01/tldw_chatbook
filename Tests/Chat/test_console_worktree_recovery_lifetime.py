"""Physical manual recovery ownership outlives cancellable UI waiters."""

import asyncio
import importlib.util
import threading
from types import SimpleNamespace

import pytest

from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity


@pytest.mark.asyncio
async def test_physical_manual_worker_retains_owner_after_waiter_cancel(
    monkeypatch, tmp_path
):
    assert (
        importlib.util.find_spec("tldw_chatbook.Chat.console_worktree_recovery")
        is not None
    )
    import tldw_chatbook.Chat.console_worktree_recovery as module
    from tldw_chatbook.Agents.agent_worktree_recovery import WorktreeRecoveryOutcome
    from tldw_chatbook.Chat.console_worktree_recovery import ConsoleWorktreeRecovery

    entered = threading.Event()
    release = threading.Event()
    capacity = RuntimeCapacity(max_child_executions=2, reserved_manual_children=1)
    controller = SimpleNamespace(
        capture_worktree_recovery_intent=lambda sid: SimpleNamespace(
            persisted_conversation_id="conv"
        ),
        request_worktree_merge_confirm=lambda *a, **kw: {"allow": True},
    )
    db = SimpleNamespace(db_path_str=str(tmp_path / "runs.db"))
    bridge = SimpleNamespace(runs_db=db, runtime_capacity=capacity)
    helper = ConsoleWorktreeRecovery(controller, bridge)
    monkeypatch.setattr(module, "validate_intent", lambda *a: object())

    def recover(*a, **kw):
        entered.set()
        release.wait(5)
        assert kw["should_cancel"]()
        return WorktreeRecoveryOutcome("apply", "Cancelled", "unresolved")

    monkeypatch.setattr(module, "recover_agent_worktree", recover)
    waiter = asyncio.create_task(helper.start("session-a", "run-a", "apply"))
    try:
        await asyncio.to_thread(entered.wait, 3)
        assert entered.is_set()
        waiter.cancel()
        with pytest.raises(asyncio.CancelledError):
            await waiter
        assert capacity.snapshot().executions
        helper.cancel_session("session-b")
        assert not helper.operations["session-a"].cancel.is_set()
        helper.cancel_session("session-a")
    finally:
        release.set()
        await helper.close()
        capacity.close()
    assert not capacity.snapshot().executions


def test_runtime_remounts_worktree_before_empty_unified_projection():
    from tldw_chatbook.Chat.console_runtime import (
        CONSOLE_VIEW_HOOK_SLOTS,
        ConsoleRuntime,
    )

    seen = []
    assert any(
        slot.name == "set_pending_worktree_merge" for slot in CONSOLE_VIEW_HOOK_SLOTS
    )
    runtime = ConsoleRuntime(app=None)
    runtime._chat_controller = SimpleNamespace(
        store=SimpleNamespace(active_session_id="session"),
        pending_decision_projection=lambda sid: None,
        _remount_parked_worktree_merge=seen.append,
    )
    runtime.remount_pending_approval()
    assert seen == ["session"]


def test_both_preview_entry_points_accept_real_surface_flag():
    import inspect

    from tldw_chatbook.Chat.console_agent_bridge import ConsoleAgentBridge

    for method in (
        ConsoleAgentBridge.build_project_instruction_preview_request,
        ConsoleAgentBridge.build_personal_context_preview_snapshot,
    ):
        assert (
            inspect.signature(method).parameters["worktree_merge_enabled"].default
            is False
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "gate", ["selected", "unselected", "readonly", "kill", "scratch"]
)
async def test_manual_authority_resolves_exact_selection_off_ui_thread(tmp_path, gate):
    from dataclasses import replace

    from Tests.Chat.test_console_agent_project_instructions import (
        _binding,
        _BindingRegistry,
    )
    from tldw_chatbook.Chat.console_project_instructions import (
        ProjectInstructionControlState,
        fingerprint_canonical_locator,
    )
    from tldw_chatbook.Chat.console_worktree_recovery import (
        RecoveryIntent,
        validate_intent,
    )

    state = ProjectInstructionControlState(
        project_instructions_enabled=True,
        working_folder_binding_id="b1" if gate != "unselected" else None,
        working_folder_locator_fingerprint=fingerprint_canonical_locator(str(tmp_path))
        if gate != "unselected"
        else None,
        project_instruction_notice_key=None,
    )
    intent = RecoveryIntent(
        "session", "w1" if gate != "scratch" else "global", "conversation", False, state
    )
    ui_thread = threading.get_ident()

    class Registry(_BindingRegistry):
        def get_runtime_binding(self, binding_id):
            assert threading.get_ident() != ui_thread
            return super().get_runtime_binding(binding_id)

        def list_runtime_bindings(self, workspace_id):
            assert threading.get_ident() != ui_thread
            return super().list_runtime_bindings(workspace_id)

    registry = Registry(
        [_binding(tmp_path, access="ro" if gate == "readonly" else "rw")]
    )
    controller = SimpleNamespace(
        store=SimpleNamespace(sessions=lambda: [intent]),
        app=SimpleNamespace(
            workspace_registry_service=registry,
            call_from_thread=lambda fn, *args: fn(*args),
        ),
        _console_tool_kill_switch_reader=lambda: lambda: gate == "kill",
    )
    authority = await asyncio.to_thread(validate_intent, controller, "session", intent)
    assert (authority is not None) is (gate == "selected")
    if authority:
        controller.store.sessions = lambda: [
            replace(
                intent,
                project_instruction_state=replace(
                    state, working_folder_binding_id=None
                ),
            )
        ]
        assert not await asyncio.to_thread(authority.guard, True)


@pytest.mark.asyncio
async def test_recovery_pages_are_exact_metadata_only_and_conversation_scoped(
    work, monkeypatch
):
    import inspect

    import tldw_chatbook.Chat.console_worktree_recovery as module
    from tldw_chatbook.Chat.console_worktree_recovery import ConsoleWorktreeRecovery

    db, authority, child, records = work
    source = records.get_for_conversation(child.run_id, "chat")
    fields = set(inspect.signature(records.record_created).parameters) - {"run_id"}
    template = {key: source[key] for key in fields}
    expected = [child.run_id]
    for i in range(52):
        run_id = f"page-{i:03d}"
        expected.append(run_id)
        db.create_run(conversation_id="chat", agent_kind="subagent", run_id=run_id)
        db.set_status(run_id, status="done", result="TRANSCRIPT_MUST_NOT_LOAD")
        records.record_created(run_id=run_id, **template)
    current_conversation = ["chat"]
    controller = SimpleNamespace(
        capture_worktree_recovery_intent=lambda sid: SimpleNamespace(
            persisted_conversation_id=current_conversation[0]
        ),
        request_worktree_merge_confirm=lambda *a, **kw: pytest.fail(
            "nonactionable record reached confirmation"
        ),
    )
    capacity = RuntimeCapacity()
    helper = ConsoleWorktreeRecovery(
        controller, SimpleNamespace(runs_db=db, runtime_capacity=capacity)
    )
    monkeypatch.setattr(module, "validate_intent", lambda *args: authority)
    try:
        first = await helper.list_work("session")
        second = await helper.list_work("session", first.next_run_id)
        assert [row["run_id"] for row in (*first.rows, *second.rows)] == sorted(
            expected
        )
        assert len(first.rows) == 50 and len(second.rows) == 3
        assert second.next_run_id is None
        assert "TRANSCRIPT_MUST_NOT_LOAD" not in repr(first)
        assert "steps" not in first.rows[0]
        # A held writer never reaches a confirmation surface.
        assert (await helper.start("session", "page-000", "apply")).reason_code
        current_conversation[0] = "foreign"
        assert not (await helper.list_work("session")).rows
        assert (await helper.start("session", child.run_id, "discard")).reason_code
        assert (
            records.get_for_conversation(child.run_id, "chat")["mutation_state"]
            == "unresolved"
        )
    finally:
        await helper.close()
        capacity.close()


from Tests.Agents.test_agent_worktree_confirmed_recovery import work as _work

work = _work



def test_worktree_remount_failure_does_not_abort_other_projection():
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime

    def broken(sid):
        raise RuntimeError("disposed view")

    shown = []
    runtime = ConsoleRuntime(app=None)
    runtime._chat_controller = SimpleNamespace(
        store=SimpleNamespace(active_session_id="a"),
        _remount_parked_worktree_merge=broken,
        pending_decision_projection=lambda sid: object(),
        set_pending_decision=lambda value: None,
        project_pending_decision_for_active_session=lambda: shown.append(True),
    )
    runtime.remount_pending_approval()
    assert shown == [True]


def test_begin_dispose_fences_manual_recovery_before_async_drain():
    from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
    from tldw_chatbook.Chat.console_worktree_recovery import ConsoleWorktreeRecovery

    helper = ConsoleWorktreeRecovery(None, None)
    runtime = ConsoleRuntime(app=None)
    runtime._worktree_recovery = helper
    runtime.begin_dispose()
    assert helper.closed


@pytest.mark.asyncio
async def test_accepted_close_fences_late_manual_start_during_actual_drain(
    tmp_path, monkeypatch
):
    import tldw_chatbook.Chat.console_worktree_recovery as module
    from Tests.Chat.test_console_runtime_shutdown import _runtime_with_fleet
    from tldw_chatbook.Agents.agent_worktree_recovery import WorktreeRecoveryOutcome
    from tldw_chatbook.DB.AgentRuns_DB import AgentRunsDB

    runtime, controller, bridge, store, session = _runtime_with_fleet()
    other = store.create_session(ephemeral=True)
    controller.app = SimpleNamespace(console_runtime=runtime)
    bridge.runs_db = AgentRunsDB(tmp_path / "runs.db")
    bridge.runtime_capacity = RuntimeCapacity()
    helper = runtime.worktree_recovery
    signal = threading.Event()
    previous = asyncio.create_task(asyncio.sleep(0))
    helper.operations[session.id] = module.RecoveryOperation(signal, previous)
    calls = []
    monkeypatch.setattr(module, "validate_intent", lambda *args: object())

    def recover(*args, **kwargs):
        calls.append(kwargs["run_id"])
        return WorktreeRecoveryOutcome("apply", "finished", "unresolved")

    monkeypatch.setattr(module, "recover_agent_worktree", recover)
    close = asyncio.create_task(
        runtime.close_session(
            session.id,
            expected_revision=controller.lifecycle_impact(
                session_id=session.id
            ).revision,
        )
    )
    try:
        await asyncio.wait_for(bridge.await_started.wait(), 3)
        assert signal.is_set()
        await previous
        helper.operations.pop(session.id)
        assert session.id in {item.id for item in store.sessions()}
        outcome = await helper.start(session.id, "late", "apply")
        assert outcome.reason_code == "missing_session"
        assert calls == []
        assert not controller.pending_worktree_merge_ids()
        assert controller.capture_worktree_recovery_intent(session.id) is None
        assert (await helper.start(other.id, "other", "apply")).message == "finished"
        assert calls == ["other"]
    finally:
        bridge.release.set()
        await close
        await runtime.dispose()
        bridge.runtime_capacity.close()
        bridge.runs_db.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("refusal", ["stale", "refused", "provisional"])
async def test_unaccepted_close_preserves_manual_cancel_signal(refusal, monkeypatch):
    from Tests.Chat.test_console_runtime_shutdown import _runtime_with_fleet
    from tldw_chatbook.Chat.console_worktree_recovery import (
        ConsoleWorktreeRecovery,
        RecoveryOperation,
    )

    runtime, controller, bridge, _store, session = _runtime_with_fleet()
    helper = ConsoleWorktreeRecovery(controller, bridge)
    runtime._worktree_recovery = helper
    signal = threading.Event()
    task = asyncio.create_task(asyncio.sleep(0))
    helper.operations[session.id] = RecoveryOperation(signal, task)
    revision = controller.lifecycle_impact(session_id=session.id).revision
    if refusal == "stale":
        controller._advance_lifecycle_revision(session.id)
    elif refusal == "refused":

        def refuse(*args, **kwargs):
            raise RuntimeError("refused")

        monkeypatch.setattr(controller, "begin_session_close", refuse)
    else:

        async def provisional(*args):
            return False

        runtime._voice_promotion_owner = SimpleNamespace(
            begin_session_close=lambda sid: object(),
            wait_for_session=provisional,
            abort_session_close=lambda token: None,
        )
    try:
        if refusal == "provisional":
            assert (
                await runtime.close_session(session.id, expected_revision=revision)
                is None
            )
        else:
            with pytest.raises(RuntimeError):
                await runtime.close_session(session.id, expected_revision=revision)
        assert not signal.is_set()
        runtime._raise_if_disposed_or_session_fenced(session.id)
    finally:
        await task
        helper.operations.clear()
        runtime._voice_promotion_owner = None
        bridge.release.set()
        await runtime.dispose()


@pytest.mark.asyncio
@pytest.mark.parametrize("submission", ["rejected", "queued", "entered"])
async def test_manual_submission_failure_hands_off_exact_owner(
    tmp_path, monkeypatch, submission
):
    """Submit can raise before enqueue, after enqueue, or after engine entry."""
    from concurrent.futures import ThreadPoolExecutor

    import tldw_chatbook.Chat.console_worktree_recovery as module
    from tldw_chatbook.Agents.agent_worktree_recovery import WorktreeRecoveryOutcome
    from tldw_chatbook.Agents.execution_capacity import current_execution_owner

    loop = asyncio.get_running_loop()
    executor = ThreadPoolExecutor(max_workers=1)
    queued_release = threading.Event()
    worker_release = threading.Event()
    entered = threading.Event()
    submitted = asyncio.Event()
    futures = []
    owners = []
    engine_owners = []
    capacity = RuntimeCapacity()
    real_begin = capacity.begin_execution

    def begin(**kwargs):
        owner = real_begin(**kwargs)
        owners.append(owner)
        return owner

    monkeypatch.setattr(capacity, "begin_execution", begin)
    helper = module.ConsoleWorktreeRecovery(
        SimpleNamespace(
            capture_worktree_recovery_intent=lambda sid: SimpleNamespace(
                persisted_conversation_id="conv"
            ),
            request_worktree_merge_confirm=lambda *a, **kw: {"allow": True},
        ),
        SimpleNamespace(
            runs_db=SimpleNamespace(db_path_str=str(tmp_path / "runs.db")),
            runtime_capacity=capacity,
        ),
    )
    monkeypatch.setattr(module, "validate_intent", lambda *a: object())

    def recover(*args, **kwargs):
        engine_owners.append(current_execution_owner())
        entered.set()
        assert worker_release.wait(5)
        return WorktreeRecoveryOutcome("apply", "finished", "unresolved")

    monkeypatch.setattr(module, "recover_agent_worktree", recover)
    if submission == "rejected":
        executor.shutdown(wait=True)
    elif submission == "queued":
        futures.append(executor.submit(queued_release.wait, 5))
    real_submit = executor.submit

    def submit(*args, **kwargs):
        try:
            future = real_submit(*args, **kwargs)
            futures.append(future)
            if submission == "entered":
                assert entered.wait(3), "worker did not enter before submission error"
            raise RuntimeError("submission raised after enqueue")
        finally:
            submitted.set()

    monkeypatch.setattr(executor, "submit", submit)
    real_run_in_executor = loop.run_in_executor

    def route_executor(default, func, *args):
        assert default is None
        return real_run_in_executor(executor, func, *args)

    monkeypatch.setattr(loop, "run_in_executor", route_executor)
    waiter = asyncio.create_task(helper.start("session", "run", "apply"))
    try:
        await asyncio.wait_for(submitted.wait(), 3)
        await asyncio.sleep(0)
        assert len(owners) == 1
        if submission == "entered":
            assert engine_owners == owners
            assert not waiter.done(), (
                "submit error abandoned an admitted physical worker"
            )
            assert "session" in helper.operations
            assert [item.execution_id for item in capacity.snapshot().executions] == [
                owners[0].execution_id
            ]
            waiter.cancel()
            with pytest.raises(asyncio.CancelledError):
                await waiter
            assert "session" in helper.operations
        else:
            with pytest.raises(RuntimeError):
                await waiter
            assert not entered.is_set()
            assert not capacity.snapshot().executions, "unstarted owner leaked"
            assert not helper.operations
        queued_release.set()
        worker_release.set()
        await helper.close()
        for future in futures:
            future.result(timeout=3)
        assert engine_owners == (owners if submission == "entered" else [])
        assert (tmp_path / "runs.db").exists() is (submission == "entered")
        assert not capacity.snapshot().executions
        assert not helper.operations
        assert helper.receipts["conv"].reason_code == "recovery_failed"
    finally:
        queued_release.set()
        worker_release.set()
        await helper.close()
        if not waiter.done():
            await asyncio.gather(waiter, return_exceptions=True)
        executor.shutdown(wait=True)
        capacity.close()
