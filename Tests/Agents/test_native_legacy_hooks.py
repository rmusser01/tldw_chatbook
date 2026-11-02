"""Actual legacy app hooks keep native custody through process and host return."""

import asyncio
import sys
import threading
import time
from types import SimpleNamespace

import pytest
import toml

from Tests.Agents.test_hook_permissions import _approve
from Tests.Agents.test_hook_permissions import hook_file as _hook_file
from tldw_chatbook.Agents.execution_capacity import WorkOrigin
from tldw_chatbook.Chat.console_chat_store import ConsoleChatStore
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

hook_file = _hook_file

pytestmark = pytest.mark.bootstrap_profile


def wait_until(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline, "hook did not reach its actual boundary"
        time.sleep(0.005)


@pytest.fixture
def native_hooks(hook_file, tmp_path):
    store = ConsoleChatStore()
    session = store.create_session(ephemeral=True)
    runtime = ConsoleRuntime(SimpleNamespace(app_config={}))
    runtime.set_chat_store(store)
    owner = runtime.ensure_hook_permissions()
    engine = runtime.ensure_run_hooks()

    def configure(*scripts, event="PreToolUse", timeout=3):
        raw = toml.loads(hook_file.read_text())
        raw["hooks"] = {
            "enabled": True,
            "hook": [
                {
                    "id": str(i),
                    "event": event,
                    "command": [sys.executable, "-c", script],
                    "timeout_s": timeout,
                }
                for i, script in enumerate(scripts)
            ],
        }
        hook_file.write_text(toml.dumps(raw))
        _approve(owner)

    host = SimpleNamespace(**locals())
    try:
        yield host
    finally:
        (tmp_path / "release").touch()
        engine.close()
        for pool in (engine._notify_worker, engine._notification_pool, engine._pool):
            pool.shutdown(wait=True, cancel_futures=True)
        asyncio.run(runtime.dispose())


def held_script(root, *, prefix=""):
    return (
        "import pathlib,time,os; "
        f"root=pathlib.Path({str(root)!r}); "
        "(root/'started').write_text(str(os.getpid())); "
        + prefix
        + "\nwhile not (root/'release').exists(): time.sleep(.01)\n"
    )


def opaque_busy(runtime):
    with runtime.worktree_activity.prelaunch(
        footprint=WorktreeFootprint(opaque=True)
    ) as claim:
        return claim is None


def test_actual_target_and_host_return_hold_one_native_event(native_hooks, monkeypatch):
    h = native_hooks
    h.configure(held_script(h.tmp_path))
    handed, finish = threading.Event(), threading.Event()
    reduce = h.engine._reduce

    def held_reduce(results):
        result = reduce(results)
        handed.set()
        assert finish.wait(5)
        return result

    monkeypatch.setattr(h.engine, "_reduce", held_reduce)
    outcomes = []
    thread = threading.Thread(
        target=lambda: outcomes.append(
            h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
        )
    )
    thread.start()
    try:
        wait_until((h.tmp_path / "started").exists)
        snap = h.runtime.execution_capacity.snapshot()
        assert snap.tool_workers == 1, "real hook process has no native operation"
        assert len(snap.executions) == 1
        assert not snap.child_executions and not snap.model_lifelines
        assert opaque_busy(h.runtime)
        (h.tmp_path / "release").touch()
        assert handed.wait(3)
        assert h.runtime.execution_capacity.snapshot().tool_workers == 1
        assert opaque_busy(h.runtime), "process return released unfinished host outcome"
    finally:
        (h.tmp_path / "release").touch()
        finish.set()
        thread.join(5)
    assert not outcomes[0].blocked
    assert not h.runtime.execution_capacity.snapshot().executions
    assert not opaque_busy(h.runtime)


@pytest.mark.parametrize("event", ["PreToolUse", "UserPromptSubmit", "Stop"])
def test_native_busy_keeps_existing_event_policy_and_never_enters(native_hooks, event):
    h = native_hooks
    h.configure(held_script(h.tmp_path), event=event)
    with h.runtime.worktree_activity.prelaunch(
        footprint=WorktreeFootprint(opaque=True)
    ) as claim:
        assert claim is not None
        if event == "Stop":
            h.engine.notify(event, session_id=h.session.id, cwd=str(h.tmp_path))
        else:
            outcome = h.engine.fire(event, session_id=h.session.id, cwd=str(h.tmp_path))
            assert outcome.blocked is (event == "PreToolUse")
        assert not h.runtime.execution_capacity.snapshot().executions
    assert not (h.tmp_path / "started").exists()


@pytest.mark.parametrize(
    "replacement", ["capacity", "activity", "permissions", "engine", "session"]
)
@pytest.mark.parametrize("warm", [False, True])
def test_original_source_replacement_refuses_before_target_entry(
    native_hooks, monkeypatch, replacement, warm
):
    from contextlib import contextmanager

    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity
    from tldw_chatbook.Agents.hook_permissions import HookPermissions
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeActivity

    h = native_hooks
    h.configure("pass")
    capacity, activity = h.runtime.execution_capacity, h.runtime.worktree_activity
    if warm:
        assert not h.engine.fire(
            "PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path)
        ).blocked
    h.configure(held_script(h.tmp_path))
    entered, release = threading.Event(), threading.Event()
    original = h.owner._current

    @contextmanager
    def held_read(*args, **kwargs):
        with original(*args, **kwargs) as value:
            entered.set()
            assert release.wait(5)
            yield value

    monkeypatch.setattr(h.owner, "_current", held_read)
    results = []
    worker = threading.Thread(
        target=lambda: results.append(
            h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
        )
    )
    worker.start()
    try:
        assert entered.wait(3)
        if replacement == "capacity":
            monkeypatch.setattr(h.runtime, "_execution_capacity", RuntimeCapacity())
        elif replacement == "activity":
            monkeypatch.setattr(h.runtime, "_worktree_activity", WorktreeActivity())
        elif replacement == "permissions":
            monkeypatch.setattr(h.runtime, "_hook_permissions", HookPermissions())
        elif replacement == "engine":
            monkeypatch.setattr(h.runtime, "_run_hooks_engine", None)
        else:
            monkeypatch.setitem(
                h.store._sessions, h.session.id, h.store.create_session()
            )
    finally:
        release.set()
        worker.join(5)
    assert results[0].blocked
    assert not (h.tmp_path / "started").exists()
    assert not capacity.snapshot().executions
    with activity.prelaunch(footprint=WorktreeFootprint(opaque=True)) as claim:
        assert claim is not None


def test_target_cancelled_in_existing_pool_unwinds_only_never_started_hold(
    native_hooks,
):
    h = native_hooks
    h.configure(held_script(h.tmp_path))
    release = threading.Event()
    blockers = [h.engine._pool.submit(release.wait, 5) for _ in range(4)]
    results = []
    worker = threading.Thread(
        target=lambda: results.append(
            h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
        )
    )
    worker.start()
    try:
        wait_until(lambda: h.runtime.execution_capacity.snapshot().tool_workers == 1)
        h.engine.close()
    finally:
        release.set()
        for blocker in blockers:
            blocker.result(3)
        worker.join(5)
    assert results[0].blocked
    assert not (h.tmp_path / "started").exists()
    assert not h.runtime.execution_capacity.snapshot().executions
    assert not opaque_busy(h.runtime)


@pytest.mark.asyncio
async def test_shutdown_joins_original_callback_even_after_repeated_cancellation(
    native_hooks, monkeypatch
):
    from tldw_chatbook.Agents import run_hooks

    h = native_hooks
    h.configure("pass", event="Stop")
    entered, release = threading.Event(), threading.Event()
    execute = run_hooks._execute_hook

    def held_return(*args, **kwargs):
        result = execute(*args, **kwargs)
        entered.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(run_hooks, "_execute_hook", held_return)
    h.engine.notify("Stop", session_id=h.session.id, cwd=str(h.tmp_path))
    assert await asyncio.to_thread(entered.wait, 3)
    task = asyncio.create_task(h.runtime.dispose(timeout_seconds=0))
    try:
        for _ in range(2):
            await asyncio.sleep(0.05)
            task.cancel()
            await asyncio.sleep(0.02)
            assert not task.done()
            assert h.runtime.execution_capacity.snapshot().tool_workers == 1
            assert opaque_busy(h.runtime)
        assert h.runtime.execution_capacity.snapshot().closed
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)
    assert not h.runtime.execution_capacity.snapshot().executions
    assert not opaque_busy(h.runtime)


def test_first_deny_does_not_retire_running_sibling(native_hooks):
    h = native_hooks
    h.configure("import sys; sys.exit(2)", held_script(h.tmp_path))
    result = h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
    assert result.blocked
    wait_until((h.tmp_path / "started").exists)
    assert opaque_busy(h.runtime), "first deny released its live sibling"
    assert h.runtime.execution_capacity.snapshot().tool_workers == 1
    (h.tmp_path / "release").touch()
    wait_until(lambda: not h.runtime.execution_capacity.snapshot().executions)


def test_parallel_finalization_joins_each_original_operation_return(
    native_hooks, monkeypatch
):
    from tldw_chatbook.Agents.execution_capacity import OwnedOperation

    h = native_hooks
    h.configure("import sys; sys.exit(2)", held_script(h.tmp_path))
    entered, release = threading.Event(), threading.Event()
    original = OwnedOperation.finish
    held = []

    def held_finish(operation):
        if not held:
            held.append(operation)
            entered.set()
            assert release.wait(5)
        return original(operation)

    monkeypatch.setattr(OwnedOperation, "finish", held_finish)
    outcomes = []
    worker = threading.Thread(
        target=lambda: outcomes.append(
            h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
        )
    )
    worker.start()
    try:
        assert entered.wait(3)
        wait_until((h.tmp_path / "started").exists)
        (h.tmp_path / "release").touch()
        wait_until(lambda: h.runtime.execution_capacity.snapshot().tool_workers == 1)
        assert opaque_busy(h.runtime), (
            "sibling settled before original release returned"
        )
        assert h.runtime._legacy_hooks_source.cleanup_pending
    finally:
        release.set()
        worker.join(5)
    assert outcomes[0].blocked
    wait_until(lambda: not h.runtime.execution_capacity.snapshot().executions)
    assert not opaque_busy(h.runtime)


def test_notification_reserves_actual_targets_before_coordinator_enqueue(native_hooks):
    h = native_hooks
    h.configure("pass", "pass", event="Stop")
    entered, release = threading.Event(), threading.Event()
    blocker = h.engine._notify_worker.submit(lambda: (entered.set(), release.wait(5)))
    assert entered.wait(2)
    try:
        h.engine.notify("Stop", session_id=h.session.id, cwd=str(h.tmp_path))
        snap = h.runtime.execution_capacity.snapshot()
        assert snap.tool_workers == 2, "queued event has no actual target holds"
        assert len(snap.executions) == 1
        assert snap.executions[0].origin is WorkOrigin.AUTOMATIC
        assert opaque_busy(h.runtime)
        h.engine.close()
    finally:
        release.set()
        blocker.result(3)
    wait_until(lambda: not h.runtime.execution_capacity.snapshot().executions)


@pytest.mark.asyncio
@pytest.mark.parametrize("deny", [False, True])
async def test_original_prompt_host_owns_outcome_until_transcript_return(
    native_hooks, monkeypatch, deny
):
    from Tests.Chat.test_console_chat_controller import RecordingStreamingGateway
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    h = native_hooks
    h.configure(
        "import sys; sys.exit(2)" if deny else "print('native hook context')",
        event="UserPromptSubmit",
    )
    gateway = RecordingStreamingGateway()
    controller = ConsoleChatController(
        store=h.store,
        provider_gateway=gateway,
        ensure_run_hooks=h.runtime.ensure_run_hooks,
        hook_permissions_accessor=h.runtime.ensure_hook_permissions,
    )
    h.runtime.set_chat_controller(controller)
    seen = []
    append = h.store.append_message

    def append_owned(*args, **kwargs):
        metadata = kwargs.get("metadata")
        if metadata is not None and metadata.origin == "hook":
            snap = h.runtime.execution_capacity.snapshot()
            seen.append(snap)
            assert snap.tool_workers == 1, (
                "fire_async released the original host handoff"
            )
            assert snap.executions[0].origin is WorkOrigin.MANUAL
            assert opaque_busy(h.runtime)
        return append(*args, **kwargs)

    monkeypatch.setattr(h.store, "append_message", append_owned)
    result = await controller.submit_draft("hello", session_id=h.session.id)
    assert result.accepted is not deny
    assert len(seen) == 1
    assert not h.runtime.execution_capacity.snapshot().executions
    assert not opaque_busy(h.runtime)


def test_pipe_closed_live_process_remains_owned(native_hooks):
    h = native_hooks
    h.configure(
        held_script(h.tmp_path, prefix="os.close(0); os.close(1); os.close(2);")
    )
    outcomes = []
    worker = threading.Thread(
        target=lambda: outcomes.append(
            h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
        )
    )
    worker.start()
    try:
        wait_until((h.tmp_path / "started").exists)
        time.sleep(0.05)
        assert worker.is_alive()
        assert opaque_busy(h.runtime)
        assert h.runtime.execution_capacity.snapshot().tool_workers == 1
    finally:
        (h.tmp_path / "release").touch()
        worker.join(5)
    assert not outcomes[0].blocked
    assert not opaque_busy(h.runtime)


def test_native_timeout_settles_only_after_original_process_and_group_exit(
    native_hooks,
):
    import os

    h = native_hooks
    h.configure(held_script(h.tmp_path), timeout=0.1)
    outcome = h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
    assert outcome.blocked
    pid = int((h.tmp_path / "started").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)
    assert not h.runtime.execution_capacity.snapshot().executions
    assert not opaque_busy(h.runtime)


def test_expired_group_evidence_retains_original_native_protection(
    native_hooks, monkeypatch
):
    from tldw_chatbook.Agents import run_hooks

    h = native_hooks
    h.configure("pass")
    signal_group = run_hooks.os.killpg

    def unavailable_group_evidence(pid, signal):
        if signal == 0:
            raise PermissionError("group identity cannot be observed")
        return signal_group(pid, signal)

    monkeypatch.setattr(run_hooks.os, "killpg", unavailable_group_evidence)
    monkeypatch.setattr(run_hooks, "HOOK_POST_KILL_REAP_TIMEOUT_S", 0.05)
    outcome = h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
    assert outcome.blocked, "unproven native cleanup allowed guarded dispatch"
    snapshot = h.runtime.execution_capacity.snapshot()
    assert snapshot.tool_workers == snapshot.stopping_tool_workers == 1
    assert opaque_busy(h.runtime)
    assert not h.engine._native_source.wait_closed()
    assert h.engine._native_source.cleanup_pending


def test_no_target_and_disabled_hooks_do_not_allocate_native_capacity(native_hooks):
    h = native_hooks
    h.configure("pass", event="Stop")
    assert h.runtime._execution_capacity is None
    assert not h.engine.fire(
        "PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path)
    ).blocked
    raw = toml.loads(h.hook_file.read_text())
    raw["hooks"]["enabled"] = False
    h.hook_file.write_text(toml.dumps(raw))
    h.owner.snapshot()
    h.engine.notify("Stop", session_id=h.session.id, cwd=str(h.tmp_path))
    assert h.runtime._execution_capacity is None
    assert h.runtime._worktree_activity is None


def test_notification_keeps_target_until_original_discard_callback_returns(
    native_hooks, monkeypatch
):
    h = native_hooks
    h.configure("pass", event="Stop")
    entered, release = threading.Event(), threading.Event()
    original = h.engine._notification_done

    def held_discard(future):
        original(future)
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(h.engine, "_notification_done", held_discard)
    h.engine.notify("Stop", session_id=h.session.id, cwd=str(h.tmp_path))
    try:
        assert entered.wait(3)
        assert h.runtime.execution_capacity.snapshot().tool_workers == 1
        assert opaque_busy(h.runtime)
    finally:
        release.set()
    wait_until(lambda: not h.runtime.execution_capacity.snapshot().executions)


def test_actual_grant_revocation_during_native_reserve_prevents_enqueue(
    native_hooks, monkeypatch
):
    from tldw_chatbook.Agents.execution_capacity import ExecutionOwner

    h = native_hooks
    h.configure(held_script(h.tmp_path))
    entered, release = threading.Event(), threading.Event()
    reserve = ExecutionOwner.reserve_tool

    def held_reserve(owner):
        assert h.engine._admission_lock.acquire(blocking=False)
        h.engine._admission_lock.release()
        operation = reserve(owner)
        entered.set()
        assert release.wait(5)
        return operation

    monkeypatch.setattr(ExecutionOwner, "reserve_tool", held_reserve)
    outcomes = []
    worker = threading.Thread(
        target=lambda: outcomes.append(
            h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
        )
    )
    worker.start()
    try:
        assert entered.wait(3)
        review = h.owner.snapshot()
        h.owner.revoke(review, review.rows[0].entry.key)
    finally:
        release.set()
        worker.join(5)
    assert outcomes[0].blocked
    assert not (h.tmp_path / "started").exists()
    assert not h.runtime.execution_capacity.snapshot().executions
    assert not opaque_busy(h.runtime)


def test_parallel_targets_use_separate_operations_on_same_event(native_hooks):
    h = native_hooks
    roots = [h.tmp_path / name for name in ("one", "two")]
    for root in roots:
        root.mkdir()
    h.configure(*(held_script(root) for root in roots))
    outcomes = []
    worker = threading.Thread(
        target=lambda: outcomes.append(
            h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
        )
    )
    worker.start()
    try:
        for root in roots:
            wait_until((root / "started").exists)
        snapshot = h.runtime.execution_capacity.snapshot()
        assert snapshot.tool_workers == 2 and len(snapshot.executions) == 1
        assert opaque_busy(h.runtime)
    finally:
        for root in roots:
            (root / "release").touch()
        worker.join(5)
    assert not outcomes[0].blocked
    assert not opaque_busy(h.runtime)


@pytest.mark.asyncio
async def test_cancelled_original_prompt_unwinds_queued_unstarted_target(native_hooks):
    from Tests.Chat.test_console_chat_controller import RecordingStreamingGateway
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    h = native_hooks
    h.configure(held_script(h.tmp_path), event="UserPromptSubmit")
    controller = ConsoleChatController(
        store=h.store,
        provider_gateway=RecordingStreamingGateway(),
        ensure_run_hooks=h.runtime.ensure_run_hooks,
        hook_permissions_accessor=h.runtime.ensure_hook_permissions,
    )
    h.runtime.set_chat_controller(controller)
    release = threading.Event()
    blockers = [h.engine._pool.submit(release.wait, 5) for _ in range(4)]
    task = asyncio.create_task(
        controller.submit_draft("hello", session_id=h.session.id)
    )
    try:
        for _ in range(200):
            if (
                h.runtime._execution_capacity
                and h.runtime.execution_capacity.snapshot().tool_workers
            ):
                break
            await asyncio.sleep(0.01)
        assert h.runtime.execution_capacity.snapshot().tool_workers == 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not h.runtime.execution_capacity.snapshot().executions
        assert not opaque_busy(h.runtime)
        assert not (h.tmp_path / "started").exists()
    finally:
        release.set()
        for blocker in blockers:
            blocker.result(3)
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.asyncio
async def test_manual_prompt_can_use_reserve_automatic_event_cannot(
    native_hooks, monkeypatch
):
    from Tests.Chat.test_console_chat_controller import RecordingStreamingGateway
    from tldw_chatbook.Agents.execution_capacity import RuntimeCapacity
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    h = native_hooks
    # Install the real finite capacity before the native source first captures it.
    h.runtime._execution_capacity = RuntimeCapacity(
        max_tool_workers=2, reserved_manual_tool_workers=1
    )
    h.configure("pass", "pass", event="Stop")
    h.engine.notify("Stop", session_id=h.session.id, cwd=str(h.tmp_path))
    assert not h.runtime.execution_capacity.snapshot().executions
    h.configure("print('context')", "print('context')", event="UserPromptSubmit")
    controller = ConsoleChatController(
        store=h.store,
        provider_gateway=RecordingStreamingGateway(),
        ensure_run_hooks=h.runtime.ensure_run_hooks,
        hook_permissions_accessor=h.runtime.ensure_hook_permissions,
    )
    h.runtime.set_chat_controller(controller)
    original = h.store.append_message
    holds = []

    def observed(*args, **kwargs):
        if getattr(kwargs.get("metadata"), "origin", None) == "hook":
            holds.append(h.runtime.execution_capacity.snapshot())
        return original(*args, **kwargs)

    monkeypatch.setattr(h.store, "append_message", observed)
    result = await controller.submit_draft("hello", session_id=h.session.id)
    assert result.accepted
    assert len(holds) == 1 and holds[0].tool_workers == 2
    assert len(holds[0].executions) == 1 and holds[0].child_executions == 0


def test_real_non_chat_event_keeps_empty_metadata_without_creating_chat(
    native_hooks, monkeypatch
):
    from tldw_chatbook.Agents import run_hooks
    from tldw_chatbook.Agents.execution_capacity import current_execution_owner

    h = native_hooks
    h.configure("pass")
    runtime = ConsoleRuntime(None)
    engine = runtime.ensure_run_hooks()
    seen = []
    original = run_hooks._execute_hook

    def observed(*args, **kwargs):
        seen.append(current_execution_owner())
        return original(*args, **kwargs)

    monkeypatch.setattr(run_hooks, "_execute_hook", observed)
    try:
        result = engine.fire("PreToolUse", session_id="", cwd=str(h.tmp_path))
        assert not result.blocked
        assert len(seen) == 1 and seen[0].conversation_id == ""
        assert seen[0].run_id is None and not seen[0].child
        assert runtime._chat_store is None and runtime._agent_bridge is None
    finally:
        asyncio.run(runtime.dispose())
