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
from tldw_chatbook.Chat.console_workflow_runs import native_workflow_qualified
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

hook_file = _hook_file

pytestmark = [
    pytest.mark.bootstrap_profile,
    pytest.mark.skipif(
        not native_workflow_qualified(), reason="native POSIX group custody required"
    ),
]


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


@pytest.mark.parametrize("held_ordinal", [0, 1])
def test_parallel_finalization_joins_each_original_operation_return(
    native_hooks, monkeypatch, held_ordinal
):
    from tldw_chatbook.Agents.execution_capacity import OwnedOperation

    h = native_hooks
    h.configure("import sys; sys.exit(2)", held_script(h.tmp_path))
    entered, release = threading.Event(), threading.Event()
    original = OwnedOperation.finish
    entries, returned = [], []

    def held_finish(operation):
        entries.append(operation)
        if len(entries) - 1 == held_ordinal:
            entered.set()
            assert release.wait(5)
        result = original(operation)
        returned.append(operation)
        return result

    monkeypatch.setattr(OwnedOperation, "finish", held_finish)
    outcomes = []
    worker = threading.Thread(
        target=lambda: outcomes.append(
            h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
        )
    )
    worker.start()
    try:
        wait_until((h.tmp_path / "started").exists)
        (h.tmp_path / "release").touch()
        assert entered.wait(3)
        assert len(returned) == held_ordinal
        assert h.runtime.execution_capacity.snapshot().tool_workers == 2 - held_ordinal
        assert opaque_busy(h.runtime), (
            "sibling settled before original release returned"
        )
        assert h.runtime._legacy_hooks_source.cleanup_pending
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive()
    wait_until(lambda: len(returned) == 2)
    assert returned == entries
    assert outcomes[0].blocked
    wait_until(lambda: not h.runtime.execution_capacity.snapshot().executions)
    wait_until(lambda: not h.runtime._legacy_hooks_source.cleanup_pending)
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


def test_native_grant_keeps_direct_topology_and_literal_payload(
    native_hooks, monkeypatch
):
    import json
    import os

    h = native_hooks
    report = h.tmp_path / "identity.json"
    special = "literal $HOME; `echo untouched` ' spaced"
    monkeypatch.setenv("TASK12_LITERAL", special)
    script = (
        "import json,os,sys; data=json.load(sys.stdin); "
        "json.dump({'pid':os.getpid(),'pgid':os.getpgrp(),'sid':os.getsid(0),"
        "'ppid':os.getppid(),'argv':sys.argv[2], 'env':os.environ['TASK12_LITERAL'],"
        "'cwd':os.getcwd(),'payload':data},open(sys.argv[1],'w')); print('native context')"
    )
    h.configure(script, event="UserPromptSubmit")
    raw = toml.loads(h.hook_file.read_text())
    raw["hooks"]["hook"][0]["command"] += [str(report), special]
    h.hook_file.write_text(toml.dumps(raw))
    _approve(h.owner)
    outcome = h.engine.fire(
        "UserPromptSubmit",
        session_id=h.session.id,
        cwd=str(h.tmp_path),
        data={"prompt": special},
    )
    assert outcome.context == "native context" and not outcome.blocked
    found = json.loads(report.read_text())
    assert found["pid"] == found["pgid"] == found["sid"]
    assert found["ppid"] == os.getpid()
    assert found["argv"] == found["env"] == special
    assert found["cwd"] == str(h.tmp_path)
    assert found["payload"]["data"]["prompt"] == special
    assert not h.runtime.execution_capacity.snapshot().executions
    assert not opaque_busy(h.runtime)


@pytest.mark.parametrize(
    ("script", "blocked", "reason"),
    [
        ("pass", False, ""),
        (
            "import sys; sys.stderr.write('explicit refusal'); sys.exit(2)",
            True,
            "explicit refusal",
        ),
        ("import os,signal; os.kill(os.getpid(),signal.SIGTERM)", True, "exit -15"),
    ],
)
def test_native_actual_signed_exits(native_hooks, script, blocked, reason):
    h = native_hooks
    h.configure(script)
    result = h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
    assert result.blocked is blocked
    assert reason in result.reason
    assert not h.runtime.execution_capacity.snapshot().executions


def test_native_closed_stdin_preserves_large_input_context_and_denial(native_hooks):
    h = native_hooks
    payload = {"prompt": "p" * 1000000}
    h.configure(
        "import os; os.close(0); print('short context')", event="UserPromptSubmit"
    )
    result = h.engine.fire(
        "UserPromptSubmit", session_id=h.session.id, cwd=str(h.tmp_path), data=payload
    )
    assert not result.blocked and result.context == "short context"
    h.configure(
        'import os; os.close(0); print(\'{"decision":"deny","reason":"closed stdin"}\')'
    )
    result = h.engine.fire(
        "PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path), data=payload
    )
    assert result.blocked and result.reason == "closed stdin"
    assert not h.runtime.execution_capacity.snapshot().executions


def test_native_fairly_drains_large_input_and_both_output_pipes(native_hooks):
    h = native_hooks
    script = (
        "import os,sys; "
        "os.write(1,b'x'*131072); os.write(2,b'y'*131072); "
        "sys.stdin.buffer.read()"
    )
    h.configure(script, timeout=3)
    result = h.engine.fire(
        "PreToolUse",
        session_id=h.session.id,
        cwd=str(h.tmp_path),
        data={"payload": "z" * 1000000},
    )
    assert result.blocked and "output exceeded budget" in result.reason
    assert not h.runtime.execution_capacity.snapshot().executions
    assert not opaque_busy(h.runtime)


@pytest.mark.parametrize("retain_streams", [False, True])
def test_native_original_group_descendant_is_drained(native_hooks, retain_streams):
    import os

    h = native_hooks
    child_started = h.tmp_path / "child_started"
    leader_pid = h.tmp_path / "leader_pid"
    child = (
        "import os,pathlib,time; "
        f"pathlib.Path({str(child_started)!r}).write_text(str(os.getpid())); "
        "time.sleep(4)"
    )
    streams = (
        "{}"
        if retain_streams
        else "{'stdout':subprocess.DEVNULL,'stderr':subprocess.DEVNULL}"
    )
    script = (
        "import os,pathlib,subprocess,sys,time; "
        f"pathlib.Path({str(leader_pid)!r}).write_text(str(os.getpid())); "
        f"subprocess.Popen([sys.executable,'-c',{child!r}],**{streams}); "
        f"p=pathlib.Path({str(child_started)!r}); "
        "deadline=time.monotonic()+2; "
        "\nwhile not p.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
        "assert p.exists(); print('partial context',flush=True)"
    )
    h.configure(script, event="UserPromptSubmit", timeout=1.5)
    result = h.engine.fire(
        "UserPromptSubmit", session_id=h.session.id, cwd=str(h.tmp_path)
    )
    assert child_started.exists()
    assert not result.blocked
    assert result.context == ("" if retain_streams else "partial context")
    with pytest.raises(ProcessLookupError):
        os.killpg(int(leader_pid.read_text()), 0)
    assert not h.runtime.execution_capacity.snapshot().executions
    assert not opaque_busy(h.runtime)


def test_native_escaped_writer_timeout_discards_partial_context(native_hooks):
    import os

    h = native_hooks
    started, finished = h.tmp_path / "escaped", h.tmp_path / "escaped_done"
    script = (
        "import os,pathlib,time; print('must be discarded',flush=True); "
        "pid=os.fork()\n"
        "if pid == 0:\n"
        " os.setsid()\n"
        f" pathlib.Path({str(started)!r}).write_text(str(os.getpid()))\n"
        " time.sleep(2)\n"
        f" pathlib.Path({str(finished)!r}).write_text('done')\n"
        " os._exit(0)\n"
        f"p=pathlib.Path({str(started)!r}); "
        "deadline=time.monotonic()+2; "
        "\nwhile not p.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
        "assert p.exists()"
    )
    h.configure(script, event="UserPromptSubmit", timeout=0.5)
    try:
        result = h.engine.fire(
            "UserPromptSubmit", session_id=h.session.id, cwd=str(h.tmp_path)
        )
        assert started.exists() and not finished.exists()
        os.kill(int(started.read_text()), 0)
        assert not result.blocked and result.context == ""
        assert not h.runtime.execution_capacity.snapshot().executions
        assert not opaque_busy(h.runtime)
    finally:
        wait_until(finished.exists, seconds=4)


def test_parallel_native_targets_do_not_cross_reap_unrelated_children(native_hooks):
    import asyncio
    import subprocess

    h = native_hooks
    script = (
        "import os,pathlib,time; "
        f"root=pathlib.Path({str(h.tmp_path)!r}); "
        "(root/str(os.getpid())).write_text('ready'); "
        "\nwhile not (root/'release').exists(): time.sleep(.01)\n"
    )
    h.configure(script, script)
    outcomes = []
    thread = threading.Thread(
        target=lambda: outcomes.append(
            h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
        )
    )
    thread.start()
    try:
        wait_until(
            lambda: len([p for p in h.tmp_path.iterdir() if p.name.isdecimal()]) == 2
        )
        unrelated = subprocess.Popen([sys.executable, "-c", "raise SystemExit(23)"])
        assert unrelated.wait(timeout=3) == 23

        async def run_async_child():
            process = await asyncio.create_subprocess_exec(
                sys.executable, "-c", "raise SystemExit(17)"
            )
            return await process.wait()

        assert asyncio.run(run_async_child()) == 17
        assert h.runtime.execution_capacity.snapshot().tool_workers == 2
    finally:
        (h.tmp_path / "release").touch()
        thread.join(5)
    assert not thread.is_alive() and len(outcomes) == 1 and not outcomes[0].blocked
    assert not h.runtime.execution_capacity.snapshot().executions
    assert not opaque_busy(h.runtime)


def test_revoked_grant_does_not_abandon_already_launched_native_target(native_hooks):
    h = native_hooks
    h.configure(held_script(h.tmp_path))
    outcomes = []
    worker = threading.Thread(
        target=lambda: outcomes.append(
            h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
        )
    )
    worker.start()
    try:
        wait_until((h.tmp_path / "started").exists)
        review = h.owner.snapshot()
        h.owner.revoke(review, review.rows[0].entry.key)
    finally:
        (h.tmp_path / "release").touch()
        worker.join(5)
    assert not worker.is_alive() and len(outcomes) == 1 and not outcomes[0].blocked
    assert not h.runtime.execution_capacity.snapshot().executions
    assert not opaque_busy(h.runtime)
    (h.tmp_path / "started").unlink()
    assert h.engine.fire(
        "PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path)
    ).blocked
    assert not (h.tmp_path / "started").exists()


def test_engine_close_drains_its_already_launched_native_child(native_hooks):
    import os

    h = native_hooks
    h.configure(held_script(h.tmp_path))
    outcomes = []
    worker = threading.Thread(
        target=lambda: outcomes.append(
            h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
        )
    )
    worker.start()
    try:
        wait_until((h.tmp_path / "started").exists)
        child_pid = int((h.tmp_path / "started").read_text())
        h.engine.close()
        worker.join(5)
        assert not worker.is_alive() and outcomes[0].blocked
        with pytest.raises(ProcessLookupError):
            os.killpg(child_pid, 0)
        assert not h.runtime.execution_capacity.snapshot().executions
        assert not opaque_busy(h.runtime)
    finally:
        (h.tmp_path / "release").touch()
        worker.join(5)


def test_fenced_chat_refuses_new_native_capture_without_leaking_counts(native_hooks):
    h = native_hooks
    h.configure(held_script(h.tmp_path))
    h.runtime._admission_fenced_sessions.add(h.session.id)
    try:
        outcome = h.engine.fire(
            "PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path)
        )
        assert outcome.blocked and not (h.tmp_path / "started").exists()
        source = h.engine._native_source
        assert source._session_admissions == {} and source._admissions == 0
        assert not source._events and not opaque_busy(h.runtime)
    finally:
        h.runtime._admission_fenced_sessions.remove(h.session.id)


def test_close_fence_at_publication_unwinds_bound_native_reservation(
    native_hooks, monkeypatch
):
    from tldw_chatbook.Workspaces.worktree_activity import WorktreeActivityClaim

    h = native_hooks
    h.configure(held_script(h.tmp_path))
    entered, release = threading.Event(), threading.Event()
    original = WorktreeActivityClaim.bind_owner

    def hold_after_bind(claim, owner):
        original(claim, owner)
        entered.set()
        assert release.wait(5)

    monkeypatch.setattr(WorktreeActivityClaim, "bind_owner", hold_after_bind)
    outcomes = []
    worker = threading.Thread(
        target=lambda: outcomes.append(
            h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
        )
    )
    worker.start()
    try:
        assert entered.wait(3)
        h.runtime._admission_fenced_sessions.add(h.session.id)
        release.set()
        worker.join(5)
        assert not worker.is_alive() and outcomes[0].blocked
        assert not (h.tmp_path / "started").exists()
        source = h.engine._native_source
        assert source._session_admissions == {} and source._admissions == 0
        assert (
            not source._events
            and not h.runtime.execution_capacity.snapshot().executions
        )
        assert not opaque_busy(h.runtime)
    finally:
        release.set()
        worker.join(5)
        h.runtime._admission_fenced_sessions.discard(h.session.id)


def test_selected_close_waits_for_original_host_return(native_hooks, monkeypatch):
    from Tests.Chat.test_console_chat_controller import RecordingStreamingGateway
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    h = native_hooks
    controller = ConsoleChatController(
        store=h.store,
        provider_gateway=RecordingStreamingGateway(),
        ensure_run_hooks=h.runtime.ensure_run_hooks,
        hook_permissions_accessor=h.runtime.ensure_hook_permissions,
    )
    h.runtime.set_chat_controller(controller)
    other = h.store.create_session(title="other")
    h.configure("pass")
    entered, release = threading.Event(), threading.Event()
    original_reduce = h.engine._reduce

    def held_reduce(results):
        result = original_reduce(results)
        entered.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(h.engine, "_reduce", held_reduce)
    outcomes = []
    worker = threading.Thread(
        target=lambda: outcomes.append(
            h.engine.fire("PreToolUse", session_id=h.session.id, cwd=str(h.tmp_path))
        )
    )
    worker.start()
    try:
        assert entered.wait(4)
        revision = controller.lifecycle_impact(session_id=h.session.id).revision

        async def close():
            from concurrent.futures import ThreadPoolExecutor

            pool_held, pool_release = threading.Event(), threading.Event()
            loop = asyncio.get_running_loop()
            loop.set_default_executor(ThreadPoolExecutor(max_workers=1))

            def occupy_pool():
                pool_held.set()
                pool_release.wait(5)

            loop.run_in_executor(None, occupy_pool)
            assert pool_held.wait(2)
            task = asyncio.create_task(
                h.runtime.close_session(h.session.id, expected_revision=revision)
            )
            try:
                await asyncio.sleep(0.15)
                assert not task.done()
                assert h.session.id in h.runtime._admission_fenced_sessions
                assert h.store._sessions[h.session.id] is h.session
                assert other.id not in h.runtime._admission_fenced_sessions
                assert (
                    h.runtime._legacy_hooks_source._session_admissions.get(other.id, 0)
                    == 0
                )
                release.set()
                result = await asyncio.wait_for(task, 4)
                assert not pool_release.is_set()
                return result
            finally:
                release.set()
                pool_release.set()

        asyncio.run(close())
    finally:
        release.set()
        worker.join(5)
    assert not worker.is_alive() and len(outcomes) == 1
    assert h.session.id not in h.store._sessions
    assert h.store._sessions[other.id] is other
    assert not h.runtime._legacy_hooks_source._session_admissions


def test_sealed_close_absence_allows_actual_other_chat_hook(hook_file, tmp_path):
    from Tests.Chat.test_console_chat_controller import RecordingStreamingGateway
    from tldw_chatbook.Chat.console_chat_controller import ConsoleChatController

    store = ConsoleChatStore()
    closing = store.create_session(ephemeral=True)
    other = store.create_session(ephemeral=True)
    runtime = ConsoleRuntime(SimpleNamespace(app_config={}))
    runtime.set_chat_store(store)
    controller = ConsoleChatController(
        store=store,
        provider_gateway=RecordingStreamingGateway(),
        ensure_run_hooks=runtime.ensure_run_hooks,
        hook_permissions_accessor=runtime.ensure_hook_permissions,
    )
    runtime.set_chat_controller(controller)
    runtime._admission_fenced_sessions.add(closing.id)
    ticket = controller.begin_session_close(
        closing.id,
        expected_revision=controller.lifecycle_impact(session_id=closing.id).revision,
    )
    state = controller._session_close_states[ticket.close_id]
    controller._legacy_close_ready(ticket, state, state[4])
    assert state[4].mode == "absent"
    marker = tmp_path / "other-chat-hook"
    raw = toml.loads(hook_file.read_text())
    raw["hooks"] = {
        "enabled": True,
        "hook": [
            {
                "id": "other",
                "event": "PreToolUse",
                "command": [
                    sys.executable,
                    "-c",
                    f"import pathlib; pathlib.Path({str(marker)!r}).touch()",
                ],
                "timeout_s": 3,
            }
        ],
    }
    hook_file.write_text(toml.dumps(raw))
    owner = runtime.ensure_hook_permissions()
    _approve(owner)
    engine = runtime.ensure_run_hooks()
    try:
        result = engine.fire("PreToolUse", session_id=other.id, cwd=str(tmp_path))
        assert not result.blocked and marker.exists()
        assert state[4].mode == "absent" and state[4].issuer is None
        controller.finalize_session_close(ticket)
        assert closing.id not in store._sessions and store._sessions[other.id] is other
    finally:
        engine.close()
        for pool in (engine._notify_worker, engine._notification_pool, engine._pool):
            pool.shutdown(wait=True, cancel_futures=True)
        asyncio.run(runtime.dispose())
