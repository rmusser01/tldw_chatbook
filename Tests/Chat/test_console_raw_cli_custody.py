"""Direct user commands retain their real native process and local-command outcome."""

import asyncio
import threading
from types import SimpleNamespace

import pytest

from Tests.console_provider_doubles import persisted_console_store
from tldw_chatbook.Agents.execution_capacity import WorkOrigin
from tldw_chatbook.Chat.console_raw_cli import RawCliRuntime
from tldw_chatbook.Chat.console_runtime import ConsoleRuntime
from tldw_chatbook.UI.Console_Modules.raw_cli import ConsoleRawCliController
from tldw_chatbook.Widgets.Console.console_composer_bar import ConsoleDraftStash
from tldw_chatbook.Workspaces.worktree_activity import WorktreeFootprint

pytestmark = [pytest.mark.bootstrap_profile, pytest.mark.asyncio]


async def until(probe):
    async with asyncio.timeout(15):
        while not probe():
            await asyncio.sleep(0.005)


@pytest.fixture
async def raw_host(tmp_path):
    store = persisted_console_store(db_path=tmp_path / "chat.db")
    session = store.create_session()
    raw = RawCliRuntime(read_permitted=lambda: True)
    assert raw.arm().armed
    app = SimpleNamespace(
        app_config={},
        chachanotes_db=store.persistence.db,
        raw_cli_runtime=raw,
        workspace_registry_service=None,
    )
    runtime = app.console_runtime = ConsoleRuntime(app)
    runtime.set_chat_store(store)
    runtime.ensure_activity_receipt_service()
    workers, restored, errors, roots = [], [], [], []

    def scratch(session_id):
        roots.append(session_id)
        return runtime.scratch_spaces.snapshot(session_id).root

    controller = ConsoleRawCliController(
        raw_cli_runtime=lambda: raw,
        native_source=runtime.raw_cli_source,
        active_session_id=lambda: session.id,
        persist_session_if_needed=store.persist_session_if_needed,
        active_leaf_anchor=store.active_leaf,
        persisted_leaf_anchor=store.persisted_message_id_for_session_node,
        selected_local_root=lambda _sid: None,
        private_scratch_root=scratch,
        refusal_stash_bank=runtime.raw_cli_refusal_stash_bank,
        accepts_raw_cli_refusal_callbacks=lambda: (
            runtime.accepts_raw_cli_refusal_callbacks
        ),
        restore_stash=lambda sid, stash: restored.append((sid, stash)) or True,
        append_local_error=lambda sid, text: errors.append((sid, text)),
        append_store_marker=store.append_message,
        update_store_marker=store.update_tool_marker,
        agent_runs_db=lambda: runtime._agent_runs_db,
        run_log_access=lambda: tmp_path / "private-logs",
        start_worker=lambda work, **kwargs: workers.append((work, kwargs)),
        marshal_to_ui=lambda callback, *args: callback(*args),
    )
    host = SimpleNamespace(**locals())
    try:
        yield host
    finally:
        raw.shutdown()
        await runtime.dispose()
        store.persistence.db.close_connection()


def stash(command="printf native-raw-output"):
    return ConsoleDraftStash(
        segments=[],
        text="! " + command,
        has_paste=False,
        raw_cli_prefix_typed=True,
    )


async def test_busy_refuses_before_scratch_and_preserves_exact_stash(raw_host):
    h = raw_host
    draft = stash()
    with h.runtime.worktree_activity.prelaunch(
        footprint=WorktreeFootprint(opaque=True)
    ) as claim:
        owner = h.runtime.execution_capacity.begin_execution(
            origin=WorkOrigin.MANUAL, conversation_id=""
        )
        claim.bind_owner(owner)
        try:
            assert not h.controller.start_user_command(draft), (
                "busy raw command was enqueued"
            )
            assert h.workers == []
            assert h.roots == []
            assert h.restored == [(h.session.id, draft)]
            assert h.session.persisted_conversation_id is None
        finally:
            owner.finish_root()
            claim.settle(durable=True)


async def test_real_local_command_owns_enqueue_and_actual_sqlite_row(raw_host):
    h = raw_host
    assert h.controller.start_user_command(stash())
    snapshot = h.runtime.execution_capacity.snapshot()
    try:
        assert snapshot.tool_workers == 1, "raw worker enqueue has no native tool owner"
        assert snapshot.model_lifelines == snapshot.child_executions == 0
        assert len(snapshot.executions) == 1
        assert snapshot.executions[0].run_id is None
    finally:
        work, flags = h.workers.pop()
        assert flags["exclusive"] is False
        await asyncio.to_thread(work)
    runs = h.runtime._agent_runs_db.list_runs(h.session.persisted_conversation_id)
    assert len(runs) == 1
    assert runs[0]["agent_kind"] == "local_command"
    assert runs[0]["status"] == "done"
    assert runs[0]["steps"][1]["args"]["cleanup_proven"] is True
    assert h.runtime.execution_capacity.snapshot().executions == ()
    assert h.runtime._agent_bridge is None


async def test_start_refusal_releases_only_after_original_stash_acceptance(raw_host):
    h = raw_host
    observed = []
    original_restore = h.controller._restore_stash
    h.controller._restore_stash = lambda sid, draft: (
        observed.append(h.runtime.execution_capacity.snapshot().tool_workers)
        or original_restore(sid, draft)
    )
    h.controller._start_worker = lambda *args, **kwargs: (_ for _ in ()).throw(
        RuntimeError("start refused")
    )
    draft = stash()
    assert h.controller.start_user_command(draft) is False
    assert observed == [1]
    assert h.restored == [(h.session.id, draft)]
    assert h.runtime.execution_capacity.snapshot().executions == ()


async def test_original_admission_helper_return_holds_native_operation(raw_host):
    h = raw_host
    entered, release = threading.Event(), threading.Event()
    original = h.controller._update_running_marker

    def held(*args):
        original(*args)
        entered.set()
        assert release.wait(15)

    h.controller._update_running_marker = held
    assert h.controller.start_user_command(stash())
    work, _ = h.workers.pop()
    task = asyncio.create_task(asyncio.to_thread(work))
    try:
        await until(entered.is_set)
        await asyncio.sleep(0.5)
        assert not task.done()
        snapshot = h.runtime.execution_capacity.snapshot()
        assert snapshot.tool_workers == 1
        assert snapshot.executions[0].run_id is not None
        assert (
            h.runtime._agent_runs_db.get_run(snapshot.executions[0].run_id)["status"]
            == "running"
        )
    finally:
        release.set()
        await task
    assert h.runtime.execution_capacity.snapshot().executions == ()


async def test_committed_then_raised_terminal_is_proven_from_original_sqlite(raw_host):
    h = raw_host
    db = h.runtime._agent_runs_db
    original = db.set_terminal_with_step
    attempts = []

    def committed(*args, **kwargs):
        assert original(*args, **kwargs) is True
        attempts.append(args[0])
        raise RuntimeError("after actual commit")

    db.set_terminal_with_step = committed
    assert h.controller.start_user_command(stash())
    work, _ = h.workers.pop()
    await asyncio.to_thread(work)
    assert len(attempts) == 1
    assert db.get_run(attempts[0])["status"] == "done"
    assert h.runtime.execution_capacity.snapshot().executions == ()


async def test_replaced_original_runtime_refuses_without_row_and_banks_origin(raw_host):
    h = raw_host
    draft = stash()
    assert h.controller.start_user_command(draft)
    h.app.console_runtime = object()
    work, _ = h.workers.pop()
    try:
        await asyncio.to_thread(work)
        assert h.session.persisted_conversation_id is None
        assert h.restored == [(h.session.id, draft)]
        assert h.runtime.execution_capacity.snapshot().executions == ()
    finally:
        h.app.console_runtime = h.runtime


async def test_temporary_chat_refusal_never_manufactures_run(raw_host):
    h = raw_host
    h.session.ephemeral = True
    assert h.controller.start_user_command(stash())
    work, _ = h.workers.pop()
    await asyncio.to_thread(work)
    assert h.session.persisted_conversation_id is None
    assert h.runtime.execution_capacity.snapshot().executions == ()
    assert h.restored


async def test_cold_storage_uses_original_receipt_db_without_bridge(raw_host):
    h = raw_host
    h.runtime._agent_runs_db.close()
    h.runtime._activity_receipts = None
    h.runtime._agent_runs_db = None
    assert h.controller.start_user_command(stash())
    work, _ = h.workers.pop()
    await asyncio.to_thread(work)
    runs = h.runtime._agent_runs_db.list_runs(h.session.persisted_conversation_id)
    assert len(runs) == 1 and runs[0]["status"] == "done"
    assert h.runtime._agent_bridge is None


async def test_disposal_joins_actual_host_return_and_scratch_lease(raw_host):
    h = raw_host
    entered, release = threading.Event(), threading.Event()
    original = h.controller._finish_marker

    def held(*args):
        original(*args)
        entered.set()
        assert release.wait(15)

    h.controller._finish_marker = held
    assert h.controller.start_user_command(stash())
    scratch = h.runtime.scratch_spaces.snapshot(h.session.id)
    work, _ = h.workers.pop()
    task = asyncio.create_task(asyncio.to_thread(work))
    dispose = None
    try:
        await until(entered.is_set)
        dispose = asyncio.create_task(h.runtime.dispose(timeout_seconds=0.01))
        await until(lambda: h.runtime._disposed)
        dispose.cancel()
        dispose.cancel()
        await asyncio.sleep(0.05)
        assert not dispose.done()
        assert h.runtime.execution_capacity.snapshot().tool_workers == 1
        assert scratch.root.is_dir()
        with h.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            assert claim is None
    finally:
        release.set()
        await task
        if dispose is not None:
            try:
                await dispose
            except asyncio.CancelledError:
                pass
    assert h.runtime.execution_capacity.snapshot().executions == ()
    assert not scratch.root.exists()


async def test_original_operation_finish_return_precedes_claim_retirement(raw_host):
    h = raw_host
    assert h.controller.start_user_command(stash())
    call = next(iter(h.runtime.raw_cli_source.pending))
    entered, release = threading.Event(), threading.Event()
    original = call.operation.finish

    def held():
        original()
        entered.set()
        assert release.wait(15)

    call.operation.finish = held
    work, _ = h.workers.pop()
    task = asyncio.create_task(asyncio.to_thread(work))
    try:
        await until(entered.is_set)
        assert not call.retired.is_set()
        assert call.owner.is_admitted
        with h.runtime.worktree_activity.prelaunch(
            footprint=WorktreeFootprint(opaque=True)
        ) as claim:
            assert claim is None
    finally:
        release.set()
        await task
    assert call.retired.is_set()


async def test_dispose_unstarted_callback_cannot_later_launch(raw_host):
    h = raw_host
    assert h.controller.start_user_command(stash())
    work, _ = h.workers.pop()
    await h.runtime.dispose()
    await asyncio.to_thread(work)
    assert h.session.persisted_conversation_id is None
    assert h.runtime.execution_capacity.snapshot().executions == ()


async def test_model_call_keeps_its_original_single_operation_through_relay_return(
    raw_host, monkeypatch
):
    from Tests.Agents.test_raw_shell_integration import (
        _fence as fence,
    )
    from Tests.Agents.test_raw_shell_integration import (
        _ScriptedChat as ScriptedChat,
    )
    from tldw_chatbook.Agents.agent_models import AgentConfig
    from tldw_chatbook.Agents.agent_service import AgentService
    from tldw_chatbook.Agents.raw_shell_tool_provider import RawShellToolProvider
    from tldw_chatbook.Agents.tool_catalog import ToolCatalogRegistry
    from tldw_chatbook.MCP.permission_store import EffectiveToolState
    from tldw_chatbook.Tools import raw_cli_executor

    h = raw_host
    h.raw.grant_model_session(h.session.id)
    provider = RawShellToolProvider(
        runtime=h.raw,
        console_session_id=h.session.id,
        initial_directory=lambda: h.tmp_path,
        resolve_state=lambda hub: EffectiveToolState(
            state="ask", origin="global_default"
        ),
    )
    registry = ToolCatalogRegistry()
    registry.register_provider(provider)
    entered, release = threading.Event(), threading.Event()
    original = raw_cli_executor._QueueRelay._run

    def held(relay):
        original(relay)
        entered.set()
        assert release.wait(15)

    monkeypatch.setattr(raw_cli_executor._QueueRelay, "_run", held)
    service = AgentService(
        db=h.runtime._agent_runs_db,
        registry=registry,
        runtime_capacity=h.runtime.execution_capacity,
        chat_call=ScriptedChat(
            [fence("shell_exec", {"command": "printf shared-owner"}), "done"]
        ),
    )
    task = asyncio.create_task(
        asyncio.to_thread(
            service.run_turn,
            conversation_id="shared-model",
            messages=[{"role": "user", "content": "run"}],
            config=AgentConfig(
                model="test", system_prompt="test", allowed_tools=("shell_exec",)
            ),
            api_endpoint="llama_cpp",
            should_cancel=lambda: False,
        )
    )
    try:
        await until(lambda: entered.is_set() or task.done())
        assert entered.is_set(), task.result() if task.done() else "no raw helper"
        assert not task.done()
        snapshot = h.runtime.execution_capacity.snapshot()
        assert snapshot.tool_workers == 1
        assert len(snapshot.executions) == 1
        assert snapshot.child_executions == 0
        assert (
            h.runtime._agent_runs_db.list_runs(
                "shared-model", agent_kind="local_command"
            )
            == []
        )
    finally:
        release.set()
        await task
    assert h.runtime.execution_capacity.snapshot().executions == ()


@pytest.mark.parametrize("failed_write", ["rollback", "false", "cleanup"])
async def test_unproven_required_terminal_keeps_real_worker_and_storage(
    raw_host, failed_write
):
    """Qualify permanent retention in an owned subprocess, then end that process."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    if os.environ.get("TASK8_RAW_UNPROVEN_MODE") != failed_write:
        receipt = raw_host.tmp_path / "retention.json"
        env = dict(
            os.environ,
            TASK8_RAW_UNPROVEN_MODE=failed_write,
            TASK8_RAW_RETENTION_RECEIPT=str(receipt),
        )
        result = await asyncio.to_thread(
            subprocess.run,
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                f"{__file__}::test_unproven_required_terminal_keeps_real_worker_and_storage[{failed_write}]",
                "--basetemp=" + str(raw_host.tmp_path / "child"),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=40,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        assert json.loads(receipt.read_text()) == {"mode": failed_write, "held": True}
        return
    h = raw_host
    db = h.runtime._agent_runs_db
    original = db.set_terminal_with_step
    calls = []

    def fail(run_id, status, result, step):
        calls.append(run_id)
        if failed_write == "false":
            db.set_status(run_id, "cancelled")
            assert original(run_id, status, result, step) is False
            return False
        with db.transaction() as conn:
            conn.execute("UPDATE agent_runs SET status='done' WHERE id=?", (run_id,))
            raise RuntimeError("actual rollback")

    if failed_write == "cleanup":
        from tldw_chatbook.Tools import raw_cli_executor

        cleanup = raw_cli_executor._cleanup_tree

        def unavailable(tree, *, terminate):
            assert cleanup(tree, terminate=terminate)
            return False

        raw_cli_executor._cleanup_tree = unavailable
    else:
        db.set_terminal_with_step = fail
    assert h.controller.start_user_command(stash())
    call = next(iter(h.runtime.raw_cli_source.pending))
    work, _ = h.workers.pop()
    worker = threading.Thread(target=work, daemon=True)
    worker.start()
    await until(lambda: call.operation._stopping)
    assert worker.is_alive() and not call.retired.is_set()
    if failed_write == "cleanup":
        assert not call.physical_proven and call.required_proven
        assert call.owner._cleanup_unproven
        row = db.get_run(call.run_id)
        assert row["status"] == "done" and len(row["steps"]) == 2
        assert row["steps"][1]["args"]["cleanup_proven"] is False
    else:
        assert call.physical_proven and not call.required_proven
        assert len(calls) == 1
        row = db.get_run(calls[0])
        assert row["status"] == ("cancelled" if failed_write == "false" else "running")
        assert len(row["steps"]) == 1
    assert h.runtime.execution_capacity.snapshot().tool_workers == 1
    assert call.snapshot.root.is_dir()
    with h.runtime.worktree_activity.prelaunch(
        footprint=WorktreeFootprint(opaque=True)
    ) as claim:
        assert claim is None
    Path(os.environ["TASK8_RAW_RETENTION_RECEIPT"]).write_text(
        json.dumps({"mode": failed_write, "held": True})
    )
    # This branch intentionally has no retirement mechanism; end only this owned process.
    os._exit(0)


async def test_source_loss_after_actual_run_creation_banks_original_draft(raw_host):
    h = raw_host
    draft = stash()
    db = h.runtime._agent_runs_db
    original = db.append_steps

    def replaced(*args, **kwargs):
        original(*args, **kwargs)
        h.app.console_runtime = object()

    db.append_steps = replaced
    assert h.controller.start_user_command(draft)
    work, _ = h.workers.pop()
    try:
        await asyncio.to_thread(work)
        rows = db.list_runs(h.session.persisted_conversation_id)
        assert len(rows) == 1 and rows[0]["status"] == "error"
        assert rows[0]["steps"][1]["args"]["terminal_state"] == "refused"
        assert h.runtime.raw_cli_refusal_stash_bank[h.session.id] == [draft]
        assert h.runtime.execution_capacity.snapshot().executions == ()
    finally:
        h.app.console_runtime = h.runtime


@pytest.mark.parametrize("failure", ["before_create", "after_create", "after_step"])
async def test_partial_start_observes_only_original_row_before_refusal(
    raw_host, failure
):
    h = raw_host
    db = h.runtime._agent_runs_db
    name = "append_steps" if failure == "after_step" else "create_run"
    original = getattr(db, name)
    attempted = []

    def fail(*args, **kwargs):
        attempted.append((args, kwargs))
        if failure != "before_create":
            original(*args, **kwargs)
        raise RuntimeError("original start failure")

    setattr(db, name, fail)
    draft = stash()
    assert h.controller.start_user_command(draft)
    work, _ = h.workers.pop()
    await asyncio.to_thread(work)
    rows = db.list_runs(h.session.persisted_conversation_id)
    assert len(attempted) == 1
    if failure == "before_create":
        assert rows == []
    else:
        assert len(rows) == 1
        assert rows[0]["status"] == "error"
        assert len(rows[0]["steps"]) == (1 if failure == "after_step" else 0)
    assert h.restored == [(h.session.id, draft)]
    assert h.runtime.execution_capacity.snapshot().executions == ()


async def test_actual_textual_cancel_before_callback_start_retires_without_launch(
    raw_host,
):
    from textual.app import App

    h = raw_host
    app = App()
    async with app.run_test() as pilot:

        def start(work, **kwargs):
            worker = app.run_worker(work, **kwargs)
            worker.cancel()
            return worker

        h.controller._start_worker = start
        assert h.controller.start_user_command(stash())
        await pilot.pause()
        await until(lambda: not h.runtime.raw_cli_source.pending)
        assert h.session.persisted_conversation_id is None
        assert h.runtime.execution_capacity.snapshot().executions == ()
        assert h.restored


async def test_terminal_readback_requires_canonical_step_types(raw_host):
    import copy

    h = raw_host
    assert h.controller.start_user_command(stash())
    call = next(iter(h.runtime.raw_cli_source.pending))
    work, _ = h.workers.pop()
    await asyncio.to_thread(work)
    row = h.runtime._agent_runs_db.get_run(call.run_id)
    step = copy.deepcopy(row["steps"][1])
    assert call.observe_terminal("done", step)
    step["args"]["cleanup_proven"] = 1
    assert not call.observe_terminal("done", step), (
        "JSON number is not the original boolean proof"
    )
