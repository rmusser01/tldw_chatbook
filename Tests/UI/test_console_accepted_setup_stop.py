"""Accepted Stop boundaries on the real received-intent path (PR3320)."""

from __future__ import annotations

import asyncio
import sqlite3
import sys
import threading
import traceback
from pathlib import Path

import pytest

from Tests.UI.test_console_received_intent_feedback import (
    _received_console_case,
    _send,
    _until,
)
from Tests.UI.test_console_send_admission_off_pump import _allow_second_turns
from Tests.UI.test_console_temporary_capture_flow import _temporary_tab
from tldw_chatbook import config as config_module
from tldw_chatbook.Chat.console_chat_models import (
    ConsoleDispatchRecoveryKind,
    ConsoleMessageRole,
    ConsoleRunStatus,
)
from tldw_chatbook.Chat.console_turn_preparation import ConsoleTurnPreparationState

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


@pytest.mark.parametrize(
    "window",
    [
        "hook",
        "memory",
        "create-chain",
        "temporary-create-chain",
        "temporary-compose",
        "temporary-dispatch",
        "postwrite",
        "prewrite",
    ],
)
async def test_accepted_setup_stop_retires_exact_owner(monkeypatch, window):
    assert config_module.save_setting_to_cli_config(
        "console", "exchange_capture", False
    )
    async with _received_console_case(
        monkeypatch, f"accepted-stop-{window}", durable=True
    ) as case:
        _allow_second_turns(case.host)
        case.probe.release.set()
        controller, store = case.controller, case.store
        temporary = window.startswith("temporary")
        session_id = await _temporary_tab(case) if temporary else case.session.id
        composer = case.console._console_composer_or_none()
        held, release = asyncio.Event(), asyncio.Event()
        composed = []
        chain_release, chain_returned = threading.Event(), threading.Event()
        loop = asyncio.get_running_loop()
        ledger = controller._agent_bridge.runs_db.automatic_work
        original_chain = ledger.create_chain

        def create_chain(*args, **kwargs):
            if window.endswith("create-chain") and not held.is_set():
                loop.call_soon_threadsafe(held.set)
                assert chain_release.wait(20)
            result = original_chain(*args, **kwargs)
            chain_returned.set()
            return result

        monkeypatch.setattr(ledger, "create_chain", create_chain)
        chain_tasks = []
        original_retain = controller._retain_maintenance_task

        def retain(coroutine):
            task = original_retain(coroutine)
            if (
                controller._active_stream_tasks.get(session_id)
                is asyncio.current_task()
            ):
                chain_tasks.append(task)
            return task

        monkeypatch.setattr(controller, "_retain_maintenance_task", retain)
        outcomes = []
        original_finish = case.runtime._finish_custodied_turn

        def finish(task, **kwargs):
            outcomes.append(
                "cancelled"
                if task.cancelled()
                else type(task.exception()).__name__
                if task.exception() is not None
                else "returned"
            )
            return original_finish(task, **kwargs)

        monkeypatch.setattr(case.runtime, "_finish_custodied_turn", finish)
        original_hook = controller.hook_admission_reason
        original_memory = controller._apply_conversation_memory_preflight
        original_compose = controller._compose_agent_request_providers
        original_profile = controller._personal_context_builder

        async def hold():
            held.set()
            await release.wait()

        async def hook():
            if (
                window == "hook"
                and not held.is_set()
                and store.dispatch_recovery_for_session(session_id) is not None
            ):
                await hold()
            return await original_hook()

        async def memory(**kwargs):
            if (
                window == "memory"
                and not held.is_set()
                and store.dispatch_recovery_for_session(session_id) is not None
            ):
                await hold()
            return await original_memory(**kwargs)

        async def compose(**kwargs):
            composed.append(True)
            if (
                window in {"temporary-compose", "postwrite", "prewrite"}
                and not held.is_set()
            ):
                await hold()
            return await original_compose(**kwargs)

        async def profile(*args, **kwargs):
            if window == "temporary-dispatch" and composed and not held.is_set():
                await hold()
            return await original_profile(*args, **kwargs)

        monkeypatch.setattr(controller, "hook_admission_reason", hook)
        monkeypatch.setattr(controller, "_apply_conversation_memory_preflight", memory)
        monkeypatch.setattr(controller, "_compose_agent_request_providers", compose)
        monkeypatch.setattr(controller, "_personal_context_builder", profile)
        original_settle = store.settle_accepted_durable_turn
        settlements = []

        def settle(*args, **kwargs):
            if window == "prewrite":
                settlements.append(False)
                raise RuntimeError("terminal write refused before persistence")
            result = original_settle(*args, **kwargs)
            settlements.append(result)
            if window == "postwrite":
                assert result
                raise RuntimeError("finite cleanup failed after actual terminal write")
            return result

        monkeypatch.setattr(store, "settle_accepted_durable_turn", settle)

        def ended():
            return not any(
                row.session_id == session_id
                for row in case.runtime._turn_custody.values()
            )

        composer.load_draft("Stop this accepted turn")
        composer.focus()
        assert await _until(
            lambda: store.session_draft(session_id) == "Stop this accepted turn", 5
        )
        _send(case, "enter")
        try:
            # Windows native storage admission has measured up to 48.296s.
            # This is whole-turn readiness, not the Stop/settlement deadline.
            assert await _until(held.is_set, 60 if sys.platform == "win32" else 20)
            recovery = store.dispatch_recovery_for_session(session_id)
            assert recovery is not None
            assistant_id = recovery.assistant_message_id
            preparation = store.preparation_for_session(session_id)
            assert preparation is not None
            assert controller._ordinary_native_commit_tasks(session_id) == ()
            assert case.provider_calls == []
            if window in {"hook", "memory"}:
                assert session_id not in controller._active_stream_tasks
            if not temporary:
                with sqlite3.connect(config_module.get_chachanotes_db_path()) as db:
                    accepted_version = db.execute(
                        "SELECT version FROM messages WHERE id=?", (assistant_id,)
                    ).fetchone()[0]
            if window.endswith("create-chain"):
                assert preparation.state is ConsoleTurnPreparationState.ACCEPTED
                assert recovery.kind is (
                    ConsoleDispatchRecoveryKind.EPHEMERAL_ACCEPTED
                    if temporary
                    else ConsoleDispatchRecoveryKind.ACCEPTED
                )
                assert recovery.preparation_id == preparation.preparation_id
                assert recovery.runtime_active and not recovery.recovery_needed
                assert not recovery.in_flight
                assert (
                    controller._active_assistant_message_ids[session_id] == assistant_id
                )
                task = controller._active_stream_tasks[session_id]
                assert (
                    controller._active_submit_preparations[task]
                    == preparation.preparation_id
                )
                assert session_id not in controller._active_cancel_events
                assert not chain_returned.is_set()
                assert len(chain_tasks) == 1
                chain_task = chain_tasks[0]
                assert chain_task in controller._maintenance_background
                assert not chain_task.done()
            assert controller.stop_active_run()
            chain_release.set()
            for _ in range(2):
                controller.stop_active_run()
                await asyncio.sleep(0)
            assert await _until(ended, 20)
            assert outcomes == ["returned"]
            if window.endswith("create-chain"):
                assert await _until(
                    lambda: chain_task.done()
                    and chain_task not in controller._maintenance_background,
                    20,
                )
                assert chain_returned.is_set()
                assert not chain_task.cancelled() and chain_task.exception() is None
                assert chain_task not in controller._maintenance_background
            assert case.provider_calls == []
            if window == "prewrite":
                assert settlements == [False]
                assert store.preparation_for_session(session_id) is preparation
                assert store.dispatch_recovery_for_session(session_id) is not None
                assert (
                    controller.run_state_for(session_id).status
                    is ConsoleRunStatus.BLOCKED
                )
                return
            assert store.get_message(assistant_id).status == "stopped"
            assert (
                controller.run_state_for(session_id).status is ConsoleRunStatus.STOPPED
            )
            assert store.dispatch_recovery_for_session(session_id) is None
            assert store.preparation_for_session(session_id) is None
            assert session_id not in controller._active_cancel_events
            assert (
                sum(
                    row.content == "Response stopped by user."
                    for row in store.all_messages_for_session(session_id)
                )
                == 1
            )
            if temporary:
                assert store.session_is_ephemeral(session_id)
                assert all(
                    row.persisted_message_id is None
                    for row in store.all_messages_for_session(session_id)
                )
                with sqlite3.connect(config_module.get_chachanotes_db_path()) as db:
                    assert (
                        db.execute(
                            "SELECT COUNT(*) FROM messages WHERE id=?", (assistant_id,)
                        ).fetchone()[0]
                        == 0
                    )
                    assert (
                        db.execute(
                            "SELECT COUNT(*) FROM console_dispatch_checkpoints WHERE assistant_message_id=?",
                            (assistant_id,),
                        ).fetchone()[0]
                        == 0
                    )
            else:
                assert settlements == [True]
                with sqlite3.connect(config_module.get_chachanotes_db_path()) as db:
                    assert (
                        db.execute(
                            "SELECT version FROM messages WHERE id=?", (assistant_id,)
                        ).fetchone()[0]
                        == accepted_version + 1
                    )
                    assert (
                        db.execute(
                            "SELECT COUNT(*) FROM console_dispatch_checkpoints WHERE assistant_message_id=?",
                            (assistant_id,),
                        ).fetchone()[0]
                        == 0
                    )
            composer.load_draft("Next send after accepted Stop")
            composer.focus()
            assert await _until(
                lambda: store.session_draft(session_id)
                == "Next send after accepted Stop",
                5,
            )
            _send(case, "enter")
            reached_provider = await _until(
                lambda: len(case.provider_calls) == 1,
                60 if sys.platform == "win32" else 45,
            )
            if not reached_provider:
                # Content-free evidence before teardown can release or cancel work.
                stacks = {
                    thread: [
                        (Path(frame.filename).name, frame.name, frame.lineno)
                        for frame in traceback.extract_stack(top)
                    ]
                    for thread, top in sys._current_frames().items()
                }
                from tldw_chatbook.Backup_Recovery import storage_admission

                acquisitions = [
                    {
                        "thread": attempt.thread.ident,
                        "initializing": attempt.initializing_root is not None,
                        "operation": attempt.operation is not None,
                    }
                    for attempt in tuple(storage_admission._pending_acquisitions)
                ]
                tasks = [
                    [
                        (frame.f_code.co_name, frame.f_lineno)
                        for frame in task.get_stack()
                    ]
                    for task in asyncio.all_tasks()
                ]
                pytest.fail(
                    f"Follow-up provider deadline; acquisitions={acquisitions}; "
                    f"threads={stacks}; tasks={tasks}"
                )
            assert await _until(ended, 20)
            assert outcomes == ["returned", "returned"]
            assert any(
                row.role is ConsoleMessageRole.ASSISTANT
                and row.content == "received intent reply"
                for row in store.all_messages_for_session(session_id)
            )
        finally:
            release.set()
            chain_release.set()
