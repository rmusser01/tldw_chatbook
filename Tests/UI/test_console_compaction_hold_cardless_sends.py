"""TASK-33621.4 AC#2: sends that cannot show the hold card stop before commit.

A composer send over the Ask compaction trigger is held before commit and
offered the hold card (TASK-34350); Continue, Regenerate and Retry stop
before commit too. A *queued* follow-up and a Persona Buddy reply still went
through to commit: the assistant row was marked Failed, a System row said
"Conversation context reached its compaction threshold", and a durable chat
showed "Response accepted; waiting for dispatch." with Retry response /
Discard (reproduced live 2026-10-29 with a queued follow-up: evidence
bd5/U23/42). Now both are refused before anything is committed: a queued
entry goes back to the head of the queue, which pauses on "Start refused",
and a Buddy reply keeps its text in Buddy with the reason.

Real app, ChatScreen, runtime, controller, store, ChaChaNotes DB, gateway and
request preparation, in a fresh real profile per case; only the provider
adapter call is doubled.
"""

from __future__ import annotations

import asyncio
import threading

import pytest
from textual.widgets import Button, Static

from Tests.private_profile import private_profile_test
from Tests.UI.test_console_compaction_hold_flow import (
    _SUMMARY,
    _ask_console,
    _drive_to_hold,
    _frame,
    _held_console,
    _send,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleMessageRole
from tldw_chatbook.Chat.console_prompt_queue import (
    PromptQueueMode,
    PromptQueuePauseReason,
)
from tldw_chatbook.UI.Console_Modules.provider_continuation_recovery import (
    TraceCallRecoveryCallout,
)

_FOLLOW_UP = "queued follow-up: and the last step?"


def _rows(controller, session_id, role):
    return [
        message
        for message in controller.store.messages_for_session(session_id)
        if message.role is role
    ]


@private_profile_test
async def test_a_queued_send_over_the_ask_trigger_returns_to_the_queue_not_failed(
    request, monkeypatch, tmp_path
):
    """The second reply crosses the trigger while a follow-up waits in the queue.

    The follow-up drains over the Ask trigger. It must not reach the provider,
    must not leave a Failed assistant row or a dispatch-recovery panel, and
    must wait at the head of the paused queue with the reason shown.
    """
    app, database, host, _adapter, adapter_calls = await _held_console(
        monkeypatch, tmp_path
    )
    entered = threading.Event()
    release = threading.Event()

    def adapter(**kwargs):
        adapter_calls.append(kwargs)
        if "Summarize" in str(kwargs.get("system_message") or ""):
            return {"choices": [{"message": {"content": _SUMMARY}}]}
        if len(adapter_calls) == 2:
            entered.set()
            assert release.wait(20), "the follow-up was never queued"
            # One long answer carries the chat past its 80% trigger.
            content = "answer-2 " + "detail " * 900
        else:
            content = f"answer-{len(adapter_calls)}"
        return {"choices": [{"message": {"content": content}}]}

    console = None
    try:
        async with host.run_test(size=(100, 32)) as pilot:
            console, controller, session, runtime, runtime_tasks = await _ask_console(
                host, pilot, monkeypatch, adapter
            )
            await _send(console, host, runtime_tasks, "first question: hello.")
            assert len(adapter_calls) == 1
            second = asyncio.create_task(
                _send(console, host, runtime_tasks, "second question: explain it.")
            )
            for _ in range(200):
                if entered.is_set():
                    break
                await asyncio.sleep(0.05)
            assert entered.is_set(), "the second send never reached the provider"

            # The live turn turns Send into Queue: queue through the screen.
            console._console_composer_or_none().load_draft(_FOLLOW_UP)
            await asyncio.wait_for(
                console._send_console_message_from_visible_action(
                    session_id=session.id
                ),
                10,
            )
            queued = controller.prompt_queue_registry.snapshot(session.id)
            assert queued.total_count == 1, queued
            release.set()
            await asyncio.wait_for(second, 60)
            for _ in range(50):
                await console._sync_native_console_chat_ui()
                await pilot.pause(0.1)
                snapshot = controller.prompt_queue_registry.snapshot(session.id)
                if snapshot.mode is PromptQueueMode.PAUSED:
                    break

            # Never sent: no third provider call, and no transcript row.
            assert len(adapter_calls) == 2
            users = [
                row.content
                for row in _rows(controller, session.id, ConsoleMessageRole.USER)
            ]
            assert _FOLLOW_UP not in users
            statuses = [
                row.status
                for row in _rows(controller, session.id, ConsoleMessageRole.ASSISTANT)
            ]
            assert len(statuses) == 2 and "failed" not in statuses, statuses
            assert runtime.recoveries_for_session(session.id) == ()
            # It waits at the head of the paused queue, with the reason.
            snapshot = controller.prompt_queue_registry.snapshot(session.id)
            assert snapshot.total_count == 1, snapshot
            assert snapshot.mode is PromptQueueMode.PAUSED
            assert snapshot.pause_reason is PromptQueuePauseReason.DISPATCH_REFUSED
            assert snapshot.entries[0].preview.startswith("queued follow-up")
            systems = [
                row.content
                for row in _rows(controller, session.id, ConsoleMessageRole.SYSTEM)
            ]
            refusal = systems[-1]
            assert refusal.startswith("Not sent from the queue:"), systems
            assert "/compact" in refusal
            frame = _frame(host)
            assert "waiting for dispatch" not in frame
            summary = console.query_one("#console-prompt-queue-summary", Static)
            for _ in range(50):
                if "Start refused" in str(summary.content):
                    break
                await console._sync_native_console_chat_ui()
                await pilot.pause(0.1)
            assert "Start refused" in str(summary.content), summary.content
            shown = {str(b.label) for b in console.query(Button) if b.region.area}
            assert not shown & {"Retry response", "Discard"}
    finally:
        release.set()
        if console is not None:
            await console._console_runtime().dispose()
        database.close()


@pytest.mark.parametrize("sigil_draft", ["$no-such-skill", "/no-such-command"])
@private_profile_test
async def test_a_sigil_composer_send_without_capture_is_refused_before_commit(
    request, monkeypatch, tmp_path, sigil_draft
):
    """A "$" or "/" composer draft with exchange capture off has no preparation.

    With no preparation there is no hold card. On dev the send was committed,
    its assistant row marked Failed under the compaction System row (review
    2026-10-30). Now it is refused before commit, and the composer keeps it.
    ``_held_console`` turns capture off with the real setting.
    """
    app, database, host, adapter, adapter_calls = await _held_console(
        monkeypatch, tmp_path
    )
    console = None
    try:
        async with host.run_test(size=(100, 32)) as pilot:
            console, controller, session, runtime, held_draft = await _drive_to_hold(
                host, pilot, monkeypatch, adapter, adapter_calls
            )
            card = console.query_one(TraceCallRecoveryCallout)
            card.query_one("#console-trace-cancel", Button).press()
            for _ in range(100):
                await console._sync_native_console_chat_ui()
                await pilot.pause(0.1)
                if (
                    not card.display
                    and controller.store.preparation_for_session(session.id) is None
                ):
                    break
            assert not card.display
            draft = f"{sigil_draft} {held_draft}"
            calls = len(adapter_calls)
            statuses_before = [
                row.status
                for row in _rows(controller, session.id, ConsoleMessageRole.ASSISTANT)
            ]
            composer = console._console_composer_or_none()
            console._session._sync_console_session_draft()
            composer.load_draft(draft)

            async def press_send():
                await asyncio.wait_for(
                    console._send_console_message_from_visible_action(
                        session_id=session.id
                    ),
                    10,
                )

            if sigil_draft.startswith("/"):
                # An unknown "/name" sends as text on the Enter after its hint.
                for _ in range(3):
                    await press_send()
                    await pilot.pause()
                    if console._console_unknown_send_armed == draft:
                        break
                assert console._console_unknown_send_armed == draft
            await press_send()
            for _ in range(100):
                await console._sync_native_console_chat_ui()
                await pilot.pause(0.1)
                if not runtime._turn_custody:
                    break
            assert not runtime._turn_custody, "the send never settled"

            assert len(adapter_calls) == calls
            users = [
                row.content
                for row in _rows(controller, session.id, ConsoleMessageRole.USER)
            ]
            assert draft not in users, users[-2:]
            statuses = [
                row.status
                for row in _rows(controller, session.id, ConsoleMessageRole.ASSISTANT)
            ]
            assert statuses == statuses_before, statuses
            systems = [
                row.content
                for row in _rows(controller, session.id, ConsoleMessageRole.SYSTEM)
            ]
            assert systems[-1].startswith("Not sent:"), systems[-2:]
            assert "/compact" in systems[-1]
            # The composer's ordinary refusal shelf holds the draft: "Not
            # sent: <reason>" (one prefix) with Restore, never the dispatch
            # panel's Retry response.
            (entry,) = runtime.recoveries_for_session(session.id)
            assert (entry.draft, entry.reason) == (draft, systems[-1])
            summary = console.query_one("#console-prompt-queue-summary", Static)
            for _ in range(50):
                if summary.region.width:
                    break
                await console._sync_native_console_chat_ui()
                await pilot.pause(0.1)
            assert str(summary.render()).startswith(
                "Not sent: this chat reached its compaction"
            ), summary.render()
            frame = _frame(host)
            assert "waiting for dispatch" not in frame
            shown = {str(b.label) for b in console.query(Button) if b.region.area}
            assert "Retry response" not in shown
            assert "Restore" in shown, shown
            console.query_one("#console-prompt-queue-manage", Button).press()
            for _ in range(50):
                await console._sync_native_console_chat_ui()
                await pilot.pause(0.1)
                if composer.draft_text() == draft:
                    break
            assert composer.draft_text() == draft
            assert runtime.recoveries_for_session(session.id) == ()
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()


@private_profile_test
async def test_a_buddy_reply_over_the_ask_trigger_is_refused_before_commit(
    request, monkeypatch, tmp_path
):
    """A Persona Buddy reply (``preserve_composer=True``) cannot show the card.

    Buddy calls ``submit_draft(text, session_id=..., preserve_composer=True)``
    (UI/Navigation/buddy_conversation.py) and keeps its own draft when the
    result is not accepted. Over the Ask trigger the reply must be refused
    with the reason, before commit, and leave the Console composer alone.
    """
    app, database, host, adapter, adapter_calls = await _held_console(
        monkeypatch, tmp_path
    )
    console = None
    try:
        async with host.run_test(size=(100, 32)) as pilot:
            console, controller, session, runtime, held_draft = await _drive_to_hold(
                host, pilot, monkeypatch, adapter, adapter_calls
            )
            card = console.query_one(TraceCallRecoveryCallout)
            card.query_one("#console-trace-cancel", Button).press()
            for _ in range(100):
                await console._sync_native_console_chat_ui()
                await pilot.pause(0.1)
                if (
                    not card.display
                    and controller.store.preparation_for_session(session.id) is None
                ):
                    break
            assert not card.display
            composer = console._console_composer_or_none()
            assert composer.draft_text() == held_draft
            calls = len(adapter_calls)
            failed_before = sum(
                row.status == "failed"
                for row in _rows(controller, session.id, ConsoleMessageRole.ASSISTANT)
            )

            result = await asyncio.wait_for(
                controller.submit_draft(
                    "buddy reply: thanks, continue.",
                    session_id=session.id,
                    preserve_composer=True,
                ),
                30,
            )
            await console._sync_native_console_chat_ui()
            await pilot.pause()

            assert not result.accepted, result
            assert result.visible_copy.startswith("Not sent:"), result.visible_copy
            assert "/compact" in result.visible_copy
            assert len(adapter_calls) == calls
            users = [
                row.content
                for row in _rows(controller, session.id, ConsoleMessageRole.USER)
            ]
            assert "buddy reply: thanks, continue." not in users
            assert failed_before == sum(
                row.status == "failed"
                for row in _rows(controller, session.id, ConsoleMessageRole.ASSISTANT)
            )
            assert runtime.recoveries_for_session(session.id) == ()
            assert controller.store.preparation_for_session(session.id) is None
            # The Console composer keeps the user's own held draft.
            assert controller.store.session_draft(session.id) == held_draft
            assert composer.draft_text() == held_draft
            assert "waiting for dispatch" not in _frame(host)
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()
