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


@pytest.mark.parametrize("producer", ["composer", "phone"])
@private_profile_test
async def test_a_queued_send_over_the_ask_trigger_returns_to_the_queue_not_failed(
    request, monkeypatch, tmp_path, producer
):
    """The second reply crosses the trigger while a follow-up waits in the queue.

    The follow-up drains over the Ask trigger. It must not reach the provider,
    must not leave a Failed assistant row or a dispatch-recovery panel, and
    must wait at the head of the paused queue with the reason shown.
    """
    if producer == "phone":
        from functools import partial

        from Tests.UI import test_console_compaction_hold_flow as hold_controls
        from tldw_chatbook import config

        monkeypatch.setattr(
            hold_controls,
            "_build_test_app",
            partial(
                hold_controls._build_test_app, user_data_dir=config.get_user_data_dir()
            ),
        )
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

            # Both explicit producers enter the same native queued owner.
            phone_prompt = None
            if producer == "composer":
                console._console_composer_or_none().load_draft(_FOLLOW_UP)
                await asyncio.wait_for(
                    console._send_console_message_from_visible_action(
                        session_id=session.id
                    ),
                    10,
                )
            else:
                from types import SimpleNamespace

                from Tests.Chat.test_console_received_intent_inputs import _attachment
                from Tests.Chat.test_console_received_intent_transitions import (
                    _command_owner,
                    _outcome,
                )

                store = controller.store
                store.set_session_draft(session.id, "desktop queued sentinel")
                attachment = _attachment("desktop-only")
                store.add_pending_attachment(session.id, attachment)
                store.set_session_one_shot_prefill(session.id, "desktop prefill")
                from tldw_chatbook.Chat.console_live_work import ConsoleLiveWorkLaunch

                staged = ConsoleLiveWorkLaunch.from_values(
                    source="rag",
                    title="Desktop staged source",
                    payload={"source_id": "desktop-only-rag"},
                )
                runtime.stage_console_staged_evidence(staged)
                desktop_before = (session.draft, session.draft_revision)
                commands, target, revisions = _command_owner(
                    SimpleNamespace(session=session, runtime=runtime)
                )
                turn_id = await commands.send(
                    target, _FOLLOW_UP, "phone-queued-ask", expected_revisions=revisions
                )
                queued_record = runtime._turn_custody[turn_id]
                proof = queued_record.received_intent.recovery_user_origin
                assert (await asyncio.wait_for(_outcome(queued_record), 30)).applied
                phone_prompt = controller.prompt_queue_registry._states[
                    session.id
                ].waiting[0]
                phone_request = phone_prompt.custody_request
                assert phone_request.configuration.recovery_user_origin is proof
                assert phone_request.input_origin == "command"
                recovery = runtime.ensure_archive_recovery()
                assert recovery._proofs[id(proof)] is proof
                assert id(proof) not in recovery._accepted_proofs
                accepted_proof = []
                native_accept = recovery.accept_user_proof

                def observe_accept(candidate, session_id, turn_id):
                    native_accept(candidate, session_id, turn_id)
                    if candidate is proof:
                        assert session_id == session.id
                        assert recovery._accepted_proofs[id(proof)] == (proof, turn_id)
                        assert id(proof) not in recovery._proofs
                        accepted_proof.append(turn_id)

                monkeypatch.setattr(recovery, "accept_user_proof", observe_accept)
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
            if producer == "phone":
                current_prompt = controller.prompt_queue_registry._states[
                    session.id
                ].waiting[0]
                assert current_prompt is phone_prompt
                assert current_prompt.custody_request is phone_request
                assert (session.draft, session.draft_revision) == desktop_before
                assert store.pending_attachments(session.id) == [attachment]
                assert session.one_shot_prefill == "desktop prefill"
                assert runtime.snapshot_console_staged_evidence()[0] is staged
                # Native provenance is single-use. Refused promotion ends its
                # custody even while the immutable queue request remains inert.
                assert accepted_proof == [phone_request.turn_id]
                assert id(proof) not in recovery._proofs
                assert id(proof) not in recovery._accepted_proofs
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
@pytest.mark.parametrize("producer", ("buddy", "phone"))
async def test_a_cardless_reply_over_the_ask_trigger_is_refused_before_commit(
    request, monkeypatch, tmp_path, producer
):
    """Native Buddy and authenticated phone sends refuse before adapter contact.

    The phone uses real remote capture, opaque provenance and runtime custody;
    both producers leave the held desktop draft and its revision unchanged.
    """
    if producer == "phone":
        from functools import partial

        from Tests.UI import test_console_compaction_hold_flow as hold_controls
        from tldw_chatbook import config

        monkeypatch.setattr(
            hold_controls,
            "_build_test_app",
            partial(
                hold_controls._build_test_app, user_data_dir=config.get_user_data_dir()
            ),
        )
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
            if producer == "phone":
                held = controller.store.preparation_for_session(session.id)
                assert held is not None
                controller.cancel_library_preparation(held.preparation_id)
            else:
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

            draft_revision = session.draft_revision
            reply = f"{producer} reply: thanks, continue."
            if producer == "buddy":
                result = await asyncio.wait_for(
                    controller.submit_draft(
                        reply,
                        session_id=session.id,
                        preserve_composer=True,
                    ),
                    30,
                )
            else:
                from types import SimpleNamespace

                from Tests.Chat.test_console_received_intent_transitions import (
                    _command_owner,
                    _outcome,
                )

                commands, target, revisions = _command_owner(
                    SimpleNamespace(session=session, runtime=runtime)
                )
                turn_id = await commands.send(
                    target,
                    reply,
                    "phone-compaction-refusal",
                    expected_revisions=revisions,
                )
                record = runtime._turn_custody[turn_id]
                proof = record.received_intent.recovery_user_origin
                recovery = runtime.ensure_archive_recovery()
                assert recovery._proofs[id(proof)] is proof
                result = await asyncio.wait_for(_outcome(record), 30)
                from tldw_chatbook.Chat.console_runtime import _ConsoleTurnRefusedError

                assert type(result) is _ConsoleTurnRefusedError, result
                assert result.card_held is False
                assert result.preparation_id is None
                assert id(proof) not in recovery._proofs
                assert id(proof) not in recovery._accepted_proofs
            assert session.draft_revision == draft_revision
            await console._sync_native_console_chat_ui()
            await pilot.pause()

            if producer == "phone":
                refusal_copy = result.reason
            else:
                assert not result.accepted, result
                refusal_copy = result.visible_copy
            assert refusal_copy.startswith("Not sent:"), refusal_copy
            assert "/compact" in refusal_copy
            assert len(adapter_calls) == calls
            users = [
                row.content
                for row in _rows(controller, session.id, ConsoleMessageRole.USER)
            ]
            assert reply not in users
            assert failed_before == sum(
                row.status == "failed"
                for row in _rows(controller, session.id, ConsoleMessageRole.ASSISTANT)
            )
            recoveries = runtime.recoveries_for_session(session.id)
            if producer == "phone":
                assert len(recoveries) == 1
                recovery_entry = recoveries[0]
                assert recovery_entry.turn_id == turn_id
                assert recovery_entry.draft == reply
                assert recovery_entry.reason == refusal_copy
                assert recovery_entry.preparation_id is None
            else:
                assert recoveries == ()
            assert controller.store.preparation_for_session(session.id) is None
            # The Console composer keeps the user's own held draft.
            assert controller.store.session_draft(session.id) == held_draft
            assert composer.draft_text() == held_draft
            assert "waiting for dispatch" not in _frame(host)
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()


@private_profile_test
@pytest.mark.parametrize(
    ("desktop_draft", "streaming", "late_edit"),
    (
        ("", False, False),
        ("desktop B", False, False),
        ("phone reply", False, False),
        ("desktop B", False, True),
        ("desktop B", True, False),
    ),
)
async def test_phone_ask_refusal_keeps_both_background_composers(
    request, monkeypatch, tmp_path, desktop_draft, streaming, late_edit
):
    from dataclasses import replace
    from functools import partial
    from types import SimpleNamespace

    from Tests.Chat.test_console_received_intent_transitions import (
        _command_owner,
        _outcome,
    )
    from Tests.UI import test_console_compaction_hold_flow as hold_controls
    from tldw_chatbook import config
    from tldw_chatbook.Chat.console_runtime import _ConsoleTurnRefusedError

    monkeypatch.setattr(
        hold_controls,
        "_build_test_app",
        partial(
            hold_controls._build_test_app, user_data_dir=config.get_user_data_dir()
        ),
    )
    _, database, host, adapter, adapter_calls = await _held_console(
        monkeypatch, tmp_path
    )
    console = None
    try:
        async with host.run_test(size=(100, 32)) as pilot:
            console, controller, phone, runtime, _ = await _drive_to_hold(
                host, pilot, monkeypatch, adapter, adapter_calls
            )
            store = controller.store
            held = store.preparation_for_session(phone.id)
            controller.cancel_library_preparation(held.preparation_id)
            gateway = controller.provider_gateway
            # End only the setup helper's non-streaming override; use the actual
            # native provider resolution with the target session's real setting.
            monkeypatch.setattr(
                gateway,
                "resolve_for_send",
                type(gateway).resolve_for_send.__get__(gateway),
            )
            store.replace_session_settings(
                phone.id, replace(phone.settings, streaming=streaming)
            )
            desktop = store.create_session()
            await console._sync_native_console_chat_ui()
            console._console_composer_or_none().load_draft("desktop A sentinel")
            console._session._sync_console_session_draft()
            store.set_session_draft(phone.id, desktop_draft)
            assert store.active_session_id == desktop.id
            before = tuple(
                (s.id, s.draft, s.draft_revision, s.settings_revision)
                for s in store.sessions()
            )
            rows_before = tuple(
                (m.id, m.role, m.status)
                for m in store.messages_for_session(phone.id)
                if m.role is not ConsoleMessageRole.SYSTEM
            )
            calls_before = len(adapter_calls)
            cancelled = []
            native_cancel = store.cancel_preparation
            record = None

            def observe_cancel(
                session_id, preparation_id, *, expected_state, restore_draft=True
            ):
                nonlocal before
                preparation = store.preparation_for_session(phone.id)
                assert (
                    preparation is not None
                    and preparation.preparation_id == preparation_id
                )
                assert preparation.input_draft_revision is None
                assert (
                    record.request.configuration.provider_selection.streaming
                    is streaming
                )
                if late_edit:
                    store.set_session_draft(
                        phone.id, "desktop B edited while phone prepared"
                    )
                    store.set_session_draft(desktop.id, "desktop A late sentinel")
                    store.replace_session_settings(
                        phone.id, replace(phone.settings, temperature=0.37)
                    )
                    before = tuple(
                        (s.id, s.draft, s.draft_revision, s.settings_revision)
                        for s in store.sessions()
                    )
                cancelled.append(restore_draft)
                return native_cancel(
                    session_id,
                    preparation_id,
                    expected_state=expected_state,
                    restore_draft=restore_draft,
                )

            monkeypatch.setattr(store, "cancel_preparation", observe_cancel)
            commands, target, revisions = _command_owner(
                SimpleNamespace(session=phone, runtime=runtime)
            )
            turn_id = await commands.send(
                target,
                "phone reply",
                "phone-background-refusal",
                expected_revisions=revisions,
            )
            record = runtime._turn_custody[turn_id]
            proof = record.received_intent.recovery_user_origin
            recovery = runtime.ensure_archive_recovery()
            assert recovery._proofs[id(proof)] is proof
            outcome = await asyncio.wait_for(_outcome(record), 30)
            assert type(outcome) is _ConsoleTurnRefusedError, outcome
            assert (
                outcome.reason.startswith("Not sent:") and "/compact" in outcome.reason
            )
            assert cancelled == [False]
            assert (
                tuple(
                    (s.id, s.draft, s.draft_revision, s.settings_revision)
                    for s in store.sessions()
                )
                == before
            )
            assert store.active_session_id == desktop.id
            assert len(adapter_calls) == calls_before
            assert (
                tuple(
                    (m.id, m.role, m.status)
                    for m in store.messages_for_session(phone.id)
                    if m.role is not ConsoleMessageRole.SYSTEM
                )
                == rows_before
            )
            assert store.preparation_for_session(phone.id) is None
            assert (
                id(proof) not in recovery._proofs
                and id(proof) not in recovery._accepted_proofs
            )
            entries = runtime.recoveries_for_session(phone.id)
            assert len(entries) == 1 and entries[0].turn_id == turn_id
            assert (
                entries[0].draft == "phone reply"
                and entries[0].reason == outcome.reason
            )
            assert runtime.recoveries_for_session(desktop.id) == ()
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()


@private_profile_test
async def test_background_phone_below_ask_trigger_uses_native_nonstreaming_send(
    request, monkeypatch, tmp_path
):
    from dataclasses import replace
    from functools import partial
    from types import SimpleNamespace

    from Tests.Chat.test_console_received_intent_transitions import (
        _command_owner,
        _outcome,
    )
    from Tests.UI import test_console_compaction_hold_flow as hold_controls
    from tldw_chatbook import config

    monkeypatch.setattr(
        hold_controls,
        "_build_test_app",
        partial(
            hold_controls._build_test_app, user_data_dir=config.get_user_data_dir()
        ),
    )
    _, database, host, adapter, adapter_calls = await _held_console(
        monkeypatch, tmp_path
    )
    console = None
    try:
        async with host.run_test(size=(100, 32)) as pilot:
            console, controller, phone, runtime, _ = await _ask_console(
                host, pilot, monkeypatch, adapter
            )
            gateway = controller.provider_gateway
            monkeypatch.setattr(
                gateway,
                "resolve_for_send",
                type(gateway).resolve_for_send.__get__(gateway),
            )
            store = controller.store
            store.replace_session_settings(
                phone.id, replace(phone.settings, streaming=False)
            )
            store.set_session_draft(phone.id, "desktop B untouched")
            desktop = store.create_session()
            await console._sync_native_console_chat_ui()
            console._console_composer_or_none().load_draft("desktop A untouched")
            console._session._sync_console_session_draft()
            before = tuple((s.id, s.draft, s.draft_revision) for s in store.sessions())
            commands, target, revisions = _command_owner(
                SimpleNamespace(session=phone, runtime=runtime)
            )
            turn_id = await commands.send(
                target,
                "phone short question",
                "phone-below-trigger",
                expected_revisions=revisions,
            )
            record = runtime._turn_custody[turn_id]
            proof = record.received_intent.recovery_user_origin
            recovery = runtime.ensure_archive_recovery()
            assert recovery._proofs[id(proof)] is proof
            outcome = await asyncio.wait_for(_outcome(record), 30)
            assert not isinstance(outcome, BaseException), outcome
            assert outcome.accepted
            assert len(adapter_calls) == 1
            assert adapter_calls[0]["streaming"] is False
            assert [
                m.content for m in _rows(controller, phone.id, ConsoleMessageRole.USER)
            ] == ["phone short question"]
            answers = _rows(controller, phone.id, ConsoleMessageRole.ASSISTANT)
            assert len(answers) == 1 and answers[0].content.startswith("answer-1 ")
            assert answers[0].status != "failed"
            assert (
                tuple((s.id, s.draft, s.draft_revision) for s in store.sessions())
                == before
            )
            assert store.active_session_id == desktop.id
            assert store.preparation_for_session(phone.id) is None
            assert runtime.recoveries_for_session(phone.id) == ()
            assert (
                id(proof) not in recovery._proofs
                and id(proof) not in recovery._accepted_proofs
            )
    finally:
        if console is not None:
            await console._console_runtime().dispose()
        database.close()
