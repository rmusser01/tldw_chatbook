"""A Temporary chat's sends, end to end on the stock received-intent path.

TASK-33621.17 / TASK-33621.18, found live on dev 38c9d2edcf (2026-10-29):

- Save & Send saved the chat and then refused the send: "Durable Console
  Library policy no longer matches acceptance". The send's Library ceiling
  was frozen while the chat was temporary, so the durable commit carried
  the temporary policy identity into a chat that now had a durable one.
  The controller test for this path froze the same ceiling and was already
  red on dev, but no CI lane runs its file, and a plain local run of it
  stops earlier on an unselected profile.
- Every send in a Temporary tab paused on that card (owner decision
  2026-10-27: a Temporary chat resolves Capture Off instead).
- With Capture On chosen, the paused send also left an "unsent turn" on the
  shelf; after Send without capture and the reply the shelf still offered
  Restore for it, and the composer still held the answered prompt. The turn
  is now kept off the shelf, not dropped: it holds the send's attachments,
  which the card's Cancel gives back (pre-merge review).

These drive the real ChatScreen, runtime, controller, store and a
file-backed database. Only the provider call is a double.
"""

from __future__ import annotations

import threading
from uuid import uuid4

import pytest
from textual.widgets import Button

from Tests.Chat.test_console_automatic_library_preparation import _pending_image
from Tests.UI.test_console_hook_review_send_freeze import _key
from Tests.UI.test_console_library_auto_outcome import _shelf_offers_unsent_turn
from Tests.UI.test_console_received_intent_feedback import (
    _install_received_wire_reply,
    _received_console_case,
    _send,
    _until,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
from tldw_chatbook.Chat.console_configuration_preparation import (
    capture_runtime_configuration_owned,
)
from tldw_chatbook.Chat.console_turn_context import ConsoleTurnCustodyRequest
from tldw_chatbook.Chat.console_turn_preparation import (
    ConsolePreparationPauseKind,
    ConsoleTurnPreparationState,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]

_REPLY = "received intent reply"


async def _temporary_tab(case) -> str:
    """Open a Temporary tab the way the tab strip does and wait for it."""
    console, store = case.console, case.store
    first = case.session.id
    case.probe.release.set()  # nothing here waits on the original read
    console.action_new_temporary_console_tab()
    assert await _until(
        lambda: (
            store.active_session_id not in (None, first)
            and store.session_is_ephemeral(store.active_session_id)
            and console._console_visible_draft_session_id == store.active_session_id
        ),
        15,
    )
    return store.active_session_id


async def _turn_capture_on(case, session_id: str) -> None:
    """Choose Capture On for this still-temporary chat (the dialog's call)."""
    controller = case.controller
    revision = controller.capture_policy_snapshot(session_id).policy_revision
    await controller.replace_conversation_trace_privacy(
        session_id,
        capture_enabled=True,
        pii_redaction_enabled=None,
        expected_policy_revision=revision,
    )
    assert controller.capture_policy_snapshot(session_id).effective_capture_enabled


def _paused_for_capture(case, session_id: str) -> bool:
    preparation = case.store.preparation_for_session(session_id)
    return (
        preparation is not None
        and preparation.state is ConsoleTurnPreparationState.PAUSED
        and preparation.pause_kind is ConsolePreparationPauseKind.TEMPORARY_CAPTURE
    )


async def _send_draft(case, text: str) -> None:
    composer = case.console._console_composer_or_none()
    composer.load_draft(text)
    composer.focus()
    assert await _until(lambda: case.host.focused is composer, 5)
    _send(case, "enter")


def _replied(case, session_id: str) -> bool:
    return any(
        message.content == _REPLY
        for message in case.store.messages_for_session(session_id)
    )


def _painted(case) -> str:
    return "\n".join(
        strip.text for strip in case.host.screen._compositor.render_strips()
    )


async def test_a_temporary_chat_sends_with_traces_off_and_never_pauses(monkeypatch):
    """TASK-33621.17 AC#1-#3 (owner decision 2026-10-27).

    On dev every send in a Temporary tab stopped on 'Save chat to capture
    this send' and showed Blocked on the rail, the transcript edge and the
    shelf. A Temporary chat now resolves Capture Off, so its send goes out;
    saving the chat hands Capture back to the normal setting.
    """
    async with _received_console_case(
        monkeypatch, "temporary-default", durable=True
    ) as case:
        _install_received_wire_reply(monkeypatch, case)
        session_id = await _temporary_tab(case)
        await _send_draft(case, "Plain temporary send")

        assert await _until(lambda: _replied(case, session_id), 20), (
            case.controller.run_state_for(session_id)
        )
        assert len(case.provider_calls) == 1
        assert case.store.session_is_ephemeral(session_id)
        assert not _paused_for_capture(case, session_id)
        runtime = case.console._console_runtime()
        assert runtime.recoveries_for_session(session_id) == ()
        snapshot = case.controller.capture_policy_snapshot(session_id)
        assert snapshot.effective_capture_enabled is False
        # Inherit, not a stored Off: saving must not pin Capture Off.
        assert snapshot.conversation_capture_enabled is None
        # TASK-37840 (owner decision 2026-10-30) added "not saved".
        assert await _until(
            lambda: "Temporary · not saved · traces off" in _painted(case), 5
        )
        painted = _painted(case)
        assert "Blocked" not in painted and "Not sent" not in painted

        assert await _until(
            lambda: (
                case.store.dispatch_recovery_for_session(session_id) is None
                and case.controller.run_state_for(session_id).status
                is ConsoleRunStatus.COMPLETED
            ),
            15,
        ), (
            case.store.dispatch_recovery_for_session(session_id),
            case.controller.run_state_for(session_id),
        )
        case.console._session._dispatch_promote_console_temporary_session()

        assert await _until(lambda: not case.store.session_is_ephemeral(session_id), 15)
        assert case.controller.capture_policy_snapshot(
            session_id
        ).effective_capture_enabled
        assert await _until(lambda: "traces off" not in _painted(case), 5)


async def test_save_and_send_saves_the_chat_and_its_send_gets_a_reply(monkeypatch):
    """AC#4/#5: Save & Send reaches the provider through the real commit."""
    async with _received_console_case(
        monkeypatch, "temporary-save-send", durable=True
    ) as case:
        _install_received_wire_reply(monkeypatch, case)
        session_id = await _temporary_tab(case)
        await _turn_capture_on(case, session_id)
        await _send_draft(case, "Keep this one")
        assert await _until(lambda: _paused_for_capture(case, session_id), 15)
        save = case.console.query_one("#console-trace-save-send", Button)
        assert await _until(lambda: save.display and not save.disabled, 10)
        assert case.provider_calls == []

        save.press()

        assert await _until(lambda: _replied(case, session_id), 20), (
            case.controller.run_state_for(session_id)
        )
        assert len(case.provider_calls) == 1
        assert case.store.session_is_ephemeral(session_id) is False
        assert await _until(
            lambda: case.store.preparation_for_session(session_id) is None, 10
        )
        run = case.controller.run_state_for(session_id)
        assert run.status is ConsoleRunStatus.COMPLETED, run
        assert "Validating" not in run.visible_copy
        assert "Preparing" not in run.visible_copy


def _unsent_turns(runtime, session_id: str) -> list:
    """Every unsent turn the runtime keeps for the chat, offered or not."""
    return [
        entry
        for entry in runtime._turn_recoveries.values()
        if entry.session_id == session_id
    ]


async def test_a_capture_pause_leaves_no_shelf_entry_and_no_answered_prompt(
    monkeypatch,
):
    """TASK-33621.18 AC#1-#3, #5, #6 on the stock received path."""
    async with _received_console_case(
        monkeypatch, "temporary-capture-shelf", durable=True
    ) as case:
        session_id = await _temporary_tab(case)
        await _turn_capture_on(case, session_id)
        sent = "Answer me once"
        await _send_draft(case, sent)
        assert await _until(lambda: _paused_for_capture(case, session_id), 15)
        send_without = case.console.query_one("#console-trace-send-without", Button)
        assert await _until(
            lambda: send_without.display and not send_without.disabled, 10
        )
        composer = case.console._console_composer_or_none()
        runtime = case.console._console_runtime()
        # The live precondition: a composer send keeps its text until accepted.
        assert composer.draft_text() == sent
        # The refused send has left the runtime's intake custody, which is
        # when dev recorded its unsent turn (the shelf then shows it).
        assert await _until(
            lambda: all(
                row.session_id != session_id for row in runtime._turn_custody.values()
            ),
            10,
        )
        # AC#1: the card owns this pause; the shelf offers no second entry.
        # The runtime keeps the turn, held off the shelf, for the card's Cancel.
        assert [entry.draft for entry in _unsent_turns(runtime, session_id)] == [sent]
        assert runtime.recoveries_for_session(session_id) == ()
        assert not _shelf_offers_unsent_turn(case.console)

        send_without.press()

        assert await _until(lambda: _replied(case, session_id), 20), (
            case.controller.run_state_for(session_id)
        )
        assert len(case.provider_calls) == 1
        # AC#2/#3: nothing left to restore once the turn was answered (the
        # card's action settles it when its send returns, just after the reply).
        assert await _until(lambda: _unsent_turns(runtime, session_id) == [], 10), (
            _unsent_turns(runtime, session_id)
        )
        assert runtime.recoveries_for_session(session_id) == ()
        assert await _until(lambda: not _shelf_offers_unsent_turn(case.console), 5)
        # AC#6: and the composer no longer holds the answered prompt.
        assert await _until(lambda: composer.draft_text() == "", 5), (
            composer.draft_text()
        )
        assert case.store.session_draft(session_id) == ""


@pytest.mark.parametrize("entry", ["composer", "voice"])
async def test_cancel_on_a_capture_pause_gives_back_the_message_and_its_image(
    monkeypatch, entry
):
    """TASK-33621.18 review: the card's Cancel must lose nothing.

    The runtime's unsent turn is the only thing holding the send's staged
    image, and the text of a send that never sat in the composer (the voice
    handoff, ``submit_accepted_voice_turn``, admits its transcript straight
    into custody). Keeping the turn off the shelf by not recording it at all
    meant Cancel returned neither: the image was gone with no Restore.
    """
    from tldw_chatbook.Chat import attachment_core
    from tldw_chatbook.Chat import console_chat_controller as controller_module

    monkeypatch.setattr(attachment_core, "vision_block_reason", lambda *a, **k: None)
    monkeypatch.setattr(controller_module, "vision_block_reason", lambda *a, **k: None)
    async with _received_console_case(
        monkeypatch, f"temporary-capture-cancel-{entry}", durable=True
    ) as case:
        session_id = await _temporary_tab(case)
        await _turn_capture_on(case, session_id)
        store, runtime = case.store, case.console._console_runtime()
        composer = case.console._console_composer_or_none()
        image = _pending_image("pic.png", b"\x89PNG\r\n\x1a\n-pic")
        assert store.add_pending_attachment(session_id, image)
        sent = "With a picture"
        if entry == "composer":
            await _send_draft(case, sent)
        else:
            assert composer.draft_text() == ""
            runtime.accept_turn(
                ConsoleTurnCustodyRequest(
                    turn_id=str(uuid4()),
                    session_id=session_id,
                    draft=sent,
                    draft_revision=store.session_input_snapshot(session_id).draft_revision,
                    configuration=await capture_runtime_configuration_owned(
                        case.controller, session_id, require_current=lambda: None
                    ),
                    attachment_ids=(image.attachment_id,),
                )
            )
        assert await _until(lambda: _paused_for_capture(case, session_id), 15)
        await case.console._sync_native_console_chat_ui()  # draw the card
        # The send took the image with it, and left the turn to its card.
        assert store.pending_attachments(session_id) == []
        assert await _until(
            lambda: all(
                row.session_id != session_id for row in runtime._turn_custody.values()
            ),
            10,
        )
        assert runtime.recoveries_for_session(session_id) == ()
        assert not _shelf_offers_unsent_turn(case.console)
        cancel = case.console.query_one("#console-trace-cancel", Button)
        assert await _until(lambda: cancel.display and not cancel.disabled, 10)
        # A voice turn's text never sat in the composer.
        assert composer.draft_text() == (sent if entry == "composer" else "")

        cancel.press()

        assert await _until(lambda: composer.draft_text() == sent, 10), (
            composer.draft_text()
        )
        assert await _until(
            lambda: [item.display_name for item in store.pending_attachments(session_id)]
            == ["pic.png"],
            10,
        ), store.pending_attachments(session_id)
        assert store.preparation_for_session(session_id) is None
        assert _unsent_turns(runtime, session_id) == []
        assert not _shelf_offers_unsent_turn(case.console)
        assert case.provider_calls == []


async def _replace_with_authored_draft(case, text: str) -> int:
    """Type through the terminal pump, including a fresh identical/empty edit."""
    composer = case.console._console_composer_or_none()
    session_id = case.store.active_session_id
    before = case.store.session_input_snapshot(session_id).draft_revision
    composer.focus()
    assert await _until(lambda: case.host.focused is composer, 5)
    _key(case.host, "ctrl+a")
    _key(case.host, "backspace")
    assert await _until(lambda: composer.draft_text() == "", 5)
    for char in text:
        _key(case.host, char, char)
    assert await _until(
        lambda: (
            composer.draft_text() == text
            and case.store.session_draft(session_id) == text
            and case.store.session_input_snapshot(session_id).draft_revision > before
        ),
        5,
    )
    return case.store.session_input_snapshot(session_id).draft_revision


@pytest.mark.parametrize("action", ["send-without", "save-send"])
async def test_capture_card_send_preserves_newer_identical_authored_draft(
    monkeypatch, action
):
    from tldw_chatbook.Chat import Chat_Functions

    entered, release = threading.Event(), threading.Event()
    async with _received_console_case(
        monkeypatch, f"capture-new-identical-{action}", durable=True
    ) as case:
        reply = Chat_Functions.chat_api_call

        def held_reply(**kwargs):
            entered.set()
            assert release.wait(20), "test did not release the provider"
            return reply(**kwargs)

        monkeypatch.setattr(Chat_Functions, "chat_api_call", held_reply)
        session_id = await _temporary_tab(case)
        await _turn_capture_on(case, session_id)
        sent = "Again"
        await _send_draft(case, sent)
        assert await _until(lambda: _paused_for_capture(case, session_id), 15)
        runtime = case.console._console_runtime()
        assert await _until(lambda: bool(_unsent_turns(runtime, session_id)), 10)
        button = case.console.query_one(f"#console-trace-{action}", Button)
        assert await _until(lambda: button.display and not button.disabled, 10)
        try:
            button.press()
            assert await _until(entered.is_set, 45), (case.controller.run_state_for(session_id), case.store.preparation_for_session(session_id), case.provider_calls)
            revision = await _replace_with_authored_draft(case, sent)
        finally:
            release.set()
        assert await _until(lambda: _replied(case, session_id), 20)
        assert await _until(lambda: _unsent_turns(runtime, session_id) == [], 10)
        composer = case.console._console_composer_or_none()
        assert composer.draft_text() == sent
        assert case.store.session_draft(session_id) == sent
        assert case.store.session_input_snapshot(session_id).draft_revision == revision
        assert len(case.provider_calls) == 1


@pytest.mark.parametrize("new_draft", ["", "Again"])
@pytest.mark.parametrize("edit_time", ["before-cancel", "during-finish"])
async def test_capture_cancel_preserves_newer_draft_and_keeps_original_recoverable(
    monkeypatch, new_draft, edit_time
):
    from tldw_chatbook.Chat import attachment_core
    from tldw_chatbook.Chat import console_chat_controller as controller_module

    monkeypatch.setattr(attachment_core, "vision_block_reason", lambda *a, **k: None)
    monkeypatch.setattr(controller_module, "vision_block_reason", lambda *a, **k: None)
    async with _received_console_case(
        monkeypatch, f"capture-cancel-new-{bool(new_draft)}-{edit_time}", durable=True
    ) as case:
        session_id = await _temporary_tab(case)
        await _turn_capture_on(case, session_id)
        store, runtime = case.store, case.console._console_runtime()
        image = _pending_image("original.png", b"\x89PNG\r\n\x1a\n-original")
        assert store.add_pending_attachment(session_id, image)
        await _send_draft(case, "Again")
        assert await _until(lambda: _paused_for_capture(case, session_id), 15)
        assert await _until(
            lambda: not runtime.has_custodied_turns(session_id), 10
        )
        assert store.pending_attachments(session_id) == []
        revisions = []
        if edit_time == "before-cancel":
            revisions.append(await _replace_with_authored_draft(case, new_draft))
        else:
            finished = case.console._session._read_trace_recovery_finished()

            async def finish_after_edit():
                revisions.append(await _replace_with_authored_draft(case, new_draft))
                await finished()

            monkeypatch.setattr(
                case.console._session, "_read_trace_recovery_finished",
                lambda: finish_after_edit,
            )
        cancel = case.console.query_one("#console-trace-cancel", Button)
        assert await _until(lambda: cancel.display and not cancel.disabled, 10)
        cancel.press()
        assert await _until(lambda: store.preparation_for_session(session_id) is None, 10)
        assert await _until(
            lambda: bool(runtime.recoveries_for_session(session_id))
            or not _unsent_turns(runtime, session_id),
            10,
        )
        assert case.console._console_composer_or_none().draft_text() == new_draft
        assert store.session_draft(session_id) == new_draft
        assert revisions
        assert store.session_input_snapshot(session_id).draft_revision == revisions[-1]
        assert store.pending_attachments(session_id) == []
        recovery, = _unsent_turns(runtime, session_id)
        assert recovery.draft == "Again"
        assert recovery.attachments == (image,)
        assert runtime.recoveries_for_session(session_id)
        assert case.provider_calls == []
