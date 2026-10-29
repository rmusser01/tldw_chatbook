"""A Temporary chat's sends, end to end on the stock received-intent path.

TASK-33621.17 / TASK-33621.18, found live on dev 38c9d2edcf (2026-10-29):

- Save & Send saved the chat and then refused the send: "Durable Console
  Library policy no longer matches acceptance". The send's Library ceiling
  was frozen while the chat was temporary, so the durable commit carried
  the temporary policy identity into a chat that now had a durable one.
  The controller-level test never froze a ceiling (it builds no turn
  configuration), so it stayed green.
- Every send in a Temporary tab paused on that card (owner decision
  2026-10-27: a Temporary chat resolves Capture Off instead).

These drive the real ChatScreen, runtime, controller, store and a
file-backed database. Only the provider call is a double.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_console_received_intent_feedback import (
    _received_console_case,
    _send,
    _until,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus
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
        assert await _until(lambda: "Temporary · traces off" in _painted(case), 5)
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
