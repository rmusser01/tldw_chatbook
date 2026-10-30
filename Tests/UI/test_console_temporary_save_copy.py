"""The composer Menu's Save chat row says what saving does to capture.

TASK-37811 (owner decision 2026-10-29). TASK-33621.17 gave a Temporary chat
Capture Off until it is saved and labelled the Menu's save row 'Save chat
(enables capture)', with the hint 'records no traces until saved'. A saved
chat follows its own Capture choice, else the global setting, so under a
stored conversation Capture Off or the global Capture Off saving turned
nothing on and the copy promised traces that never came.

Drives the real ChatScreen, controller, store and a file-backed database,
the real Trace Privacy and global-capture calls, and the Menu's own row.
Only the provider call is a double (nothing here sends). Kept out of
test_console_temporary_capture_flow.py so neither UI PR-gate entry nears
the 30 s per-entry budget.
"""

from __future__ import annotations

import pytest
from textual.widgets import Button

from Tests.UI.test_console_received_intent_feedback import (
    _received_console_case,
    _until,
)
from Tests.UI.test_console_temporary_capture_flow import (
    _temporary_tab,
    _turn_capture_on,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.bootstrap_profile]


async def _menu_save_row(case) -> Button:
    """Open the composer Menu the way its button does; return its Save row."""
    from tldw_chatbook.Chat.console_ephemeral import ACTION_SAVE_CHAT
    from tldw_chatbook.Widgets.Console.console_composer_menu_modal import (
        ConsoleComposerMenuModal,
    )

    await case.console._open_console_composer_menu()
    assert await _until(
        lambda: isinstance(case.host.screen, ConsoleComposerMenuModal), 5
    )
    return case.host.screen.query_one(
        f"#console-composer-menu-{ACTION_SAVE_CHAT}", Button
    )


async def _save_from_menu(case, session_id: str) -> tuple[tuple[str, str], bool]:
    """Read the Save row's label and hint, press it, and return them with
    the Capture the saved chat then resolves to."""
    row = await _menu_save_row(case)
    copy = (str(row.label), str(row.tooltip))
    row.press()
    assert await _until(lambda: not case.store.session_is_ephemeral(session_id), 15)
    snapshot = case.controller.capture_policy_snapshot(session_id)
    return copy, snapshot.effective_capture_enabled


async def test_save_chat_copy_says_what_saving_does_to_capture(monkeypatch):
    """TASK-37811 (owner decision 2026-10-29).

    'Save chat (enables capture)' and its hint 'records no traces until
    saved' were shown in every Temporary chat. Saving hands Capture back to
    the chat's own setting, so with a stored conversation Capture Off or the
    global Capture Off it turned nothing on, and the copy promised traces
    that never came. Each case reads the real Menu row, then saves through
    it and reads what the saved chat resolves to.
    """
    from tldw_chatbook.Chat.console_chat_controller import (
        CapturePolicyMutationStatus,
    )

    enables = (
        "Save chat (enables capture)",
        "Not saved locally; records no traces until saved",
    )
    stays_off = ("Save chat", "Not saved locally; traces stay off once saved")
    seen = {}
    async with _received_console_case(
        monkeypatch, "temporary-save-copy", durable=True
    ) as case:
        controller = case.controller

        # Global capture On, nothing chosen for the chat: saving turns it on.
        normal = await _temporary_tab(case)
        seen["global On"] = await _save_from_menu(case, normal)

        # A stored conversation Capture Off is saved with the chat.
        chose_off = await _temporary_tab(case)
        revision = controller.capture_policy_snapshot(chose_off).policy_revision
        result = await controller.replace_conversation_trace_privacy(
            chose_off,
            capture_enabled=False,
            pii_redaction_enabled=None,
            expected_policy_revision=revision,
        )
        assert result.status is CapturePolicyMutationStatus.APPLIED, result
        seen["conversation Off"] = await _save_from_menu(case, chose_off)

        # Global capture Off (the Trace Privacy dialog's Global default).
        global_off = await _temporary_tab(case)
        snapshot = controller.capture_policy_snapshot(global_off)
        result = controller.apply_global_capture_settings(
            enabled=False,
            detail=snapshot.global_detail,
            expected_config_generation=snapshot.config_generation,
            expected_policy_revision=snapshot.policy_revision,
        )
        assert result.status is CapturePolicyMutationStatus.APPLIED, result
        assert controller.capture_policy_snapshot(global_off).enabled is False
        seen["global Off"] = await _save_from_menu(case, global_off)

        # The chat's own Capture On outranks the global Off once saved, so
        # here saving does turn capture on and the row says so.
        chose_on = await _temporary_tab(case)
        await _turn_capture_on(case, chose_on)
        seen["global Off, conversation On"] = await _save_from_menu(case, chose_on)

    # The copy, and what saving then did, for every case at once.
    assert seen == {
        "global On": (enables, True),
        "conversation Off": (stays_off, False),
        "global Off": (stays_off, False),
        "global Off, conversation On": (enables, True),
    }
