"""The Temporary status segment and tab decode say what a Temporary chat is.

TASK-37840 (owner decision 2026-10-30). TASK-33621.17 renamed the status
strip's Temporary segment 'Temporary · traces off', and the tab strip's ◌
tooltip decode with it. Two things went missing:

- Nothing always on screen said the chat is not saved. The chip tooltip that
  did has not been rendered since TASK-35160, so a glance at the strip told
  someone traces were off but not that the whole chat is lost on close.
- After someone chose Capture On for the Temporary chat (This conversation,
  or Next send) the segment still read 'traces off', while each send (or the
  next one) paused on 'Save chat to capture this send'.

The segment and the decode now read 'Temporary · not saved · traces off',
or 'Temporary · not saved · capture on' while the chat's next send is
captured. Measured first on the pre-fix head through the same dialog path:
both surfaces kept the old copy in all three states, and nothing repainted
them when the dialog closed.

Drives the real ChatScreen, controller, store and a file-backed database,
the Conversation Inspector's Trace Privacy dialog, and the pause card. Only
the provider call is a double. Kept out of test_console_temporary_capture_flow
so neither UI PR-gate entry nears the 30 s per-entry budget.
"""

from __future__ import annotations

import pytest
from textual.css.query import NoMatches
from textual.widgets import Button, RadioButton

from Tests.UI.test_console_received_intent_feedback import (
    _received_console_case,
    _until,
)
from Tests.UI.test_console_temporary_capture_flow import (
    _paused_for_capture,
    _replied,
    _send_draft,
    _temporary_tab,
)
from tldw_chatbook.Chat.console_chat_models import ConsoleRunStatus

pytestmark = [pytest.mark.bootstrap_profile]

TRACES_OFF = "Temporary · not saved · traces off"
CAPTURE_ON = "Temporary · not saved · capture on"


def _segment(case) -> str | None:
    """The Temporary segment as painted: the strip line that starts with it."""
    for strip in case.host.screen._compositor.render_strips():
        line = strip.text.strip()
        for label in (TRACES_OFF, CAPTURE_ON):
            if line.startswith(f"{label} · "):
                return label
        if line.startswith("Temporary"):
            return line.split(" · Provider", 1)[0]
    return None


def _decode(case, session_id: str) -> str | None:
    """The ◌ decode at the end of the chat's tab tooltip (None until mounted)."""
    try:
        tab = case.console.query_one(f"#console-session-tab-{session_id}", Button)
    except NoMatches:
        return None
    return str(tab.tooltip).rsplit(". ", 1)[-1].rstrip(".")


async def _choose_in_trace_privacy(case, scope: str, capture: str) -> None:
    """Choose Capture in Trace Privacy over the Console, Apply, and close it.

    The dialog gets the bindings the Conversation Inspector's Capture
    settings (c) hands it, and is pushed over the Console the way the
    Inspector pushes it. The Inspector itself is not opened: its Next Send
    estimate reads the workspace on the UI thread, which this harness's
    configuration-read probe records and its teardown then refuses. Closing
    the dialog resumes the Console exactly as closing the Inspector does.
    """
    from tldw_chatbook.UI.Console_Modules.capture_policy_bindings import (
        build_inspector_capture_policy_wiring,
    )
    from tldw_chatbook.UI.Screens.chat_screen import ChatScreen
    from tldw_chatbook.Widgets.Console.console_capture_policy_dialog import (
        ConsoleTracePrivacyDialog,
    )

    wiring = build_inspector_capture_policy_wiring(case.controller)
    assert wiring.target_session_id == case.store.active_session_id
    await case.host.push_screen(ConsoleTracePrivacyDialog(wiring.bindings))
    dialog = case.host.screen
    assert isinstance(dialog, ConsoleTracePrivacyDialog)
    # on_mount syncs the radios to the stored choice, then focuses Apply; a
    # choice made before that sync is overwritten by it.
    apply = dialog.query_one("#trace-privacy-apply", Button)
    assert await _until(lambda: apply.has_focus, 5)
    dialog.query_one(f"#trace-privacy-scope-{scope}", RadioButton).value = True
    wanted_scope = {"next": "next_send", "conversation": "conversation"}[scope]
    assert await _until(lambda: dialog.selected_scope.value == wanted_scope, 5), (
        dialog.selected_scope
    )
    wanted = {"on": True, "off": False, "inherit": None}[capture]
    dialog.query_one(f"#trace-privacy-capture-{capture}", RadioButton).value = True
    assert await _until(lambda: dialog.selected_capture is wanted, 5)
    apply.press()
    assert await _until(lambda: dialog.status_text == "Saved and active", 10), (
        dialog.status_text
    )
    await dialog.request_safe_cancel(source="button")
    assert await _until(lambda: isinstance(case.host.screen, ChatScreen), 10)


@pytest.mark.unit
def test_the_segment_and_the_decode_say_not_saved_and_what_capture_does():
    """The visible copy, both cases, from the pure renderers."""
    from tldw_chatbook.Chat.console_chat_store import ConsoleChatSession
    from tldw_chatbook.Widgets.Console.console_session_surface import (
        _session_tab_tooltip,
    )
    from tldw_chatbook.Widgets.Console.console_status_chips import (
        ConsoleStatusChips,
    )

    render = ConsoleStatusChips._temporary_chip_render
    assert render(True)[0] == TRACES_OFF
    assert render(True, capture_on=False)[0] == TRACES_OFF
    assert render(True, capture_on=True)[0] == CAPTURE_ON
    assert render(False)[2] is True and render(False, capture_on=True)[2] is True

    chat = ConsoleChatSession(title="Notes", ephemeral=True)
    assert _session_tab_tooltip(chat, active=True).endswith(f" {TRACES_OFF}.")
    chat.capture_enabled_override = True
    assert _session_tab_tooltip(chat, active=True).endswith(f" {CAPTURE_ON}.")
    chat.next_capture_enabled = False
    assert _session_tab_tooltip(chat, active=False).endswith(f" {TRACES_OFF}.")
    saved = ConsoleChatSession(title="Saved", capture_enabled_override=True)
    assert "Temporary" not in _session_tab_tooltip(saved, active=True)


@pytest.mark.unit
def test_capture_on_means_the_temporary_chats_next_send_is_captured():
    """'capture on' follows the controller's own resolution for a Temporary
    chat: a Next send choice wins for that send, else This conversation's,
    else Off (TASK-33621.17); the global default never applies."""
    from tldw_chatbook.Chat.console_exchange_capture import (
        temporary_chat_captures_next_send,
    )

    expected = {
        (None, None): False,
        (True, None): True,
        (False, None): False,
        (None, True): True,
        (False, True): True,
        (True, True): True,
        (None, False): False,
        (True, False): False,
        (False, False): False,
    }
    seen = {
        (conversation, next_send): temporary_chat_captures_next_send(
            conversation, next_send
        )
        for conversation, next_send in expected
    }
    assert seen == expected


@pytest.mark.asyncio
async def test_the_temporary_segment_follows_the_chats_capture_choice(monkeypatch):
    """TASK-37840 AC#2-#4 on the real dialog path.

    Pre-fix, the strip and the tab decode read 'Temporary · traces off' in
    every state below, including the two whose sends pause for capture.
    """
    async with _received_console_case(
        monkeypatch, "temporary-status-copy", durable=True
    ) as case:
        chat = await _temporary_tab(case)
        assert await _until(lambda: _segment(case) == TRACES_OFF, 5), _segment(case)
        assert await _until(lambda: _decode(case, chat) == TRACES_OFF, 5), (
            _decode(case, chat)
        )

        # This conversation On: every send pauses, so the strip says so.
        await _choose_in_trace_privacy(case, "conversation", "on")
        assert case.controller.capture_policy_snapshot(chat).effective_capture_enabled
        assert await _until(lambda: _segment(case) == CAPTURE_ON, 5), _segment(case)
        assert await _until(lambda: _decode(case, chat) == CAPTURE_ON, 5), (
            _decode(case, chat)
        )

        # Back to Inherit: a Temporary chat inherits Off.
        await _choose_in_trace_privacy(case, "conversation", "inherit")
        assert await _until(lambda: _segment(case) == TRACES_OFF, 5), _segment(case)
        assert await _until(lambda: _decode(case, chat) == TRACES_OFF, 5), (
            _decode(case, chat)
        )

        # Next send On: 'capture on' until that send is spent. Send without
        # capture spends it, and the strip goes back to traces off.
        await _choose_in_trace_privacy(case, "next", "on")
        assert await _until(lambda: _segment(case) == CAPTURE_ON, 5), _segment(case)
        assert await _until(lambda: _decode(case, chat) == CAPTURE_ON, 5), (
            _decode(case, chat)
        )
        await _send_draft(case, "Spend the one-shot")
        assert await _until(lambda: _paused_for_capture(case, chat), 15)
        send_without = case.console.query_one("#console-trace-send-without", Button)
        assert await _until(
            lambda: send_without.display and not send_without.disabled, 10
        )
        send_without.press()
        assert await _until(lambda: _replied(case, chat), 20), (
            case.controller.run_state_for(chat)
        )
        assert case.controller.capture_policy_snapshot(chat).next_capture_enabled is None
        assert await _until(lambda: _segment(case) == TRACES_OFF, 10), _segment(case)
        assert await _until(lambda: _decode(case, chat) == TRACES_OFF, 10), (
            _decode(case, chat)
        )
        assert await _until(
            lambda: (
                case.store.dispatch_recovery_for_session(chat) is None
                and case.controller.run_state_for(chat).status
                is ConsoleRunStatus.COMPLETED
            ),
            15,
        ), case.controller.run_state_for(chat)
