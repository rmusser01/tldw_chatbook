"""Sessions folds into Conversations (TASK-23199 AC1).

The 2026-08-29 audit's S2-C: the rail presented Sessions, Workspaces,
Conversations and Chats at once, and the user's single chat appeared in two
of them. After TASK-23199's first pass removed the tautological
"Conversation" status row, Sessions was reduced to a header plus one row
naming the active chat -- which the Conversations browser already showed,
marked "active session", on a row it renders as selected.

A list with its current item marked is one concept, not two. Sessions is
gone; Conversations carries the active-chat summary.

The legacy-restore test below is the one that matters for existing users:
``session_open`` was not only this section's flag, it was the migration seed
TASK-14810 used when it split one mixed Session body into three sections. A
payload written before that split carries only ``session_open``, and it must
still restore Workspaces and Conversations correctly now that the flag's own
section is gone.
"""

from __future__ import annotations

import pytest

from Tests.UI.test_console_left_rail import make_console_pilot


@pytest.mark.asyncio
async def test_the_rail_has_no_separate_sessions_section() -> None:
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        screen = pilot.app.screen
        assert not screen.query("#console-rail-section-header-session")
        assert not screen.query("#console-rail-section-body-session")
        assert not screen.query("#console-session-context")


@pytest.mark.asyncio
async def test_the_active_chat_is_still_named_in_the_rail() -> None:
    """Deleting the section must not delete the fact it carried."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        await pilot.pause(0.3)
        screen = pilot.app.screen

        body = screen.query_one("#console-rail-section-body-conversations")
        # Rows are Buttons, so their text is on `.label`, not `.renderable`.
        text = " ".join(
            f"{getattr(widget, 'renderable', '')} {getattr(widget, 'label', '')}"
            for widget in body.query("*")
            if widget.display
        )
        assert "Chat 1" in text, (
            f"the active chat is no longer named anywhere in the rail: {text!r}"
        )


@pytest.mark.unit
def test_the_section_vocabulary_lost_exactly_one_entry():
    from tldw_chatbook.Chat.console_rail_state import CONSOLE_RAIL_SECTION_IDS
    from tldw_chatbook.UI.Console_Modules.left_rail import (
        CONTEXT_SECTION_DESCRIPTORS,
    )

    assert "session" not in CONSOLE_RAIL_SECTION_IDS
    assert CONSOLE_RAIL_SECTION_IDS == (
        "workspace",
        "conversations",
        "model",
        "details",
        "agent",
        "character",
    )
    descriptor_ids = tuple(d.section_id for d in CONTEXT_SECTION_DESCRIPTORS)
    assert "session" not in descriptor_ids
    assert len(descriptor_ids) == len(CONSOLE_RAIL_SECTION_IDS)


@pytest.mark.unit
def test_preferences_no_longer_carry_a_session_flag():
    from tldw_chatbook.Chat.console_rail_state import (
        ConsoleRailPreferences,
        serialize_console_rail_preferences,
    )

    assert not hasattr(ConsoleRailPreferences(), "session_open")
    assert "session_open" not in serialize_console_rail_preferences(
        ConsoleRailPreferences()
    )


@pytest.mark.unit
def test_a_pre_split_payload_still_restores_workspaces_and_conversations():
    """``session_open`` was the TASK-14810 migration seed, not just a flag.

    Before that split there was ONE mixed Session body; its stored flag seeds
    all three sections it became. Removing the Sessions section must not
    strand users whose payload predates the split.
    """
    from tldw_chatbook.Chat.console_rail_state import (
        coerce_console_rail_preferences,
    )

    opened = coerce_console_rail_preferences({"session_open": True})
    assert opened.workspace_open is True
    assert opened.conversations_open is True

    closed = coerce_console_rail_preferences({"session_open": False})
    assert closed.workspace_open is False
    assert closed.conversations_open is False


@pytest.mark.unit
def test_an_explicit_modern_flag_still_beats_the_legacy_seed():
    from tldw_chatbook.Chat.console_rail_state import (
        coerce_console_rail_preferences,
    )

    prefs = coerce_console_rail_preferences(
        {"session_open": True, "workspace_open": False}
    )
    assert prefs.workspace_open is False
    assert prefs.conversations_open is True


@pytest.mark.unit
def test_a_payload_without_the_legacy_seed_uses_the_shipped_defaults():
    from tldw_chatbook.Chat.console_rail_state import (
        ConsoleRailPreferences,
        coerce_console_rail_preferences,
    )

    defaults = ConsoleRailPreferences()
    prefs = coerce_console_rail_preferences({"left_open": True})
    assert prefs.workspace_open is defaults.workspace_open
    assert prefs.conversations_open is defaults.conversations_open
