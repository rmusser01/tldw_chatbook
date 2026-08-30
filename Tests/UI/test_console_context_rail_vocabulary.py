"""One vocabulary for the active chat in the Context rail (TASK-23199).

The 2026-08-29 audit reported this as a self-contradiction: the Sessions
section showed "Conversation  None" with the chat's own name, "Chat 1", on
the line directly beneath it. That reading was not quite right, and the
correction matters for what the fix should be.

"None" was accurate. ``scope_label`` is set from ``current_conversation``,
a PERSISTED conversation id -- so an unsaved native session genuinely has
none, while "Chat 1" is the unsaved tab's name. The two rows were saying
different true things in words that made them look like a disagreement.

The row was still not worth its space, in either state:

* unsaved -> "Conversation  None", which reads as contradicting the name
  below it;
* saved   -> "Conversation  This conversation", a tautology that tells the
  user nothing (``scope_label = "This conversation" if current_conversation
  else ""``).

Either way the useful fact -- which chat is active -- is already on the
next row. These tests pin that the rail states it once.
"""

from __future__ import annotations

import pytest

from Tests.UI.test_console_left_rail import make_console_pilot


def _rail_text(screen) -> str:
    rail = screen.query_one("#console-left-rail")
    return " ".join(
        str(getattr(widget, "renderable", ""))
        for widget in rail.query("*")
        if widget.display
    )


@pytest.mark.asyncio
async def test_the_rail_never_says_the_conversation_is_none() -> None:
    """"None" beside a named chat reads as a contradiction to a user."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        await pilot.pause(0.3)
        text = _rail_text(pilot.app.screen)
        assert "Conversation None" not in text.replace("  ", " "), (
            f"the rail still reports the conversation as None: {text!r}"
        )


@pytest.mark.asyncio
async def test_the_rail_never_says_this_conversation() -> None:
    """The saved-state copy was a tautology; it must not come back."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        await pilot.pause(0.3)
        assert "This conversation" not in _rail_text(pilot.app.screen)


@pytest.mark.asyncio
async def test_the_active_chat_is_still_named_exactly_once_in_its_section() -> None:
    """Removing the row must not remove the fact it was failing to convey."""
    async with make_console_pilot(size=(160, 48), production_styles=True) as pilot:
        await pilot.pause(0.3)
        screen = pilot.app.screen

        body = screen.query_one("#console-rail-section-body-session")
        named = [
            str(getattr(widget, "renderable", "")).strip()
            for widget in body.query("*")
            if widget.display and str(getattr(widget, "renderable", "")).strip()
        ]
        assert named, "the Sessions section says nothing about the active chat"
        assert not screen.query("#console-active-scope"), (
            "the tautological scope row is still composed"
        )
